"""CPU-side transition archive and optional checkpoint support for beam search."""

from __future__ import annotations

import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import data_loader

STATUS_CURRENT_SIZE = 0


def enabled_from_env() -> bool:
    return os.environ.get("HISTORY_BACKEND", "gpu").strip().lower() == "cpu"


def checkpoint_from_env() -> bool:
    return os.environ.get("CPU_HISTORY_CHECKPOINT", "0").strip().lower() not in {"", "0", "false", "no", "off"}


def resume_from_env() -> bool:
    return os.environ.get("RESUME_BEAMSEARCH", "0").strip().lower() not in {"", "0", "false", "no", "off"}


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class CPUHistoryArchive:
    def __init__(self, cfg: dict[str, Any], buffers: dict[str, torch.Tensor], sample_id: int | str):
        self.cfg = cfg
        self.buffers = buffers
        self.rank = int(cfg["rank"])
        self.world_size = int(cfg["world_size"])
        self.n_local = int(buffers["current_active_flags"].numel())
        self.max_depth = int(cfg["max_depth"])
        self.state_size = int(cfg["state_size_bytes"])
        self.sample_id = str(sample_id)
        self.enabled = str(cfg.get("history_backend", "gpu")).lower() == "cpu"
        self.checkpoint = bool(cfg.get("cpu_history_checkpoint", False)) or checkpoint_from_env()
        self.workers = max(1, int(os.environ.get("CPU_HISTORY_WORKERS", "1")))
        root = Path(os.environ.get("CPU_HISTORY_DIR", str(Path("runtime") / "cpu_history_archive")))
        self.root = root / f"sample_{self.sample_id}"
        self.action_table = np.frombuffer(data_loader.get_action_table_u8(), dtype=np.uint8).reshape(data_loader.FANOUT, data_loader.STATE_SIZE)
        self.config_hash = self._make_config_hash()

        self.parent_idx = np.empty((self.max_depth, self.n_local), dtype=np.int32)
        self.parent_rank = np.empty((self.max_depth, self.n_local), dtype=np.uint8)
        self.action = np.empty((self.max_depth, self.n_local), dtype=np.uint8)
        self.valid = np.zeros((self.max_depth, self.n_local), dtype=np.uint8)
        self.layer_sizes = np.zeros((self.max_depth + 1,), dtype=np.int64)
        self.transition_sizes = np.zeros((self.max_depth,), dtype=np.int64)
        self.frontier_a = np.empty((self.n_local, self.state_size), dtype=np.uint8)
        self.frontier_b = np.empty((self.n_local, self.state_size), dtype=np.uint8)
        self.frontier = self.frontier_a
        self.next_frontier = self.frontier_b
        self.frontier_size = 0
        self.current_depth = 0
        self.executor = ThreadPoolExecutor(max_workers=1) if self.checkpoint else None
        self.pending = None
        self.pending_manifest_depth: int | None = None
        self.transfer_parent_idx = None
        self.transfer_parent_rank = None
        self.transfer_action = None
        self.transfer_valid = None
        self._init_transfer_slabs()

    def finish(self) -> None:
        self._complete_pending_checkpoint()
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def start_new(self, initial_state: np.ndarray, active_on_this_rank: bool) -> None:
        if not self.enabled:
            return
        self.frontier_size = 1 if active_on_this_rank else 0
        self.frontier[: max(1, self.frontier_size)] = 0
        if active_on_this_rank:
            self.frontier[0] = np.asarray(initial_state, dtype=np.uint8)
        self.layer_sizes[0] = self.frontier_size
        self.current_depth = 0
        if self.checkpoint:
            self._write_frontier(0)
            _barrier()
            self._write_manifest(0)

    def try_resume(self, device: torch.device) -> int | None:
        if not (self.enabled and self.checkpoint and resume_from_env()):
            return None
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        depth = int(manifest["depth"])
        if depth < 0 or depth > self.max_depth:
            raise ValueError(f"checkpoint depth out of range: {depth}")
        if int(manifest.get("rank_count", manifest.get("world_size", -1))) != self.world_size:
            raise ValueError("checkpoint world size mismatch")
        if int(manifest.get("n_local", -1)) != self.n_local:
            raise ValueError("checkpoint n_local mismatch")
        if int(manifest.get("global_beam_width", -1)) != int(self.cfg["global_beam_width"]):
            raise ValueError("checkpoint global_beam_width mismatch")
        if str(manifest.get("config_hash", "")) != self.config_hash:
            raise ValueError("checkpoint config_hash mismatch")
        if not bool(manifest.get("complete", False)):
            raise ValueError("checkpoint manifest is not complete")
        self._read_history_until(depth)
        self.frontier_size = self._read_frontier(depth, self.rank, self.frontier)
        self.layer_sizes[depth] = self.frontier_size
        self.current_depth = depth
        self._upload_frontier_to_gpu(device)
        _barrier()
        return depth

    def after_step(self, depth: int, current_size: int) -> None:
        if not self.enabled:
            return
        if depth <= 0:
            return
        self._complete_pending_checkpoint()
        transition_depth = depth - 1
        if transition_depth >= self.max_depth:
            raise ValueError(f"transition depth out of range: {transition_depth}")
        n = int(current_size)
        self._copy_current_transition_layer(transition_depth, n)
        self.transition_sizes[transition_depth] = n
        self.layer_sizes[depth] = n
        self.current_depth = depth
        if self.checkpoint:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=1)
            self.pending = self.executor.submit(self._checkpoint_depth, depth, transition_depth, n)
            self.pending_manifest_depth = depth

    def history_entry(self, depth: int, local_index: int) -> dict[str, int]:
        if depth < 0 or depth >= self.max_depth:
            raise ValueError(f"history depth out of range: {depth}")
        if local_index < 0 or local_index >= self.n_local:
            raise ValueError(f"history local index out of range: {local_index}")
        return {
            "valid": int(self.valid[depth, local_index]),
            "parent_idx": int(self.parent_idx[depth, local_index]),
            "parent_rank": int(self.parent_rank[depth, local_index]),
            "action": int(self.action[depth, local_index]),
        }

    def _copy_current_transition_layer(self, depth: int, n: int) -> None:
        if n <= 0:
            return
        if self.transfer_parent_idx is None:
            self.parent_idx[depth, :n] = self.buffers["history_parent_idx"][:n].detach().cpu().numpy()
            self.parent_rank[depth, :n] = self.buffers["history_parent_rank"][:n].detach().cpu().numpy()
            self.action[depth, :n] = self.buffers["history_action"][:n].detach().cpu().numpy()
            self.valid[depth, :n] = self.buffers["history_valid"][:n].detach().cpu().numpy()
            return
        self.transfer_parent_idx[:n].copy_(self.buffers["history_parent_idx"][:n], non_blocking=True)
        self.transfer_parent_rank[:n].copy_(self.buffers["history_parent_rank"][:n], non_blocking=True)
        self.transfer_action[:n].copy_(self.buffers["history_action"][:n], non_blocking=True)
        self.transfer_valid[:n].copy_(self.buffers["history_valid"][:n], non_blocking=True)
        src = self.buffers["history_parent_idx"]
        if src.is_cuda:
            torch.cuda.current_stream(src.device).synchronize()
        self.parent_idx[depth, :n] = self.transfer_parent_idx[:n].numpy()
        self.parent_rank[depth, :n] = self.transfer_parent_rank[:n].numpy()
        self.action[depth, :n] = self.transfer_action[:n].numpy()
        self.valid[depth, :n] = self.transfer_valid[:n].numpy()

    def _init_transfer_slabs(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            self.transfer_parent_idx = torch.empty((self.n_local,), dtype=torch.int32, device="cpu", pin_memory=True)
            self.transfer_parent_rank = torch.empty((self.n_local,), dtype=torch.uint8, device="cpu", pin_memory=True)
            self.transfer_action = torch.empty((self.n_local,), dtype=torch.uint8, device="cpu", pin_memory=True)
            self.transfer_valid = torch.empty((self.n_local,), dtype=torch.uint8, device="cpu", pin_memory=True)
        except RuntimeError:
            self.transfer_parent_idx = None
            self.transfer_parent_rank = None
            self.transfer_action = None
            self.transfer_valid = None

    def _make_config_hash(self) -> str:
        payload = {
            "backend": "cpu",
            "beta": float(self.cfg.get("beta", 0.0)),
            "fanout": int(self.cfg.get("fanout", data_loader.FANOUT)),
            "global_beam_width": int(self.cfg["global_beam_width"]),
            "hash_load_factor": float(self.cfg.get("hash_load_factor", 0.0)),
            "n_local": int(self.n_local),
            "state_size_bytes": int(self.state_size),
            "world_size": int(self.world_size),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _wait_pending(self) -> None:
        if self.pending is not None:
            self.pending.result()
            self.pending = None

    def _complete_pending_checkpoint(self) -> None:
        if self.pending is None:
            return
        depth = int(self.pending_manifest_depth) if self.pending_manifest_depth is not None else int(self.current_depth)
        self._wait_pending()
        _barrier()
        self._write_manifest(depth)
        self.pending_manifest_depth = None

    def _checkpoint_depth(self, depth: int, transition_depth: int, n: int) -> None:
        self._write_transition(transition_depth, n)
        self._reconstruct_frontier(depth, n)
        self._write_frontier(depth)

    def _upload_frontier_to_gpu(self, device: torch.device) -> None:
        for name in ("beam_current", "current_active_flags", "beam_status", "history_depth_cell"):
            self.buffers[name].zero_()
        if self.frontier_size > 0:
            states = torch.from_numpy(self.frontier[: self.frontier_size]).to(device=device, non_blocking=False)
            self.buffers["beam_current"][: self.frontier_size].copy_(states)
            self.buffers["current_active_flags"][: self.frontier_size].fill_(1)
        self.buffers["beam_status"][STATUS_CURRENT_SIZE] = int(self.frontier_size)
        self.buffers["history_depth_cell"][0] = int(self.current_depth)
        torch.cuda.synchronize(device)

    def _frontier_path(self, depth: int, rank: int) -> Path:
        return self.root / f"frontier_depth{depth:04d}_rank{rank:04d}.bin"

    def _frontier_meta_path(self, depth: int, rank: int) -> Path:
        return self.root / f"frontier_depth{depth:04d}_rank{rank:04d}.json"

    def _transition_prefix(self, depth: int, rank: int) -> Path:
        return self.root / f"transition_depth{depth:04d}_rank{rank:04d}"

    def _write_frontier(self, depth: int) -> None:
        n = int(self.frontier_size)
        path = self._frontier_path(depth, self.rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".bin.tmp")
        self.frontier[:n].tofile(tmp)
        os.replace(tmp, path)
        _atomic_write_json(self._frontier_meta_path(depth, self.rank), {"depth": depth, "rank": self.rank, "count": n})

    def _read_frontier(self, depth: int, rank: int, dst: np.ndarray | None = None) -> int:
        meta = json.loads(self._frontier_meta_path(depth, rank).read_text(encoding="utf-8"))
        count = int(meta["count"])
        arr = np.fromfile(self._frontier_path(depth, rank), dtype=np.uint8, count=count * self.state_size).reshape(count, self.state_size)
        if dst is not None and count > 0:
            dst[:count] = arr
        return count

    def _load_frontier_array(self, depth: int, rank: int) -> np.ndarray:
        meta = json.loads(self._frontier_meta_path(depth, rank).read_text(encoding="utf-8"))
        count = int(meta["count"])
        return np.memmap(self._frontier_path(depth, rank), dtype=np.uint8, mode="r", shape=(count, self.state_size))

    def _write_transition(self, depth: int, n: int) -> None:
        prefix = self._transition_prefix(depth, self.rank)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix, arr in (
            ("parent_idx", self.parent_idx[depth, :n]),
            ("parent_rank", self.parent_rank[depth, :n]),
            ("action", self.action[depth, :n]),
            ("valid", self.valid[depth, :n]),
        ):
            path = Path(str(prefix) + f".{suffix}.bin")
            tmp = path.with_suffix(path.suffix + ".tmp")
            arr.tofile(tmp)
            os.replace(tmp, path)
        _atomic_write_json(Path(str(prefix) + ".json"), {"depth": depth, "rank": self.rank, "count": int(n)})

    def _read_history_until(self, depth: int) -> None:
        for d in range(depth):
            prefix = self._transition_prefix(d, self.rank)
            meta_path = Path(str(prefix) + ".json")
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n = int(meta["count"])
            self.transition_sizes[d] = n
            self.parent_idx[d, :n] = np.fromfile(Path(str(prefix) + ".parent_idx.bin"), dtype=np.int32, count=n)
            self.parent_rank[d, :n] = np.fromfile(Path(str(prefix) + ".parent_rank.bin"), dtype=np.uint8, count=n)
            self.action[d, :n] = np.fromfile(Path(str(prefix) + ".action.bin"), dtype=np.uint8, count=n)
            self.valid[d, :n] = np.fromfile(Path(str(prefix) + ".valid.bin"), dtype=np.uint8, count=n)

    def _write_manifest(self, depth: int) -> None:
        if self.rank != 0:
            return
        payload = {
            "sample_id": self.sample_id,
            "depth": int(depth),
            "rank_count": self.world_size,
            "world_size": self.world_size,
            "n_local": self.n_local,
            "global_beam_width": int(self.cfg["global_beam_width"]),
            "max_depth": self.max_depth,
            "config_hash": self.config_hash,
            "complete_layer_depth": int(depth),
            "complete": True,
        }
        _atomic_write_json(self.root / "manifest.json", payload)

    def _reconstruct_frontier(self, depth: int, n: int) -> None:
        if n <= 0:
            self.frontier_size = 0
            return
        prev_depth = depth - 1
        layer = prev_depth
        parent_idx = self.parent_idx[layer, :n]
        parent_rank = self.parent_rank[layer, :n]
        action = self.action[layer, :n]
        valid = self.valid[layer, :n] == 1
        self.next_frontier[:n] = 0
        sources: dict[int, np.ndarray] = {}
        for rank in range(self.world_size):
            if rank == self.rank:
                sources[rank] = self.frontier
            elif np.any(parent_rank[valid] == rank):
                sources[rank] = self._load_frontier_array(prev_depth, rank)

        tasks: list[tuple[np.ndarray, np.ndarray, int]] = []
        for rank, src in sources.items():
            rank_mask = valid & (parent_rank == rank)
            if not np.any(rank_mask):
                continue
            for act in np.unique(action[rank_mask]):
                idxs = np.nonzero(rank_mask & (action == act))[0]
                if idxs.size:
                    tasks.append((idxs, src, int(act)))

        def fill(task: tuple[np.ndarray, np.ndarray, int]) -> None:
            idxs, src, act = task
            self.next_frontier[idxs] = src[parent_idx[idxs]][:, self.action_table[act]]

        if self.workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(fill, tasks))
        else:
            for task in tasks:
                fill(task)

        self.frontier, self.next_frontier = self.next_frontier, self.frontier
        self.frontier_size = n
