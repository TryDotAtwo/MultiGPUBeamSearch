from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import beam_engine


def pack_meta(lo: int, hi: int, parent_idx: int, score_key: int, route: int) -> bytes:
    return struct.pack("<QQQII", lo, hi, parent_idx, score_key, route)


def make_cfg() -> dict:
    cfg = beam_engine.make_default_config()
    cfg["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    cfg["rank"] = int(os.environ.get("RANK", "0"))
    cfg["global_beam_width"] = 64
    cfg["b_micro"] = 2
    cfg["score_ring_depth"] = 1
    cfg["net_ring_depth"] = 1
    cfg["bucket_cap_per_peer"] = 8
    cfg["k_expand_tile"] = 48
    cfg["inference_parallelism"] = 1
    cfg["max_depth"] = 1
    cfg["inference_backend"] = "fullbeamnice_static"
    return cfg


def create_nccl_id(ext, cfg: dict) -> bytes:
    if cfg["world_size"] <= 1:
        return b""
    obj = [None]
    if cfg["rank"] == 0:
        obj[0] = bytes(ext.get_nccl_unique_id())
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Stream5 exchange smoke")

    cfg = make_cfg()
    if cfg["world_size"] > torch.cuda.device_count():
        print(
            "STREAM5_EXCHANGE_SMOKE_SKIPPED "
            f"world={cfg['world_size']} visible_cuda_devices={torch.cuda.device_count()} "
            "reason=NCCL_requires_distinct_visible_GPU_per_rank"
        )
        return
    device_index = int(os.environ.get("LOCAL_RANK", str(cfg["rank"] % torch.cuda.device_count()))) % torch.cuda.device_count()
    if cfg["world_size"] > 1 and not dist.is_initialized():
        torch.cuda.set_device(device_index)
        dist.init_process_group("nccl", rank=cfg["rank"], world_size=cfg["world_size"])
    os.environ["LOCAL_RANK"] = str(device_index)
    device = torch.device("cuda", device_index)

    ext = beam_engine.build_extension(verbose=False)
    buffers = beam_engine.allocate_buffers(ext, cfg)
    engine = ext.BeamEngine(cfg, buffers, "fullbeamnice_static")
    if cfg["world_size"] > 1:
        engine.init_nccl(create_nccl_id(ext, cfg))

    world = cfg["world_size"]
    rank = cfg["rank"]
    max_records = max(4, world * 2)
    send_count_host = np.zeros((world,), dtype=np.int32)
    send_offset_host = np.zeros((world + 1,), dtype=np.int32)
    send_records: list[bytes] = []

    if world == 1:
        send_count_host[0] = 2
        send_offset_host[0] = 0
        send_offset_host[1] = 2
        send_records = [
            pack_meta(0x11, 0x22, 101, 7, 0x010203),
            pack_meta(0x33, 0x44, 102, 8, 0x040506),
        ]
        recv_count_host = np.zeros((world,), dtype=np.int32)
        recv_offset_host = np.zeros((world + 1,), dtype=np.int32)
    else:
        peer = 1 - rank
        send_count_host[peer] = 2
        cursor = 0
        for p in range(world):
            send_offset_host[p] = cursor
            cursor += int(send_count_host[p])
        send_offset_host[world] = cursor
        for p in range(world):
            for j in range(int(send_count_host[p])):
                send_records.append(pack_meta(
                    lo=0x1000 + rank * 0x100 + p * 0x10 + j,
                    hi=0x2000 + rank * 0x100 + p * 0x10 + j,
                    parent_idx=rank * 1000 + p * 10 + j,
                    score_key=rank * 100 + j,
                    route=(rank << 16) | (p << 8) | j,
                ))
        recv_count_host = np.zeros((world,), dtype=np.int32)
        recv_count_host[peer] = 2
        recv_offset_host = np.zeros((world + 1,), dtype=np.int32)
        cursor = 0
        for p in range(world):
            recv_offset_host[p] = cursor
            cursor += int(recv_count_host[p])
        recv_offset_host[world] = cursor

    send_blob = b"".join(send_records)
    send_blob += bytes(max_records * 32 - len(send_blob))
    remote_send_buffer = torch.tensor(np.frombuffer(send_blob, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    remote_recv_buffer = torch.zeros((max_records * 32,), dtype=torch.uint8, device=device)
    send_count = torch.tensor(send_count_host, dtype=torch.int32, device=device)
    send_offset = torch.tensor(send_offset_host, dtype=torch.int32, device=device)
    recv_count = torch.tensor(recv_count_host, dtype=torch.int32, device=device)
    recv_offset = torch.tensor(recv_offset_host, dtype=torch.int32, device=device)

    engine.v6_stream5_exchange_candidate_meta(
        remote_send_buffer,
        remote_recv_buffer,
        send_count,
        send_offset,
        recv_count,
        recv_offset,
    )
    torch.cuda.synchronize()

    recv_count_after = recv_count.cpu().numpy()
    recv_offset_after = recv_offset.cpu().numpy()
    recv_raw = remote_recv_buffer.cpu().numpy().tobytes()

    if world == 1:
        assert recv_count_after.tolist() == [2]
        assert recv_offset_after.tolist() == [0, 2]
        assert recv_raw[:64] == send_blob[:64]
    else:
        peer = 1 - rank
        assert recv_count_after[peer] == 2
        assert recv_offset_after[peer + 1] - recv_offset_after[peer] == 2
        expected = b"".join(
            pack_meta(
                lo=0x1000 + peer * 0x100 + rank * 0x10 + j,
                hi=0x2000 + peer * 0x100 + rank * 0x10 + j,
                parent_idx=peer * 1000 + rank * 10 + j,
                score_key=peer * 100 + j,
                route=(peer << 16) | (rank << 8) | j,
            )
            for j in range(2)
        )
        start = int(recv_offset_after[peer]) * 32
        assert recv_raw[start:start + len(expected)] == expected
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()
    print(f"STREAM5_EXCHANGE_SMOKE_OK rank={rank} world={world}")


if __name__ == "__main__":
    main()
