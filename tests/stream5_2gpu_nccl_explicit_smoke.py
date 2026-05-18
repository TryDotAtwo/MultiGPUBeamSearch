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


WORLD_SIZE_REQUIRED = 2
RECORD_BYTES = 32
PRODUCTION_B_MICRO = 8192
PRODUCTION_K_EXPAND_TILE = PRODUCTION_B_MICRO * 24

assert PRODUCTION_B_MICRO == 8192
assert PRODUCTION_K_EXPAND_TILE == 196608


def pack_meta(lo: int, hi: int, parent_idx: int, score_key: int, route: int) -> bytes:
    return struct.pack("<QQQII", lo, hi, parent_idx, score_key, route)


def make_cfg(rank: int) -> dict:
    cfg = beam_engine.make_default_config()
    cfg["world_size"] = WORLD_SIZE_REQUIRED
    cfg["rank"] = rank
    cfg["global_beam_width"] = 65536
    cfg["b_micro"] = PRODUCTION_B_MICRO
    cfg["score_ring_depth"] = 1
    cfg["net_ring_depth"] = 1
    cfg["bucket_cap_per_peer"] = 262144
    cfg["k_expand_tile"] = PRODUCTION_K_EXPAND_TILE
    cfg["inference_parallelism"] = 1
    cfg["max_depth"] = 1
    cfg["inference_backend"] = "fullbeamnice_static"
    return cfg


def create_nccl_id(ext, rank: int) -> bytes:
    obj = [None]
    if rank == 0:
        obj[0] = bytes(ext.get_nccl_unique_id())
    dist.broadcast_object_list(obj, src=0)
    if not isinstance(obj[0], bytes):
        raise RuntimeError("NCCL unique id broadcast failed")
    return obj[0]


def expected_records(src_rank: int, dst_rank: int, count: int) -> bytes:
    return b"".join(
        pack_meta(
            lo=0x5100_0000 + src_rank * 0x10000 + dst_rank * 0x100 + j,
            hi=0x6200_0000 + src_rank * 0x10000 + dst_rank * 0x100 + j,
            parent_idx=0x7000_0000 + src_rank * 1000 + dst_rank * 10 + j,
            score_key=0x8000 + src_rank * 100 + dst_rank * 10 + j,
            route=(src_rank << 16) | (dst_rank << 8) | j,
        )
        for j in range(count)
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for explicit Stream5 2GPU NCCL smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != WORLD_SIZE_REQUIRED:
        raise SystemExit(f"WORLD_SIZE must be {WORLD_SIZE_REQUIRED}, got {world_size}")
    if rank not in (0, 1):
        raise SystemExit(f"RANK must be 0 or 1, got {rank}")
    if torch.cuda.device_count() < WORLD_SIZE_REQUIRED:
        raise SystemExit(
            "visible CUDA device count must be >=2 for explicit NCCL 2GPU smoke; "
            f"got {torch.cuda.device_count()}"
        )

    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    ext = beam_engine.build_extension(verbose=False)
    cfg = make_cfg(rank)
    buffers = beam_engine.allocate_buffers(ext, cfg)
    engine = ext.BeamEngine(cfg, buffers, "fullbeamnice_static")
    engine.init_nccl(create_nccl_id(ext, rank))

    peer = 1 - rank
    send_to_peer = 1 if rank == 0 else 3
    recv_from_peer = 3 if rank == 0 else 1
    max_records = int(cfg["bucket_cap_per_peer"])

    send_count_host = np.zeros((world_size,), dtype=np.int32)
    send_count_host[peer] = send_to_peer
    send_offset_host = np.zeros((world_size + 1,), dtype=np.int32)
    cursor = 0
    for p in range(world_size):
        send_offset_host[p] = cursor
        cursor += int(send_count_host[p])
    send_offset_host[world_size] = cursor

    recv_count_host = np.zeros((world_size,), dtype=np.int32)
    recv_count_host[peer] = recv_from_peer
    recv_offset_host = np.zeros((world_size + 1,), dtype=np.int32)
    cursor = 0
    for p in range(world_size):
        recv_offset_host[p] = cursor
        cursor += int(recv_count_host[p])
    recv_offset_host[world_size] = cursor

    send_blob = bytearray(max_records * RECORD_BYTES)
    peer_records = expected_records(rank, peer, send_to_peer)
    start = int(send_offset_host[peer]) * RECORD_BYTES
    send_blob[start:start + len(peer_records)] = peer_records

    remote_send_buffer = torch.tensor(np.frombuffer(bytes(send_blob), dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    remote_recv_buffer = torch.zeros((max_records * RECORD_BYTES,), dtype=torch.uint8, device=device)
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
    assert recv_count_after.tolist() == recv_count_host.tolist()
    assert recv_offset_after.tolist() == recv_offset_host.tolist()
    expected = expected_records(peer, rank, recv_from_peer)
    recv_start = int(recv_offset_after[peer]) * RECORD_BYTES
    assert recv_raw[recv_start:recv_start + len(expected)] == expected

    dist.barrier()
    dist.destroy_process_group()
    print(f"STREAM5_2GPU_NCCL_EXPLICIT_SMOKE_OK rank={rank} sent={send_to_peer} received={recv_from_peer}")


if __name__ == "__main__":
    main()
