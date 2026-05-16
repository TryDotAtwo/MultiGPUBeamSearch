from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import beam_engine


MOVE_COUNT = 24


def pack_hash(lo: int, hi: int) -> bytes:
    return struct.pack("<QQ", lo, hi)


def unpack_hashes(raw: bytes, count: int) -> list[tuple[int, int]]:
    return [struct.unpack_from("<QQ", raw, i * 16) for i in range(count)]


def unpack_u64(raw: bytes, count: int) -> list[int]:
    return [struct.unpack_from("<Q", raw, i * 8)[0] for i in range(count)]


def stream3_val(score_key: int, payload_id: int) -> int:
    return (score_key << 32) | payload_id


def owner_from_hash128(hi: int, lo: int, world_size: int) -> int:
    if world_size <= 1:
        return 0
    return ((hi ^ ((lo * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)) % world_size)


def unpack_candidate_meta(raw: bytes, idx: int) -> dict[str, int]:
    off = idx * 32
    lo, hi, parent_idx, score_key, route = struct.unpack_from("<QQQII", raw, off)
    return {
        "lo": lo,
        "hi": hi,
        "parent_idx": parent_idx,
        "score_key": score_key,
        "route": route,
        "source_rank": route >> 16,
        "owner": (route >> 8) & 0xFF,
        "move": route & 0xFF,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Stream3 dedup smoke")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    ext = beam_engine.build_extension(verbose=False)
    device = torch.device("cuda", 0)

    b_micro = 2
    ring = 0
    ring_slot_count = 1
    stream3_batch_candidates = b_micro * MOVE_COUNT
    world_size = 3
    local_rank = 1
    current_threshold = 100
    parent_base_value = 1000

    hash_a = (0x101, 0xAAA)
    hash_b = (0x202, 0xBBB)
    hash_c = (0x303, 0xCCC)
    score = np.full((stream3_batch_candidates,), 999, dtype=np.int32)
    hashes = [(0, 0)] * stream3_batch_candidates

    score[0] = 50
    hashes[0] = hash_a
    score[1] = 40
    hashes[1] = hash_a
    score[2] = 40
    hashes[2] = hash_b
    score[3] = 40
    hashes[3] = hash_b
    score[24] = 10
    hashes[24] = hash_c
    score[25] = 200
    hashes[25] = (0x404, 0xDDD)

    hash_bytes = b"".join(pack_hash(lo, hi) for lo, hi in hashes)
    score_ring = torch.tensor(score, dtype=torch.int32, device=device)
    hash_ring = torch.tensor(np.frombuffer(hash_bytes, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    parent_base = torch.tensor([parent_base_value], dtype=torch.int64, device=device)
    count = torch.tensor([b_micro], dtype=torch.int32, device=device)
    stream3_key_a = torch.zeros((stream3_batch_candidates * 16,), dtype=torch.uint8, device=device)
    stream3_key_b = torch.zeros_like(stream3_key_a)
    stream3_val_a = torch.zeros((stream3_batch_candidates,), dtype=torch.int64, device=device)
    stream3_val_b = torch.zeros_like(stream3_val_a)
    compact_count = torch.zeros((1,), dtype=torch.int32, device=device)

    ext.v6_stream3_pack_threshold_compact(
        score_ring,
        hash_ring,
        parent_base,
        count,
        stream3_key_a,
        stream3_val_a,
        compact_count,
        current_threshold,
        ring,
        ring_slot_count,
        b_micro,
        stream3_batch_candidates,
    )
    torch.cuda.synchronize()
    compact_n = int(compact_count.cpu()[0])
    assert compact_n == 5
    compact_vals = stream3_val_a[:compact_n].cpu().numpy().astype(np.uint64).tolist()
    assert set(compact_vals) == {
        stream3_val(50, 0),
        stream3_val(40, 1),
        stream3_val(40, 2),
        stream3_val(40, 3),
        stream3_val(10, 24),
    }

    temp_bytes = int(ext.v6_stream3_sort_temp_bytes(compact_n))
    temp_storage = torch.empty((temp_bytes,), dtype=torch.uint8, device=device)
    ext.v6_stream3_sort_pairs(temp_storage, stream3_key_a, stream3_key_b, stream3_val_a, stream3_val_b, compact_n)
    torch.cuda.synchronize()

    unique_key = torch.zeros((stream3_batch_candidates * 16,), dtype=torch.uint8, device=device)
    unique_val = torch.zeros((stream3_batch_candidates,), dtype=torch.int64, device=device)
    unique_count = torch.zeros((1,), dtype=torch.int32, device=device)
    ext.v6_stream3_dedup_sorted(stream3_key_b, stream3_val_b, unique_key, unique_val, unique_count, compact_n)
    torch.cuda.synchronize()
    unique_n = int(unique_count.cpu()[0])
    assert unique_n == 3

    got_unique_hashes = unpack_hashes(unique_key.cpu().numpy().tobytes(), unique_n)
    got_unique_vals = unpack_u64(unique_val.cpu().numpy().tobytes(), unique_n)
    got = {h: v for h, v in zip(got_unique_hashes, got_unique_vals)}
    assert got[hash_a] == stream3_val(40, 1)
    assert got[hash_b] == stream3_val(40, 2)
    assert got[hash_c] == stream3_val(10, 24)

    expected_owners = {
        h: owner_from_hash128(h[1], h[0], world_size)
        for h in [hash_a, hash_b, hash_c]
    }
    send_counts_expected = [0] * world_size
    for h in [hash_a, hash_b, hash_c]:
        owner = expected_owners[h]
        if owner != local_rank:
            send_counts_expected[owner] += 1
    send_offsets = [0]
    for c in send_counts_expected:
        send_offsets.append(send_offsets[-1] + c)

    local_pending_buffer = torch.zeros((unique_n * 32,), dtype=torch.uint8, device=device)
    remote_send_buffer = torch.zeros((unique_n * 32,), dtype=torch.uint8, device=device)
    local_count = torch.zeros((1,), dtype=torch.int32, device=device)
    send_count = torch.zeros((world_size,), dtype=torch.int32, device=device)
    send_offset = torch.tensor(send_offsets, dtype=torch.int32, device=device)

    ext.v6_stream3_restore_split(
        unique_key,
        unique_val,
        parent_base,
        local_pending_buffer,
        remote_send_buffer,
        local_count,
        send_count,
        send_offset,
        unique_n,
        local_rank,
        world_size,
        ring,
        ring_slot_count,
        b_micro,
    )
    torch.cuda.synchronize()

    assert send_count.cpu().numpy().tolist() == send_counts_expected
    assert int(local_count.cpu()[0]) == sum(1 for h in [hash_a, hash_b, hash_c] if expected_owners[h] == local_rank)

    local_raw = local_pending_buffer.cpu().numpy().tobytes()
    remote_raw = remote_send_buffer.cpu().numpy().tobytes()
    metas = []
    for i in range(int(local_count.cpu()[0])):
        metas.append(unpack_candidate_meta(local_raw, i))
    for peer in range(world_size):
        for pos in range(send_offsets[peer], send_offsets[peer] + send_counts_expected[peer]):
            metas.append(unpack_candidate_meta(remote_raw, pos))

    by_hash = {(m["lo"], m["hi"]): m for m in metas}
    assert by_hash[hash_a]["score_key"] == 40
    assert by_hash[hash_a]["parent_idx"] == parent_base_value
    assert by_hash[hash_a]["move"] == 1
    assert by_hash[hash_b]["score_key"] == 40
    assert by_hash[hash_b]["parent_idx"] == parent_base_value
    assert by_hash[hash_b]["move"] == 2
    assert by_hash[hash_c]["score_key"] == 10
    assert by_hash[hash_c]["parent_idx"] == parent_base_value + 1
    assert by_hash[hash_c]["move"] == 0
    for h, owner in expected_owners.items():
        assert by_hash[h]["owner"] == owner
        assert by_hash[h]["source_rank"] == local_rank

    print("STREAM3_DEDUP_SMOKE_OK")


if __name__ == "__main__":
    main()
