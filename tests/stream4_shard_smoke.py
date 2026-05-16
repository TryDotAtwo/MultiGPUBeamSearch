from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import beam_engine


def pack_meta(lo: int, hi: int, parent_idx: int, score_key: int, route: int) -> bytes:
    return struct.pack("<QQQII", lo, hi, parent_idx, score_key, route)


def unpack_meta(raw: bytes, idx: int) -> dict[str, int]:
    lo, hi, parent_idx, score_key, route = struct.unpack_from("<QQQII", raw, idx * 32)
    return {
        "lo": lo,
        "hi": hi,
        "parent_idx": parent_idx,
        "score_key": score_key,
        "route": route,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Stream4 shard smoke")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    ext = beam_engine.build_extension(verbose=False)
    device = torch.device("cuda", 0)

    threshold = 100
    hash_a = (0x101, 0xAAA)
    hash_b = (0x202, 0xBBB)
    hash_c = (0x303, 0xCCC)
    hash_d = (0x404, 0xDDD)
    # clean + dirty input region; one threshold-pruned item; no shard cap.
    metas = [
        pack_meta(*hash_a, 10, 80, 0x010101),   # worse duplicate
        pack_meta(*hash_b, 20, 90, 0x020202),   # tie candidate, worse parent
        pack_meta(*hash_a, 9, 70, 0x030303),    # best hash_a by score
        pack_meta(*hash_b, 19, 90, 0x040404),   # best hash_b by parent tie
        pack_meta(*hash_c, 30, 200, 0x050505),  # threshold drop
        pack_meta(*hash_d, 40, 10, 0x060606),   # keep
    ]
    input_count = len(metas)
    survivor_shard = torch.tensor(np.frombuffer(b"".join(metas), dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    stream4_key_a = torch.zeros((input_count * 16,), dtype=torch.uint8, device=device)
    stream4_key_b = torch.zeros_like(stream4_key_a)
    stream4_val_a = torch.zeros((input_count * 32,), dtype=torch.uint8, device=device)
    stream4_val_b = torch.zeros_like(stream4_val_a)
    compact_count = torch.zeros((1,), dtype=torch.int32, device=device)

    ext.v6_stream4_threshold_compact(
        survivor_shard,
        stream4_key_a,
        stream4_val_a,
        compact_count,
        input_count,
        threshold,
    )
    torch.cuda.synchronize()
    compact_n = int(compact_count.cpu()[0])
    assert compact_n == 5

    temp_bytes = int(ext.v6_stream4_sort_temp_bytes(compact_n))
    temp_storage = torch.empty((temp_bytes,), dtype=torch.uint8, device=device)
    ext.v6_stream4_sort_pairs(temp_storage, stream4_key_a, stream4_key_b, stream4_val_a, stream4_val_b, compact_n)
    torch.cuda.synchronize()

    clean_tmp = torch.zeros((input_count * 32,), dtype=torch.uint8, device=device)
    new_clean_count = torch.zeros((1,), dtype=torch.int32, device=device)
    ext.v6_stream4_dedup_sorted(stream4_key_b, stream4_val_b, clean_tmp, new_clean_count, compact_n)
    torch.cuda.synchronize()
    new_clean_n = int(new_clean_count.cpu()[0])
    assert new_clean_n == 3

    clean_count = torch.tensor([2], dtype=torch.int32, device=device)
    dirty_count = torch.tensor([4], dtype=torch.int32, device=device)
    processing_flag = torch.tensor([1], dtype=torch.uint8, device=device)
    ext.v6_stream4_write_clean(survivor_shard, clean_tmp, clean_count, dirty_count, processing_flag, new_clean_n)
    torch.cuda.synchronize()

    assert int(clean_count.cpu()[0]) == 3
    assert int(dirty_count.cpu()[0]) == 0
    assert int(processing_flag.cpu()[0]) == 0

    raw = survivor_shard.cpu().numpy().tobytes()
    result = [unpack_meta(raw, i) for i in range(new_clean_n)]
    by_hash = {(m["lo"], m["hi"]): m for m in result}
    assert by_hash[hash_a]["score_key"] == 70
    assert by_hash[hash_a]["parent_idx"] == 9
    assert by_hash[hash_b]["score_key"] == 90
    assert by_hash[hash_b]["parent_idx"] == 19
    assert by_hash[hash_b]["route"] == 0x040404
    assert by_hash[hash_d]["score_key"] == 10
    assert hash_c not in by_hash
    assert len(result) == 3

    print("STREAM4_SHARD_SMOKE_OK")


if __name__ == "__main__":
    main()
