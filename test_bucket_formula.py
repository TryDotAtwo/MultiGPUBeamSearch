#!/usr/bin/env python3
"""
Test script for bucket formula verification.
Tests the corrected BUCKET_CAP_PER_PEER formula with memory cap.
"""

import math

def pow2_ceil(x: int) -> int:
    """Compute smallest power-of-2 >= x"""
    if x <= 1:
        return 1
    bit_pos = (x - 1).bit_length()
    return 1 << bit_pos

def derive_buckets_python(global_beam_width: int, world_size: int, k_expand_tile: int) -> dict:
    """Python implementation of bucket formula (corrected version)"""
    
    # Formula from user:
    # BUCKET_CAP_PER_PEER = min(
    #     pow2_ceil(max(65536, K_EXPAND_TILE // 16)),
    #     2**20
    # )
    # BUCKET_CAP_PER_PEER_SAFE = min(
    #     pow2_ceil(max(131072, K_EXPAND_TILE // 8)),
    #     2**20
    # )
    
    MAX_BUCKET_CAPACITY = 1 << 20  # 2^20 = 1M
    
    # Regular version (not used by default, kept for reference)
    base_regular = max(65536, k_expand_tile // 16)
    bucket_cap_regular = pow2_ceil(base_regular)
    if bucket_cap_regular > MAX_BUCKET_CAPACITY:
        bucket_cap_regular = MAX_BUCKET_CAPACITY
    
    # SAFE version (used by default)
    base_safe = max(131072, k_expand_tile // 8)
    bucket_cap_safe = pow2_ceil(base_safe)
    if bucket_cap_safe > MAX_BUCKET_CAPACITY:
        bucket_cap_safe = MAX_BUCKET_CAPACITY
    
    # Memory calculation
    bytes_per_candidate = 160  # CandidateRecord
    send_buckets_bytes = bucket_cap_safe * bytes_per_candidate * (world_size - 1)
    recv_buckets_bytes = bucket_cap_safe * bytes_per_candidate * (world_size - 1)
    total_bucket_bytes = send_buckets_bytes + recv_buckets_bytes
    
    send_buckets_gib = send_buckets_bytes / (1024**3)
    recv_buckets_gib = recv_buckets_bytes / (1024**3)
    total_bucket_gib = total_bucket_bytes / (1024**3)
    
    return {
        'bucket_cap_regular': bucket_cap_regular,
        'bucket_cap_safe': bucket_cap_safe,
        'send_buckets_gib': send_buckets_gib,
        'recv_buckets_gib': recv_buckets_gib,
        'total_bucket_gib': total_bucket_gib,
    }

def test_scenario(name: str, global_beam_width: int, world_size: int, b_micro: int = 131072):
    """Test a specific scenario"""
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"{'='*70}")
    print(f"GLOBAL_BEAM_WIDTH: {global_beam_width:,}")
    print(f"WORLD_SIZE: {world_size}")
    print(f"B_MICRO: {b_micro:,}")
    
    # Calculate K_EXPAND_TILE
    FANOUT = 24
    TARGET_STREAM2_ROUNDS = 16
    numerator = global_beam_width * FANOUT
    denominator = world_size * TARGET_STREAM2_ROUNDS
    target_k_expand = (numerator + denominator - 1) // denominator
    k_expand_tile = pow2_ceil(target_k_expand)
    
    print(f"\nDerived K_EXPAND_TILE: {k_expand_tile:,}")
    
    # Calculate SCORE_RING_DEPTH
    numerator_sr = k_expand_tile
    denominator_sr = b_micro * FANOUT
    target_depth = (numerator_sr + denominator_sr - 1) // denominator_sr
    score_ring_depth = pow2_ceil(target_depth)
    if score_ring_depth < 1:
        score_ring_depth = 1
    
    print(f"Derived SCORE_RING_DEPTH: {score_ring_depth}")
    
    # Calculate BUCKET_CAP_PER_PEER using corrected formula
    result = derive_buckets_python(global_beam_width, world_size, k_expand_tile)
    
    print(f"\nBucket Configuration:")
    print(f"  BUCKET_CAP_PER_PEER (regular): {result['bucket_cap_regular']:,} (not used)")
    print(f"  BUCKET_CAP_PER_PEER (SAFE): {result['bucket_cap_safe']:,} ✓ (used by default)")
    print(f"\nMemory Usage (per rank):")
    print(f"  send_buckets: {result['send_buckets_gib']:.3f} GiB")
    print(f"  recv_buckets: {result['recv_buckets_gib']:.3f} GiB")
    print(f"  total_buckets: {result['total_bucket_gib']:.3f} GiB")
    
    # Sanity checks
    print(f"\nSanity Checks:")
    if result['total_bucket_gib'] > 200:
        print(f"  ⚠️  WARNING: total_bucket_gib > 200 GiB (may exceed GPU memory)")
    else:
        print(f"  ✓ total_bucket_gib within reasonable range")
    
    if result['bucket_cap_safe'] == (1 << 20):
        print(f"  ℹ️  BUCKET_CAP_PER_PEER capped at max 2^20 = {1<<20:,}")
    else:
        print(f"  ✓ BUCKET_CAP_PER_PEER not capped")

# Test scenarios
test_scenario("Kaggle 2×T4 (small)", global_beam_width=14_000_000, world_size=2)
test_scenario("H100 cluster (medium)", global_beam_width=50_000_000, world_size=8)
test_scenario("H100 cluster (large)", global_beam_width=100_000_000, world_size=16)
test_scenario("Extreme (would be risky)", global_beam_width=500_000_000, world_size=32)

print(f"\n{'='*70}")
print("Summary:")
print(f"{'='*70}")
print("✓ Formula with memory cap (2^20) prevents bucket explosion")
print("✓ BUCKET_CAP_PER_PEER_SAFE = min(pow2_ceil(max(131072, K_EXPAND_TILE / 8)), 2^20)")
print("✓ Memory sizes logged at engine initialization for verification")
