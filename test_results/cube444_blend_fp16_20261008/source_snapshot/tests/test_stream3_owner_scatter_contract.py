"""Source-bound CPU schedule regression; hardware execution is a separate gate."""
from pathlib import Path
import random
import re

SOURCE = Path(__file__).resolve().parents[1] / "cuda/stream3.cu"


def scatter_schedule(world, local):
    source = SOURCE.read_text()
    launches = re.findall(r"stream3_scatter_owner_kernel<<<([^>]+)>>>(.*?)\);",
                          source, re.S)
    schedule = []
    for config, arguments in launches:
        grid, block, shared, stream = [x.strip() for x in config.split(",")]
        assert block == "block" and shared == "0" and stream == "stream"
        mode = arguments.strip().rsplit(",", 1)[-1].strip()
        if mode == "false":
            assert grid == "world_size"
            schedule.extend(peer for peer in range(world) if peer != local)
        elif mode == "true":
            assert grid == "1"
            schedule.append(local)
        else:
            # Old single launch legally permits the local block to finish first.
            assert grid == "world_size" and mode == "world_size"
            schedule.extend([local] + [peer for peer in range(world) if peer != local])
    return schedule


def execute_tiles(records, owners, local, schedule):
    scratch = list(records)
    remote = {peer: [] for peer in set(schedule) if peer != local}
    for peer in schedule:
        running = 0
        for base in range(0, len(records), 256):
            loaded = scratch[base:base + 256]
            matched = [record for lane, record in enumerate(loaded)
                       if owners[base + lane] == peer]
            if peer == local:
                scratch[running:running + len(matched)] = matched
            else:
                remote[peer].extend(matched)
            running += len(matched)
    return scratch[:owners.count(local)], remote


def expected(records, owners, world, local):
    return ([r for r, owner in zip(records, owners) if owner == local],
            {peer: [r for r, owner in zip(records, owners) if owner == peer]
             for peer in range(world) if peer != local})


def test_actual_launch_schedule_preserves_minimal_remote_candidate():
    records = [("remote_A", 17, 2, 1), ("local_B", 29, 1, 0)]
    assert execute_tiles(records, [1, 0], 0, scatter_schedule(2, 0)) == \
        expected(records, [1, 0], 2, 0)


def test_actual_launch_schedule_preserves_all_metadata_at_tile_boundaries():
    rng = random.Random(910003)
    for world in (2, 8):
        for local in range(world):
            for count in (0, 1, 2, 255, 256, 257, 513):
                records = [(i, i * 11, i % 7, i % 24) for i in range(count)]
                cases = [[rng.randrange(world) for _ in records],
                         [local] * count, [(local + 1) % world] * count]
                for owners in cases:
                    assert execute_tiles(records, owners, local,
                                         scatter_schedule(world, local)) == \
                        expected(records, owners, world, local)


def test_source_model_matches_reads_writes_and_block_local_barriers():
    source = SOURCE.read_text()
    kernel = source.split("__global__ void stream3_scatter_owner_kernel", 1)[1].split(
        "__global__ void stream3_partition_fill_input_kernel", 1)[0]
    assert "const CandidateMeta candidate = i < count ? meta_scratch[i]" in kernel
    assert "meta_scratch[out] = candidate;" in kernel
    assert "remote_send_buffer[send_offset[peer] + out] = candidate;" in kernel
    assert kernel.index("meta_scratch[i]") < kernel.index("__syncthreads()") < kernel.index("meta_scratch[out]")
    assert "bool local_only" in kernel
    assert "local_only ? static_cast<std::uint32_t>(local_rank) : blockIdx.x" in kernel
    assert "if (!local_only && peer == static_cast<std::uint32_t>(local_rank))" in kernel
    assert scatter_schedule(8, 3) == [0, 1, 2, 4, 5, 6, 7, 3]


def test_parallel_peer_blocks_need_ordering_even_with_tile_barriers():
    records = ["remote_A", "local_B"]
    assert execute_tiles(records, [1, 0], 0, [0, 1]) != expected(records, [1, 0], 2, 0)
    assert execute_tiles(records, [1, 0], 0, [1, 0]) == expected(records, [1, 0], 2, 0)
