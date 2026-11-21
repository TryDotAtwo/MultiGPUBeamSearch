#!/usr/bin/env python3
"""CPU contracts for the saturated, rank-aware permutation acceptance case."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import cayleypy as cp
from multigpubeamsearch.build import configure_command
from multigpubeamsearch.contracts import GraphContract
from permutation_acceptance import (
    MT19937_64,
    build_saturated_reference_fixture,
    decode_frontier_snapshot,
    validate_depth_flow_trace,
)


def _trace(rank: int, *, pid: int, gpu_uuid: str, run_id="run-42", case_id="permutation-saturated-b3", call_id=2):
    fields = {
        "source_sha": "a" * 40,
        "wheel_sha256": "b" * 64,
        "build_key": "c" * 64,
        "binary_sha256": "d" * 64,
        "run_id": run_id, "case_id": case_id, "call_id": call_id,
        "rank": rank, "world_size": 2, "depth": 1, "pid": pid,
        "gpu_uuid": gpu_uuid, "shard_count": 64,
        "frontier_size": 2 if rank == 0 else 1,
        "generated": 32 if rank == 0 else 16,
        "stream3_unique": 11 if rank == 0 else 7,
        "final_global_less": 0, "final_global_equal": 19,
        "final_total_available": 19, "final_global_keep": 3,
        "final_candidate_count": 11 if rank == 0 else 8,
        "final_request_count": 2 if rank == 0 else 1,
        "next_frontier_size": 2 if rank == 0 else 1,
    }
    return "depth_flow_trace " + " ".join(f"{key}={value}" for key, value in fields.items())


class PermutationAcceptanceTests(unittest.TestCase):
    def test_mt19937_64_matches_std_engine_seed_vector(self):
        self.assertEqual(MT19937_64(5489).next(), 14514284786278117030)

    def test_trace_build_flag_is_explicit_and_off_by_default(self):
        contract = SimpleNamespace(state_len=4, move_count=16)
        programs = {"cmake": "cmake", "cxx": "c++", "nvcc": "nvcc"}
        off = configure_command(Path("."), Path("build"), Path("cutlass"), contract,
                                "mlp", (86,), programs)
        on = configure_command(Path("."), Path("build"), Path("cutlass"), contract,
                               "mlp", (86,), programs, acceptance_rank_trace=True)
        self.assertIn("-DBEAM_DEBUG_DEPTH_FLOW_TRACE=OFF", off)
        self.assertIn("-DBEAM_DEBUG_DEPTH_FLOW_TRACE=ON", on)

    @classmethod
    def setUpClass(cls):
        graph = cp.CayleyGraph(cp.PermutationGroups.lrx(4), device="cpu")
        contract = GraphContract.from_graph(graph, graph.definition.central_state)
        cls.fixture = build_saturated_reference_fixture(
            contract.center, contract.generators, beam_width=3
        )

    def test_reference_paths_replay_to_each_survivor_independently(self):
        for world_size in (1, 2):
            for candidate in self.fixture.worlds[world_size].steps[1].ordered_survivors:
                state = tuple(self.fixture.start)
                for move_id in candidate.path:
                    state = tuple(state[source] for source in self.fixture.generators[move_id])
                self.assertEqual(state, candidate.state)

    def test_lrx4_b3_reference_saturates_and_deduplicates_across_ranks(self):
        fixture = self.fixture
        self.assertGreaterEqual(fixture.distance, 3)
        for world_size in (1, 2):
            step = fixture.worlds[world_size].steps[1]
            self.assertGreater(step.eligible_distinct, 3)
            self.assertEqual(step.keep_count, 3)
            self.assertTrue(step.all_scores_tied)
            self.assertEqual(sum(map(len, step.frontier_by_rank)), 3)
            self.assertTrue(all(len(frontier) > 0 for frontier in step.frontier_by_rank))
        two_gpu = fixture.worlds[2].steps[1]
        self.assertEqual(two_gpu.input_counts, (2, 1))
        self.assertGreater(two_gpu.cross_source_duplicate_count, 0)
        self.assertEqual(tuple(map(len, two_gpu.frontier_by_rank)), (2, 1))
        self.assertEqual(len({c.hash128 for c in two_gpu.ordered_survivors}), 3)

    def test_trace_binds_call_case_and_both_observed_rank_devices(self):
        log = "\n".join((
            _trace(0, pid=1001, gpu_uuid="0" * 32),
            _trace(1, pid=1002, gpu_uuid="1" * 32),
        ))
        rows = validate_depth_flow_trace(
            log, run_id="run-42", case_id="permutation-saturated-b3",
            call_id=2, world_size=2, depth=1, beam_width=3,
            source_sha="a" * 40, wheel_sha256="b" * 64,
            build_key="c" * 64, binary_sha256="d" * 64,
        )
        self.assertEqual(tuple(row["rank"] for row in rows), (0, 1))
        self.assertEqual(tuple(row["gpu_uuid"] for row in rows), ("0" * 32, "1" * 32))

    def test_trace_rejects_missing_rank_duplicate_identity_and_wrong_run(self):
        rows = (
            _trace(0, pid=1001, gpu_uuid="0" * 32),
            _trace(1, pid=1002, gpu_uuid="1" * 32),
        )
        cases = (
            "\n".join(rows[:1]),
            "\n".join((rows[0], rows[1].replace("pid=1002", "pid=1001"))),
            "\n".join((rows[0], rows[1].replace("gpu_uuid=" + "1" * 32, "gpu_uuid=" + "0" * 32))),
            "\n".join((rows[0], rows[1].replace("run_id=run-42", "run_id=foreign"))),
            "\n".join((rows[0], rows[1].replace("final_global_keep=3", "final_global_keep=2"))),
            "\n".join((rows[0], rows[1].replace("final_global_equal=19", "final_global_equal=3"))),
        )
        for log in cases:
            with self.subTest(log=log):
                with self.assertRaises(ValueError):
                    validate_depth_flow_trace(
                        log, run_id="run-42", case_id="permutation-saturated-b3",
                        call_id=2, world_size=2, depth=1, beam_width=3,
                    )

    def test_frontier_snapshot_checks_shape_and_zero_padding(self):
        raw = bytes((1, 0, 2, 3)) + bytes(12) + bytes((3, 2, 1, 0)) + bytes(12)
        self.assertEqual(
            decode_frontier_snapshot(raw, state_len=4, storage_len=16),
            ((1, 0, 2, 3), (3, 2, 1, 0)),
        )
        with self.assertRaises(ValueError):
            decode_frontier_snapshot(raw[:-1], state_len=4, storage_len=16)
        with self.assertRaises(ValueError):
            decode_frontier_snapshot(raw[:16] + bytes((3, 2, 1, 0)) + bytes((1,)) + bytes(11),
                                     state_len=4, storage_len=16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
