from __future__ import annotations

from pathlib import Path

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.data import PuzzleContract
from tools.cayleypy_public.model import ExportedModel
from tools.cayleypy_public.profile import RuntimePlan
from tools.cayleypy_public.runner import build_runner_invocation, parse_runner_output


def _config(mode: str = "first") -> PublicRunConfig:
    return PublicRunConfig.from_mapping({
        "author_name": "alice", "checkpoint_path": "/weights/model.pth",
        "puzzle_info_json": "/data/puzzle_info.json", "test_csv": "/data/test.csv",
        "sample_submission_csv": "/data/submission.csv", "puzzle_id_start": 7,
        "puzzle_id_end": 7, "beam_width": 2**21, "max_depth": 100,
        "reflect_mode": "off", "reflect_source_csv": None, "solution_mode": mode,
        "collect_until_depth": 12, "max_collected_solutions": 3, "touch_bfs_radius": 4,
        "publish_results": False, "results_ingest_url": "https://results.example/",
    })


def _plan() -> RuntimePlan:
    return RuntimePlan(2**21, 2**21, 0, 21, "output1", 2**20, 2048, 196608, 262144,
                       {"b_micro": 49152, "stream1_concurrency": 2, "stream3_ring_slots": 4,
                        "shard_count": 8, "shard_capacity_scale_ppm": 1000000,
                        "stream4_batch_candidates": 98304, "stream4_trigger_candidates": 98304,
                        "stream4_active_sort_slots": 2}, "")


def test_first_invocation_uses_exact_two_rank_torchrun_and_runtime_contract(tmp_path: Path) -> None:
    invocation = build_runner_invocation(_config(), _plan(), 7, "original", tmp_path / "weights", tmp_path)

    assert invocation.command[:4] == ("python", "-m", "torch.distributed.run", "--nproc-per-node=2")
    assert "--redirects=3" in invocation.command and "--tee=0:3" in invocation.command
    assert invocation.command[-3:] == ("7", "100", str(2**21))
    assert invocation.env["BEAM_SOLVED_NEIGHBORHOOD_RADIUS"] == "4"
    assert invocation.env["BEAM_REPAIR_K1_RADIUS"] == "4"
    assert invocation.env["BEAM_REPAIR_K2_RADIUS"] == "0"
    assert invocation.env["BEAM_SHARD_CAPACITY_CANDIDATES"] == "262144"
    assert invocation.env["BEAM_HISTORY_DIR"].startswith("/tmp/beam_history_public/")
    assert not {"WORLD_SIZE", "RANK", "LOCAL_RANK"}.intersection(invocation.env)


def test_collect_invocation_has_bounded_collection_controls(tmp_path: Path) -> None:
    invocation = build_runner_invocation(_config("collect"), _plan(), 7, "original", tmp_path / "weights", tmp_path)

    assert {key: invocation.env[key] for key in (
        "BEAM_SOLVE_BUCKET_MODE", "BEAM_SOLVE_BUCKET_STOP_DEPTH",
        "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS", "BEAM_SOLVED_RESULT_CAPACITY",
    )} == {"BEAM_SOLVE_BUCKET_MODE": "1", "BEAM_SOLVE_BUCKET_STOP_DEPTH": "12",
          "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS": "3", "BEAM_SOLVED_RESULT_CAPACITY": "196608"}
    assert invocation.result_tsv is not None
    assert invocation.env["BEAM_SOLVE_BUCKET_RESULT_TSV"] == str(invocation.result_tsv)


def test_parser_preserves_partial_tsv_and_reports_successful_capacity_limit(tmp_path: Path) -> None:
    result_tsv = tmp_path / "results.tsv"
    result_tsv.write_text("puzzle_id\tdepth_index\tfound_depth\ttotal_depth\tknown_length\tdelta\towner_rank\tsolution\n7\t4\t5\t5\t0\t0\t0\tcounterclockwise\n", encoding="utf-8")

    parsed = parse_runner_output("collection_status=capacity_reached\n", result_tsv, 7, "original")

    assert parsed.collection_status == "capacity_reached"
    assert [record.original_oriented_path for record in parsed.records] == ["counterclockwise"]


def test_production_runner_has_distinct_depth_and_capacity_stop_controls() -> None:
    source = Path("tools/production_runner.cu").read_text(encoding="utf-8")

    assert 'BEAM_SOLVE_BUCKET_STOP_DEPTH' in source
    assert 'BEAM_SOLVE_BUCKET_MAX_SOLUTIONS' in source
    assert 'collection_status=depth_reached' in source
    assert 'collection_status=capacity_reached' in source
    assert 'solve bucket overflow: increase BEAM_SOLVED_RESULT_CAPACITY' in source
