from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import beam_engine


def test_normal_path() -> None:
    result = beam_engine.v6_dispatcher_skeleton_single_gpu_smoke(stop_path=False, verbose=False)
    assert result["path"] == "normal"
    assert result["stream2_hash_written"] is True
    assert result["compact_count"] == 3
    assert result["unique_count"] == 2
    assert result["local_count"] == 2
    assert result["stream5_byte_identical"] is True
    assert result["collector_dirty_count_initial"] == 2
    assert result["stream4_launched"] is True
    assert result["clean_count"] == 2
    assert result["dirty_count"] == 0
    assert result["processing_flag"] == 0
    assert result["final_launched"] is True
    assert result["final_count"] == 2
    assert result["final_response_target0"] == 0
    assert result["next_padding_zero"] is True
    assert result["current_frontier_updated"] is True
    assert result["stream1_production_called"] is False
    assert result["fallback_backend_called"] is False


def test_stop_path() -> None:
    result = beam_engine.v6_dispatcher_skeleton_single_gpu_smoke(stop_path=True, verbose=False)
    assert result["path"] == "stop"
    assert result["stream2_hash_written"] is True
    assert result["solved_count"] >= 1
    assert result["first_solved_score_key"] == 0
    assert result["solved_flag"] == 1
    assert result["stop_flag"] == 1
    assert result["stream3_launched"] is False
    assert result["stream4_launched"] is False
    assert result["final_launched"] is False
    assert result["solved_list_copied_to_cpu"] is True
    assert result["stream1_production_called"] is False
    assert result["fallback_backend_called"] is False


def main() -> None:
    test_normal_path()
    test_stop_path()
    print("DISPATCHER_SKELETON_SMOKE_OK")


if __name__ == "__main__":
    main()
