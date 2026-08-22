from __future__ import annotations

import numpy as np

from tools.sm120_quant_evaluate import (
    build_initial_mixed_precision_policies,
    build_cumulative_rollback_policies,
    build_incremental_fp8_policies,
    select_stratified_states,
    _three_way_metrics,
)
from tools.sm120_quant_tuner import CORE_OPERATORS


def test_select_stratified_states_balances_depths_and_is_deterministic() -> None:
    states = np.arange(30 * 4, dtype=np.uint8).reshape(30, 4)
    depths = np.repeat(np.array([4, 5, 6], dtype=np.int32), 10)
    selected, selected_depths, indices = select_stratified_states(
        states, depths, max_states=8
    )
    assert selected.shape == (8, 4)
    assert np.bincount(selected_depths - 4).tolist() == [3, 3, 2]
    np.testing.assert_array_equal(selected, states[indices])
    selected2, depths2, indices2 = select_stratified_states(states, depths, max_states=8)
    np.testing.assert_array_equal(selected2, selected)
    np.testing.assert_array_equal(depths2, selected_depths)
    np.testing.assert_array_equal(indices2, indices)

    calibration = select_stratified_states(states, depths, max_states=8, split="calibration")
    holdout = select_stratified_states(states, depths, max_states=8, split="holdout")
    assert set(calibration[2]).isdisjoint(set(holdout[2]))
    assert set(calibration[1]) == {4, 5, 6}
    assert set(holdout[1]) == {4, 5, 6}


def test_initial_mixed_precision_policies_include_fp8_and_each_single_rollback() -> None:
    policies = build_initial_mixed_precision_policies()
    assert policies[0][0] == "all_core_fp8"
    assert all(value == "sm120_block_fp8" for value in policies[0][1].values())
    assert len(policies) == 1 + 2 * len(CORE_OPERATORS)
    rollback_policies = [item for item in policies if item[0].startswith("rollback_")]
    only_policies = [item for item in policies if item[0].startswith("only_fp8_")]
    for name, policy in rollback_policies:
        assert name.startswith("rollback_")
        assert list(policy.values()).count("fp16") == 1
        assert list(policy.values()).count("sm120_block_fp8") == len(CORE_OPERATORS) - 1
    for name, policy in only_policies:
        assert name.startswith("only_fp8_")
        assert list(policy.values()).count("sm120_block_fp8") == 1


def test_cumulative_rollbacks_prioritize_quality_and_end_at_fp16() -> None:
    rows = []
    for index, operator in enumerate(CORE_OPERATORS):
        rows.append({
            "name": "rollback_" + operator,
            "top1_agreement": 0.9 + index * 0.001,
            "threshold_band_agreement": 0.8,
            "global_top_per_state_overlap": 0.7,
            "topk_set_overlap": 0.6,
        })
    policies = build_cumulative_rollback_policies(rows)
    assert policies[0][0].startswith("cumulative_02_")
    assert list(policies[0][1].values()).count("fp16") == 2
    assert list(policies[-1][1].values()).count("fp16") == len(CORE_OPERATORS)
    assert policies[-1][1][CORE_OPERATORS[-1]] == "fp16"


def test_incremental_fp8_path_starts_with_two_and_ends_at_all_fp8() -> None:
    rows = [{
        "name": "only_fp8_" + operator,
        "top1_agreement": 1.0,
        "threshold_band_agreement": 1.0,
        "global_top_per_state_overlap": 1.0 - index * 0.001,
        "topk_set_overlap": 1.0,
    } for index, operator in enumerate(CORE_OPERATORS)]
    policies = build_incremental_fp8_policies(rows)
    assert list(policies[0][1].values()).count("sm120_block_fp8") == 2
    assert list(policies[-1][1].values()).count("sm120_block_fp8") == len(CORE_OPERATORS)


def test_three_way_metrics_use_fp32_as_truth_and_report_incremental_fp16_loss() -> None:
    fp32 = np.array([[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]])
    fp16 = fp32 + np.array([[0.0, 0.1, -0.1], [0.1, 0.0, 0.0]])
    mixed = fp16 + np.array([[0.0, 0.02, 0.0], [0.0, -0.02, 0.0]])
    metrics = _three_way_metrics(fp32, fp16, mixed, 0.5)
    assert metrics["logit_rmse"] > metrics["vs_fp16"]["logit_rmse"]
    assert metrics["wall_seconds"] == 0.5
    assert metrics["vs_fp16"]["wall_seconds"] == 0.5
