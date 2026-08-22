from __future__ import annotations

import numpy as np

from tools.sm120_quant_evaluate import (
    build_initial_mixed_precision_policies,
    build_cumulative_rollback_policies,
    select_stratified_states,
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
    assert len(policies) == 1 + len(CORE_OPERATORS)
    for name, policy in policies[1:]:
        assert name.startswith("rollback_")
        assert list(policy.values()).count("fp16") == 1
        assert list(policy.values()).count("sm120_block_fp8") == len(CORE_OPERATORS) - 1


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
