"""CPU tests for the installed CNOT Gloo lifecycle receipt contract."""
from __future__ import annotations

import unittest

from cnot_gloo_lifecycle import EXPECTED_CASES, validate_lifecycle


def fixture():
    events = {0: [], 1: []}
    for rank in (0, 1):
        for case_id, (phase, common_outcome) in EXPECTED_CASES.items():
            event = {
                "rank": rank,
                "case_id": case_id,
                "phase": phase,
                "common_outcome": common_outcome,
            }
            if case_id == "engine_mismatch":
                event.update(
                    requested_engine="auto" if rank == 0 else "python",
                    error_type="ValueError",
                    expected_error=True,
                )
            elif case_id.startswith("scorer_failure_rank_") and not case_id.endswith("_retry"):
                failed_rank = int(case_id.rsplit("_", 1)[1])
                event.update(
                    failed_rank=failed_rank,
                    error_type="RuntimeError",
                    expected_phase=True,
                    expected_rank=True,
                )
            elif case_id.endswith("_retry") and case_id.startswith("scorer_failure_rank_"):
                event["failed_rank"] = int(
                    case_id.removeprefix("scorer_failure_rank_").removesuffix("_retry")
                )
            events[rank].append(event)
    reports = {
        "engine_mismatch_retry": {"status": "solved", "bitwise_replay": True},
        "scorer_failure_rank_0_retry": {"status": "solved", "bitwise_replay": True},
        "scorer_failure_rank_1_retry": {"status": "solved", "bitwise_replay": True},
    }
    return events, reports


class CnotGlooLifecycleContractTests(unittest.TestCase):
    def test_rank_local_engine_inputs_may_differ_with_common_outcome(self):
        events, reports = fixture()
        self.assertEqual(events[0][0]["requested_engine"], "auto")
        self.assertEqual(events[1][0]["requested_engine"], "python")
        validate_lifecycle(events, reports)

    def test_missing_phase_is_rejected(self):
        events, reports = fixture()
        events[1] = [event for event in events[1] if event["case_id"] != "scorer_failure_rank_1_retry"]
        with self.assertRaisesRegex(ValueError, "missing lifecycle phases"):
            validate_lifecycle(events, reports)

    def test_wrong_rank_is_rejected(self):
        events, reports = fixture()
        events[1][0]["rank"] = 0
        with self.assertRaisesRegex(ValueError, "rank mismatch"):
            validate_lifecycle(events, reports)

    def test_missing_rank_zero_replay_is_rejected(self):
        events, reports = fixture()
        reports["scorer_failure_rank_0_retry"]["bitwise_replay"] = False
        with self.assertRaisesRegex(ValueError, "lacks bitwise replay"):
            validate_lifecycle(events, reports)

    def test_unexpected_report_set_is_rejected(self):
        events, reports = fixture()
        reports.pop("scorer_failure_rank_1_retry")
        with self.assertRaisesRegex(ValueError, "rank-zero replay reports"):
            validate_lifecycle(events, reports)


if __name__ == "__main__":
    unittest.main()
