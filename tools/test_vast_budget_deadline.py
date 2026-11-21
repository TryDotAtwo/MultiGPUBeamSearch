"""CPU tests for absolute billing deadlines and stop/retrieval reserves."""
from __future__ import annotations

import unittest

from vast_budget_deadline import phase_timeout_seconds


class VastBudgetDeadlineTests(unittest.TestCase):
    def test_timeout_and_cleanup_must_fit_before_stop_and_future_reserves(self):
        seconds = phase_timeout_seconds(
            3000,
            stop_reserve_seconds=600,
            future_reserve_seconds=300,
            phase_overhead_seconds=30,
            termination_grace_seconds=60,
            max_seconds=1500,
            min_seconds=10,
            phase="remote-build-and-run",
            now_epoch=1000,
        )
        self.assertEqual(seconds, 1010)
        self.assertLessEqual(seconds + 60 + 30 + 600 + 300, 2000)

    def test_exact_deadline_boundary_includes_termination_grace(self):
        seconds = phase_timeout_seconds(
            2000,
            stop_reserve_seconds=600,
            future_reserve_seconds=300,
            phase_overhead_seconds=30,
            termination_grace_seconds=60,
            max_seconds=100,
            min_seconds=10,
            phase="retrieval",
            now_epoch=1000,
        )
        self.assertEqual(seconds, 10)
        with self.assertRaisesRegex(ValueError, "retrieval cannot start"):
            phase_timeout_seconds(
                1999,
                stop_reserve_seconds=600,
                future_reserve_seconds=300,
                phase_overhead_seconds=30,
                termination_grace_seconds=60,
                max_seconds=100,
                min_seconds=10,
                phase="retrieval",
                now_epoch=1000,
            )

    def test_unresponsive_ssh_costs_time_from_the_next_phase_budget(self):
        with self.assertRaisesRegex(ValueError, "retrieval cannot start"):
            phase_timeout_seconds(
                1800,
                stop_reserve_seconds=600,
                future_reserve_seconds=0,
                phase_overhead_seconds=200,
                termination_grace_seconds=30,
                max_seconds=300,
                min_seconds=1,
                phase="retrieval",
                now_epoch=1000,
            )

    def test_retrieval_is_refused_when_only_stop_readback_reserve_remains(self):
        with self.assertRaisesRegex(ValueError, "retrieval cannot start"):
            phase_timeout_seconds(
                1510,
                stop_reserve_seconds=480,
                future_reserve_seconds=0,
                phase_overhead_seconds=15,
                termination_grace_seconds=30,
                max_seconds=300,
                min_seconds=1,
                phase="retrieval",
                now_epoch=1000,
            )

    def test_phase_timeout_uses_one_absolute_deadline_and_preserves_future_reserves(self):
        seconds = phase_timeout_seconds(
            5000,
            stop_reserve_seconds=600,
            future_reserve_seconds=300,
            max_seconds=1500,
            min_seconds=600,
            phase="upload",
            now_epoch=3000,
        )
        self.assertEqual(seconds, 1100)

    def test_phase_timeout_is_capped_by_the_phase_limit(self):
        seconds = phase_timeout_seconds(
            9000,
            stop_reserve_seconds=600,
            future_reserve_seconds=300,
            max_seconds=900,
            min_seconds=60,
            phase="upload",
            now_epoch=1000,
        )
        self.assertEqual(seconds, 900)

    def test_exhausted_budget_after_staging_fails_before_the_next_phase(self):
        with self.assertRaisesRegex(ValueError, "remote-build.*at least 1200"):
            phase_timeout_seconds(
                4000,
                stop_reserve_seconds=1200,
                future_reserve_seconds=300,
                max_seconds=5400,
                min_seconds=1200,
                phase="remote-build",
                now_epoch=3100,
            )

    def test_deadline_refuses_a_short_timeout_that_would_eat_the_stop_reserve(self):
        with self.assertRaisesRegex(ValueError, "retrieval cannot start"):
            phase_timeout_seconds(
                4000,
                stop_reserve_seconds=1200,
                future_reserve_seconds=0,
                max_seconds=300,
                min_seconds=1,
                phase="retrieval",
                now_epoch=3000,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
