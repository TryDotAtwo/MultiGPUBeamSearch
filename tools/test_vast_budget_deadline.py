"""CPU tests for absolute billing deadlines and stop/retrieval reserves."""
from __future__ import annotations

import unittest

from vast_budget_deadline import phase_timeout_seconds


class VastBudgetDeadlineTests(unittest.TestCase):
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
