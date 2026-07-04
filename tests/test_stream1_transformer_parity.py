import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stream1_transformer_parity import BackendRun, compare_runs, parse_pairs


def run(
    backend: str,
    *,
    checksum: int = 100,
    digest: int | None = 123456789,
    first_score_keys: list[int] | None = None,
    status: str = "ok",
) -> BackendRun:
    return BackendRun(
        backend=backend,
        mode="eager",
        command=[backend],
        env={},
        return_code=0,
        checksum=checksum,
        score_key_digest=digest,
        first_score_keys=first_score_keys if first_score_keys is not None else [10, 20, 30],
        log_path="",
        status=status,
    )


class Stream1TransformerParityTests(unittest.TestCase):
    def test_parse_pairs_reads_score_key_digest(self) -> None:
        parsed = parse_pairs(
            "stream1_transformer_micro checksum=42 score_key_digest=987 first_score_keys=1,2,3"
        )
        self.assertEqual(parsed["checksum"], "42")
        self.assertEqual(parsed["score_key_digest"], "987")
        self.assertEqual(parsed["first_score_keys"], "1,2,3")

    def test_compare_passes_matching_digest_with_first_row_tolerance(self) -> None:
        result = compare_runs(
            [
                run("pytorch", first_score_keys=[100, 200]),
                run("libtorch", first_score_keys=[101, 199]),
            ],
            tolerance=2,
        )
        self.assertEqual(result["status"], "pass")

    def test_compare_fails_digest_mismatch_even_when_row_matches(self) -> None:
        result = compare_runs(
            [
                run("pytorch", checksum=100, digest=111, first_score_keys=[1, 2, 3]),
                run("native_cutlass", checksum=100, digest=222, first_score_keys=[1, 2, 3]),
            ],
            tolerance=0,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["comparisons"][1]["reason"], "score key digest mismatch")

    def test_compare_fails_missing_digest(self) -> None:
        result = compare_runs(
            [run("pytorch", digest=111), run("libtorch", digest=None)],
            tolerance=0,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["comparisons"][1]["reason"], "missing score key digest")


if __name__ == "__main__":
    unittest.main()