import tempfile
import unittest
from pathlib import Path

from tools.stream1_transformer_autotune import baseline_environment
from tools.stream1_transformer_policy_from_cache import load_policy_environment, policies_to_environment


class PolicyFromCacheTests(unittest.TestCase):
    def test_maps_every_family_to_environment(self):
        policies = {"qkv": "m128n128", "attn_out": "m64n64", "ff1": "baseline", "ff2": "m128n128"}
        self.assertEqual(
            policies_to_environment(policies),
            {
                "BEAM_STREAM1_TRANSFORMER_QKV_POLICY": "m128n128",
                "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY": "m64n64",
                "BEAM_STREAM1_TRANSFORMER_FF1_POLICY": "baseline",
                "BEAM_STREAM1_TRANSFORMER_FF2_POLICY": "m128n128",
            },
        )

    def test_bad_cache_fails_closed_to_all_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            cache.write_text('{"schema_version": 3, "signature": {"gpu": "other"}, "selected_environment": {}}')
            environment, status = load_policy_environment(cache, {"gpu": "sm86"})
        self.assertEqual(status, "signature_mismatch")
        self.assertEqual(environment, baseline_environment())


if __name__ == "__main__":
    unittest.main()