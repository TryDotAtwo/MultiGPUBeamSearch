import json
import tempfile
import unittest
from pathlib import Path

from tools.stream1_transformer_autotune import (
    Observation,
    baseline_environment,
    baseline_policy_map,
    canonical_signature,
    load_cached_policy,
    policy_environment_variable,
    resolve_cached_environment,
    resolve_cached_policy,
    resolve_cached_policies,
    select_family_policy,
    select_policy,
    write_cache_atomic,
)


class AutotuneTests(unittest.TestCase):
    def test_signature_is_canonical(self):
        self.assertEqual(canonical_signature({"b": 2, "a": 1}), '{"a":1,"b":2}')

    def test_selects_exact_faster_candidate(self):
        rows = [Observation("baseline", value, "same") for value in (10, 11, 9, 10, 10)]
        rows += [Observation("m64n128", value, "same") for value in (9, 9, 9, 9, 9)]
        result = select_policy(rows)
        self.assertEqual(result["selected_policy"], "m64n128")

    def test_rejects_output_mismatch(self):
        rows = [Observation("baseline", 10, "same") for _ in range(5)]
        rows += [Observation("m64n128", 8, "same") for _ in range(4)]
        rows += [Observation("m64n128", 8, "different")]
        result = select_policy(rows)
        self.assertEqual(result["selected_policy"], "baseline")
        self.assertEqual(result["rejected"]["m64n128"], "output_mismatch")

    def test_rejects_nondeterministic_baseline(self):
        rows = [Observation("baseline", 10, "same") for _ in range(4)]
        rows += [Observation("baseline", 10, "different")]
        self.assertEqual(select_policy(rows)["status"], "baseline_nondeterministic")

    def test_requires_three_percent_speedup(self):
        rows = [Observation("baseline", 10, "same") for _ in range(5)]
        rows += [Observation("m64n128", 9.8, "same") for _ in range(5)]
        self.assertEqual(select_policy(rows)["selected_policy"], "baseline")

    def test_cache_is_exact_and_fail_closed(self):
        signature = {"gpu": "sm86", "shapes": [[1, 2, 3]]}
        valid = {"schema_version": 1, "signature": signature, "selected_policy": "m64n128"}
        self.assertEqual(resolve_cached_policy(valid, signature), ("m64n128", "cache_hit"))
        self.assertEqual(resolve_cached_policy(valid, {"gpu": "sm80"})[0], "baseline")
        invalid = dict(valid, selected_policy="unknown")
        self.assertEqual(resolve_cached_policy(invalid, signature)[0], "baseline")

    def test_policy_family_maps_to_explicit_environment_variable(self):
        self.assertEqual(policy_environment_variable("qkv"), "BEAM_STREAM1_TRANSFORMER_QKV_POLICY")
        self.assertEqual(policy_environment_variable("attn_out"), "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY")
        self.assertEqual(policy_environment_variable("ff2"), "BEAM_STREAM1_TRANSFORMER_FF2_POLICY")
        with self.assertRaisesRegex(ValueError, "unknown policy family"):
            policy_environment_variable("unknown")

    def test_cache_accepts_compiled_m128n128_policy(self):
        signature = {"gpu": "sm86"}
        cache = {"schema_version": 1, "signature": signature, "selected_policy": "m128n128"}
        self.assertEqual(resolve_cached_policy(cache, signature), ("m128n128", "cache_hit"))
    def test_multifamily_cache_is_exact_and_fail_closed(self):
        signature = {"gpu": "sm86", "shapes": [[1, 2, 3]]}
        selected = {"qkv": "m128n128", "attn_out": "m128n128", "ff1": "m128n128", "ff2": "m128n128"}
        cache = {"schema_version": 2, "signature": signature, "selected_policies": selected}
        self.assertEqual(resolve_cached_policies(cache, signature), (selected, "cache_hit"))
        malformed = dict(cache, selected_policies={**selected, "ff2": "unknown"})
        self.assertEqual(resolve_cached_policies(malformed, signature), (baseline_policy_map(), "unknown_policy"))
        self.assertEqual(resolve_cached_policies(cache, {"gpu": "sm80"}), (baseline_policy_map(), "signature_mismatch"))

    def test_environment_cache_is_exact_complete_and_fail_closed(self):
        signature = {"gpu": "sm86", "workload": "block51-b512-c1"}
        selected = baseline_environment()
        selected.update(
            {
                "BEAM_STREAM1_TRANSFORMER_QKV_POLICY": "m128n128",
                "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY": "m128n128",
                "BEAM_STREAM1_TRANSFORMER_FF1_POLICY": "m128n128",
                "BEAM_STREAM1_TRANSFORMER_FF2_POLICY": "m128n128",
                "BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE": "8",
                "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE": "2",
                "BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE": "8",
                "BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE": "2",
                "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE": "fused",
                "BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE": "fused",
            }
        )
        cache = {"schema_version": 3, "signature": signature, "selected_environment": selected}
        self.assertEqual(resolve_cached_environment(cache, signature), (selected, "cache_hit"))
        self.assertEqual(
            resolve_cached_environment(cache, {"gpu": "sm75"}),
            (baseline_environment(), "signature_mismatch"),
        )
        incomplete = dict(cache, selected_environment={key: value for key, value in selected.items() if "FF2_SWIZZLE" not in key})
        self.assertEqual(resolve_cached_environment(incomplete, signature)[1], "malformed_cache")

    def test_environment_cache_rejects_cross_field_incompatibility(self):
        signature = {"gpu": "sm86"}
        selected = baseline_environment()
        selected["BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE"] = "8"
        cache = {"schema_version": 3, "signature": signature, "selected_environment": selected}
        environment, status = resolve_cached_environment(cache, signature)
        self.assertEqual(status, "incompatible_environment")
        self.assertEqual(environment, baseline_environment())

        selected = baseline_environment()
        selected["BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY"] = "q64k64v4"
        cache = {"schema_version": 3, "signature": signature, "selected_environment": selected}
        self.assertEqual(resolve_cached_environment(cache, signature), (selected, "cache_hit"))
    def test_family_selection_preserves_previous_policies(self):
        current = baseline_policy_map()
        current["ff1"] = "m128n128"
        rows = [Observation("baseline", 10.0, "same") for _ in range(20)]
        rows += [Observation("m128n128", 9.0, "same") for _ in range(20)]
        updated, decision = select_family_policy(current, "qkv", rows, min_repeats=20)
        self.assertEqual(updated["ff1"], "m128n128")
        self.assertEqual(updated["qkv"], "m128n128")
        self.assertEqual(decision["status"], "candidate_selected")
    def test_atomic_cache_roundtrip_and_malformed_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            signature = {"gpu": "sm86"}
            payload = {"schema_version": 1, "signature": signature, "selected_policy": "baseline"}
            write_cache_atomic(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertEqual(load_cached_policy(path, signature), ("baseline", "cache_hit"))
            path.write_text("not-json")
            self.assertEqual(load_cached_policy(path, signature)[0], "baseline")


if __name__ == "__main__":
    unittest.main()
