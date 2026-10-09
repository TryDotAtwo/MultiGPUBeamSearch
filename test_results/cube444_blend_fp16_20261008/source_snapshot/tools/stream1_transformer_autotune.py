#!/usr/bin/env python3
"""Shape/GPU-aware offline autotuner for native Stream1 transformer FF1 policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MULTIFAMILY_SCHEMA_VERSION = 2
BASELINE_POLICY = "baseline"
KNOWN_POLICIES = (BASELINE_POLICY, "m64n128", "m128n128", "m128n128w64n32", "m64n64")
POLICY_FAMILIES = ("qkv", "attn_out", "ff1", "ff2")
FAMILY_POLICIES = {
    "qkv": frozenset((BASELINE_POLICY, "m64n128", "m128n128")),
    "attn_out": frozenset((BASELINE_POLICY, "m64n64", "m128n128")),
    "ff1": frozenset((BASELINE_POLICY, "m64n128", "m128n128", "m128n128w64n32")),
    "ff2": frozenset((BASELINE_POLICY, "m64n64", "m128n128")),
}
POLICY_ENVIRONMENT = {
    "qkv": "BEAM_STREAM1_TRANSFORMER_QKV_POLICY",
    "attn_out": "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY",
    "ff1": "BEAM_STREAM1_TRANSFORMER_FF1_POLICY",
    "ff2": "BEAM_STREAM1_TRANSFORMER_FF2_POLICY",
}
ENVIRONMENT_SCHEMA_VERSION = 3
ENVIRONMENT_ALLOWED_VALUES = {
    **{POLICY_ENVIRONMENT[family]: FAMILY_POLICIES[family] for family in POLICY_FAMILIES},
    "BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE": frozenset(("1", "4", "8")),
    "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE": frozenset(("1", "2")),
    "BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE": frozenset(("1", "4", "8")),
    "BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE": frozenset(("1", "2")),
    "BEAM_STREAM1_TRANSFORMER_FF1_STAGES": frozenset(("2", "3")),
    "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE": frozenset(("separate", "fused")),
    "BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE": frozenset(("separate", "fused")),
    "BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY": frozenset(("row", "persistent")),
    "BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY": frozenset(("q64k64", "q32k64", "q64k64v4")),
    "BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY": frozenset(("padded64", "exact32")),
}
MS_PATTERN = re.compile(r"stream1_transformer_micro .*?ms_per_launch_group=([0-9.]+)")


@dataclass(frozen=True)
class Observation:
    policy: str
    milliseconds: float
    dump_sha256: str


def policy_environment_variable(family: str) -> str:
    try:
        return POLICY_ENVIRONMENT[family]
    except KeyError as exc:
        raise ValueError(f"unknown policy family: {family}") from exc


def baseline_policy_map() -> dict[str, str]:
    return {family: BASELINE_POLICY for family in POLICY_FAMILIES}

def policies_to_environment_values(policies: dict[str, str]) -> dict[str, str]:
    return {POLICY_ENVIRONMENT[family]: policies[family] for family in POLICY_FAMILIES}


def baseline_environment() -> dict[str, str]:
    environment = policies_to_environment_values(baseline_policy_map())
    environment.update(
        {
            "BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE": "1",
            "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE": "1",
            "BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE": "1",
            "BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE": "1",
            "BEAM_STREAM1_TRANSFORMER_FF1_STAGES": "3",
            "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE": "separate",
            "BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE": "separate",
            "BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY": "row",
            "BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY": "q64k64",
            "BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY": "padded64",
        }
    )
    return environment


def _environment_is_compatible(environment: dict[str, str]) -> bool:
    if set(environment) != set(ENVIRONMENT_ALLOWED_VALUES):
        return False
    if any(environment[key] not in allowed for key, allowed in ENVIRONMENT_ALLOWED_VALUES.items()):
        return False
    qkv_policy = environment["BEAM_STREAM1_TRANSFORMER_QKV_POLICY"]
    attn_policy = environment["BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY"]
    ff1_policy = environment["BEAM_STREAM1_TRANSFORMER_FF1_POLICY"]
    ff2_policy = environment["BEAM_STREAM1_TRANSFORMER_FF2_POLICY"]
    ff1_stages = environment["BEAM_STREAM1_TRANSFORMER_FF1_STAGES"]
    if environment["BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE"] in ("4", "8") and qkv_policy != "m128n128":
        return False
    if environment["BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE"] in ("4", "8") and (
        ff1_policy != "m128n128" or ff1_stages != "3"
    ):
        return False
    for family, policy in (("ATTN_OUT", attn_policy), ("FF2", ff2_policy)):
        swizzle = environment[f"BEAM_STREAM1_TRANSFORMER_{family}_SWIZZLE"]
        epilogue = environment[f"BEAM_STREAM1_TRANSFORMER_{family}_EPILOGUE"]
        if epilogue == "fused" and policy != "m128n128":
            return False
        if swizzle == "2" and (policy != "m128n128" or epilogue != "fused"):
            return False
    return True


def resolve_cached_environment(cache: Any, signature: dict[str, Any]) -> tuple[dict[str, str], str]:
    fallback = baseline_environment()
    if not isinstance(cache, dict):
        return fallback, "malformed_cache"
    if cache.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        return fallback, "schema_mismatch"
    if cache.get("signature") != signature:
        return fallback, "signature_mismatch"
    selected = cache.get("selected_environment")
    if not isinstance(selected, dict) or set(selected) != set(ENVIRONMENT_ALLOWED_VALUES):
        return fallback, "malformed_cache"
    normalized = {str(key): str(value) for key, value in selected.items()}
    if any(normalized[key] not in allowed for key, allowed in ENVIRONMENT_ALLOWED_VALUES.items()):
        return fallback, "unknown_environment_value"
    if not _environment_is_compatible(normalized):
        return fallback, "incompatible_environment"
    return normalized, "cache_hit"


def resolve_cached_policies(cache: Any, signature: dict[str, Any]) -> tuple[dict[str, str], str]:
    fallback = baseline_policy_map()
    if not isinstance(cache, dict):
        return fallback, "malformed_cache"
    if cache.get("schema_version") != MULTIFAMILY_SCHEMA_VERSION:
        return fallback, "schema_mismatch"
    if cache.get("signature") != signature:
        return fallback, "signature_mismatch"
    selected = cache.get("selected_policies")
    if not isinstance(selected, dict) or set(selected) != set(POLICY_FAMILIES):
        return fallback, "malformed_cache"
    if any(selected[family] not in FAMILY_POLICIES[family] for family in POLICY_FAMILIES):
        return fallback, "unknown_policy"
    return {family: str(selected[family]) for family in POLICY_FAMILIES}, "cache_hit"


def select_family_policy(
    current: dict[str, str],
    family: str,
    observations: Iterable[Observation],
    *,
    min_repeats: int = 5,
    min_improvement: float = 0.03,
) -> tuple[dict[str, str], dict[str, Any]]:
    if family not in FAMILY_POLICIES:
        raise ValueError(f"unknown policy family: {family}")
    if set(current) != set(POLICY_FAMILIES):
        raise ValueError("current policy map must contain every policy family")
    decision = select_policy(
        observations, min_repeats=min_repeats, min_improvement=min_improvement
    )
    updated = dict(current)
    selected = decision["selected_policy"]
    if decision["status"] == "candidate_selected" and selected in FAMILY_POLICIES[family]:
        updated[family] = selected
    return updated, decision

def canonical_signature(signature: dict[str, Any]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def signature_digest(signature: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_signature(signature).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(child)))
    return digest.hexdigest()


def select_policy(
    observations: Iterable[Observation],
    *,
    min_repeats: int = 5,
    min_improvement: float = 0.03,
) -> dict[str, Any]:
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.policy, []).append(observation)

    baseline = grouped.get(BASELINE_POLICY, [])
    if len(baseline) < min_repeats:
        return {"selected_policy": BASELINE_POLICY, "status": "baseline_insufficient_repeats"}
    baseline_hashes = {item.dump_sha256 for item in baseline}
    baseline_median = statistics.median(item.milliseconds for item in baseline)
    if len(baseline_hashes) != 1:
        return {
            "selected_policy": BASELINE_POLICY,
            "status": "baseline_nondeterministic",
            "baseline_median_ms": baseline_median,
            "baseline_dump_hashes": sorted(baseline_hashes),
        }

    reference_hash = next(iter(baseline_hashes))
    accepted: list[tuple[float, str, float]] = []
    rejected: dict[str, str] = {}
    for policy, rows in grouped.items():
        if policy == BASELINE_POLICY:
            continue
        if policy not in KNOWN_POLICIES:
            rejected[policy] = "unknown_policy"
            continue
        if len(rows) < min_repeats:
            rejected[policy] = "insufficient_repeats"
            continue
        if any(row.dump_sha256 != reference_hash for row in rows):
            rejected[policy] = "output_mismatch"
            continue
        candidate_median = statistics.median(row.milliseconds for row in rows)
        improvement = (baseline_median - candidate_median) / baseline_median
        if improvement < min_improvement:
            rejected[policy] = "insufficient_speedup"
            continue
        accepted.append((candidate_median, policy, improvement))

    if not accepted:
        return {
            "selected_policy": BASELINE_POLICY,
            "status": "no_candidate_passed",
            "baseline_median_ms": baseline_median,
            "rejected": rejected,
        }
    candidate_median, policy, improvement = min(accepted)
    return {
        "selected_policy": policy,
        "status": "candidate_selected",
        "baseline_median_ms": baseline_median,
        "selected_median_ms": candidate_median,
        "improvement": improvement,
        "rejected": rejected,
    }


def resolve_cached_policy(
    cache: Any,
    signature: dict[str, Any],
    allowed_policies: Iterable[str] = KNOWN_POLICIES,
) -> tuple[str, str]:
    if not isinstance(cache, dict):
        return BASELINE_POLICY, "malformed_cache"
    if cache.get("schema_version") != SCHEMA_VERSION:
        return BASELINE_POLICY, "schema_mismatch"
    if cache.get("signature") != signature:
        return BASELINE_POLICY, "signature_mismatch"
    policy = cache.get("selected_policy")
    if policy not in set(allowed_policies):
        return BASELINE_POLICY, "unknown_policy"
    return str(policy), "cache_hit"


def load_cached_policy(path: Path, signature: dict[str, Any]) -> tuple[str, str]:
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return BASELINE_POLICY, "cache_unavailable"
    return resolve_cached_policy(cache, signature)

def load_cached_policies(path: Path, signature: dict[str, Any]) -> tuple[dict[str, str], str]:
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return baseline_policy_map(), "cache_unavailable"
    return resolve_cached_policies(cache, signature)

def load_cached_environment(path: Path, signature: dict[str, Any]) -> tuple[dict[str, str], str]:
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return baseline_environment(), "cache_unavailable"
    return resolve_cached_environment(cache, signature)


def write_cache_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def query_gpu() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,uuid",
        "--format=csv,noheader,nounits",
    ]
    fields = [part.strip() for part in subprocess.check_output(command, text=True).splitlines()[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError("unexpected nvidia-smi GPU metadata")
    return dict(zip(("name", "compute_capability", "driver", "uuid"), fields, strict=True))


def run_benchmark(
    executable: Path,
    puzzle_id: int,
    environment: dict[str, str],
    dump_path: Path,
    policy_env_var: str = "BEAM_STREAM1_TRANSFORMER_FF1_POLICY",
) -> Observation:
    run_env = os.environ.copy()
    run_env.update(environment)
    run_env["BEAM_STREAM1_TRANSFORMER_SCORE_DUMP"] = str(dump_path)
    completed = subprocess.run(
        [str(executable), str(puzzle_id)], env=run_env, text=True, capture_output=True, check=True
    )
    match = MS_PATTERN.search(completed.stdout)
    if match is None:
        raise RuntimeError("benchmark output is missing ms_per_launch_group")
    policy = environment[policy_env_var]
    return Observation(policy, float(match.group(1)), file_sha256(dump_path))


def build_multifamily_signature(args: argparse.Namespace) -> dict[str, Any]:
    full_rows = args.b_micro * args.seq_len
    cls_rows = args.b_micro
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "gpu": query_gpu(),
        "dtype": args.dtype,
        "b_micro": args.b_micro,
        "concurrency": args.concurrency,
        "execution": {
            "block51": True,
            "final_cls_only": True,
            "final_cls_attention": False,
        },
        "families": {
            "qkv": {
                "shapes": [[full_rows, 3 * args.d_model, args.d_model, args.full_layers],
                           [cls_rows, 3 * args.d_model, args.d_model, args.cls_layers]],
                "epilogue": "bias_fp32_accumulate_fp16_output",
            },
            "attn_out": {
                "shapes": [[full_rows, args.d_model, args.d_model, args.full_layers],
                           [cls_rows, args.d_model, args.d_model, args.cls_layers]],
                "epilogue": "residual_fp32_accumulate_fp16_output",
            },
            "ff1": {
                "shapes": [[full_rows, args.ff_dim, args.d_model, args.full_layers],
                           [cls_rows, args.ff_dim, args.d_model, args.cls_layers]],
                "epilogue": "bias_silu_fp32_accumulate_fp16_output",
            },
            "ff2": {
                "shapes": [[full_rows, args.d_model, args.ff_dim, args.full_layers],
                           [cls_rows, args.d_model, args.ff_dim, args.cls_layers]],
                "epilogue": "residual_fp32_accumulate_fp16_output",
            },
        },
        "model_fingerprint": directory_fingerprint(args.weights),
    }

def build_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "gpu": query_gpu(),
        "dtype": args.dtype,
        "gemm_shapes": [
            {"m": args.b_micro * args.seq_len, "n": args.ff_dim, "k": args.d_model, "count": args.full_layers},
            {"m": args.b_micro, "n": args.ff_dim, "k": args.d_model, "count": args.cls_layers},
        ],
        "concurrency": args.concurrency,
        "epilogue": "bias_silu_fp32_accumulate_fp16_output",
        "model_fingerprint": directory_fingerprint(args.weights),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--puzzle-id", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--b-micro", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=51)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--full-layers", type=int, default=3)
    parser.add_argument("--cls-layers", type=int, default=1)
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    parser.add_argument("--policies", nargs="+", default=list(KNOWN_POLICIES))
    parser.add_argument("--policy-family", choices=tuple(POLICY_ENVIRONMENT), default="ff1")
    parser.add_argument("--graph-bench", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    policy_env_var = policy_environment_variable(args.policy_family)

    if args.repeats < 5:
        parser.error("--repeats must be at least 5")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    signature = build_signature(args)
    base_env = {
        "BEAM_WEIGHT_DIR": str(args.weights),
        "BEAM_STREAM_BENCH_REPORT": str(args.work_dir / "latest.md"),
        "BEAM_STREAM_MICRO_ONLY": "1",
        "BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH": "1",
        "BEAM_STREAM1_TRANSFORMER_B_MICRO": str(args.b_micro),
        "BEAM_STREAM1_TRANSFORMER_CONCURRENCY": str(args.concurrency),
        "BEAM_STREAM1_TRANSFORMER_BLOCK51": "1",
        "BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY": "1",
        "BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION": "0",
    }
    if not args.graph_bench:
        base_env.pop("BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH")

    observations: list[Observation] = []
    for policy in args.policies:
        if policy not in KNOWN_POLICIES:
            raise ValueError(f"unknown compiled policy: {policy}")
        policy_env = dict(base_env)
        policy_env[policy_env_var] = policy
        warmup = args.work_dir / f"{policy}-warmup.bin"
        run_benchmark(args.benchmark, args.puzzle_id, policy_env, warmup, policy_env_var)
        warmup.unlink(missing_ok=True)
        for repetition in range(args.repeats):
            dump = args.work_dir / f"{policy}-{repetition + 1}.bin"
            observations.append(run_benchmark(args.benchmark, args.puzzle_id, policy_env, dump, policy_env_var))

    decision = select_policy(observations, min_repeats=args.repeats)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "signature": signature,
        "signature_sha256": signature_digest(signature),
        **decision,
        "evidence": [observation.__dict__ for observation in observations],
    }
    write_cache_atomic(args.cache, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
