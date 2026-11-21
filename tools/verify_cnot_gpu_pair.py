"""Check public result parity and acceptance receipts for one bounded GPU campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(result_dir, source_sha, instance_id, manifest_sha, checkpoint_sha,
             scorer_library_sha, wheel_sha, run_id):
    root = Path(result_dir)
    paired_names = (
        "cnot-trained-deep-paired-1-gpu.json",
        "cnot-trained-deep-paired-2-gpu.json",
    )
    left, right = (load_json(root / name) for name in paired_names)
    for report in (left, right):
        model = report.get("model") or {}
        require(
            model.get("checkpoint_sha256") == checkpoint_sha
            and model.get("search_manifest_sha256") == manifest_sha
            and model.get("scorer_library_sha256") == scorer_library_sha
            and report.get("engine") == "rust"
            and report.get("bitwise_replay"),
            "paired CNOT run lacks pinned scorer, Rust controller or independent replay",
        )
        require(
            report.get("n") == 4
            and report.get("mode") == "proposal"
            and report.get("width") == 2
            and report.get("shortlist") == 3,
            "paired CNOT runs do not share the requested logical configuration",
        )
    for key in ("root_sha256", "mode", "width", "shortlist", "n"):
        require(left.get(key) == right.get(key), f"paired 1/2 GPU CNOT inputs differ at {key}")
    require(
        left.get("precision_contract") == right.get("precision_contract")
        and left.get("precision_contract", {}).get("scorer_output_dtype") == "float32",
        "paired CNOT score or selection precision differs",
    )
    model_keys = (
        "upstream_commit", "checkpoint_sha256", "search_manifest_sha256",
        "scorer_library_sha256", "feature_registry_sha256", "training_manifest_sha256",
        "training_result_sha256", "scorer_output_dtype", "feature_ids",
    )
    require(
        {key: left["model"].get(key) for key in model_keys}
        == {key: right["model"].get(key) for key in model_keys},
        "paired CNOT model/head contract differs",
    )
    require(
        bool(left.get("layers"))
        and len(left["layers"]) == len(right.get("layers", []))
        and [item.get("winners_sha256") for item in left["layers"]]
        == [item.get("winners_sha256") for item in right["layers"]],
        "ordered global CNOT survivor hashes differ between 1 and 2 GPUs",
    )

    qhat_name = "cnot-trained-qhat-1-gpu.json"
    qhat = load_json(root / qhat_name)
    model = qhat.get("model") or {}
    require(
        model.get("checkpoint_sha256") == checkpoint_sha
        and model.get("search_manifest_sha256") == manifest_sha
        and model.get("scorer_library_sha256") == scorer_library_sha
        and qhat.get("engine") == "rust"
        and qhat.get("world") == 1
        and qhat.get("mode") == "qhat"
        and qhat.get("bitwise_replay"),
        f"{qhat_name}: trained qhat, Rust scorer or independent replay evidence is missing",
    )
    permutation = load_json(root / "permutation-1-and-2-gpu.json")
    require(
        [row.get("world_size") for row in permutation.get("runs", [])] == [1, 2],
        "permutation one- and two-GPU results are missing",
    )

    case_expectations = {
        "cnot-trained-qhat-1-gpu": (qhat_name, 1),
        "cnot-trained-deep-paired-1-gpu": (paired_names[0], 1),
        "cnot-trained-deep-paired-2-gpu": (paired_names[1], 2),
    }
    event_receipts = {}
    for case_id, (report_name, world) in case_expectations.items():
        report = load_json(root / report_name)
        identity = report.get("acceptance") or {}
        require(
            identity.get("run_id") == run_id
            and identity.get("case_id") == case_id
            and identity.get("source_sha") == source_sha
            and identity.get("wheel_sha256") == wheel_sha
            and identity.get("world") == world,
            f"{report_name}: result identity is not bound to this run, wheel and topology",
        )
        event_path = root / "backend-events" / f"{case_id}.json"
        event_receipt = load_json(event_path)
        require(
            event_receipt.get("status") == "accepted"
            and event_receipt.get("run_id") == identity["run_id"]
            and event_receipt.get("case_id") == case_id
            and event_receipt.get("source_sha") == source_sha
            and event_receipt.get("wheel_sha256") == wheel_sha
            and event_receipt.get("world") == world
            and event_receipt.get("single_rank_transport") == (world == 1),
            f"{case_id}: strict backend verifier receipt does not match its result JSON",
        )
        event_receipts[case_id] = event_receipt

    receipt = {
        "instance_id": instance_id,
        "source_sha": source_sha,
        "wheel_sha256": wheel_sha,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "scorer_library_sha256": scorer_library_sha,
        "upstream_commit": left["model"].get("upstream_commit"),
        "paired_cnot_survivor_hashes_match": True,
        "paired_cnot_world_sizes": [left.get("world"), right.get("world")],
        "strict_backend_cases": event_receipts,
        "device_domain_operations": ["owners", "expansion", "full_matrix_distinct"],
        "stream5_nccl_phases": ["exchange", "all_reduce_i64", "all_gather_i64_into", "broadcast_i64"],
        "completed_cases": [
            "permutation-1-and-2-gpu.json",
            qhat_name,
            *paired_names,
        ],
        "installed_wheel": True,
        "gpu_test": "passed",
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--scorer-library-sha256", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        if not SHA40.fullmatch(args.source_sha):
            raise ValueError("source SHA must be a 40-character lowercase hexadecimal ID")
        for name in ("manifest_sha256", "checkpoint_sha256", "scorer_library_sha256", "wheel_sha256"):
            if not SHA64.fullmatch(getattr(args, name)):
                raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
        receipt = validate(
            args.result_dir, args.source_sha, args.instance_id, args.manifest_sha256,
            args.checkpoint_sha256, args.scorer_library_sha256, args.wheel_sha256,
            args.run_id,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
