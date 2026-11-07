"""Mounted real Tetraminx FP16 artifact acceptance; no downloads or import-time GPU work."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback


def fixture_spec():
    return json.loads(Path(__file__).with_name("tetraminx_fixture.json").read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_export(directory, fixture=None):
    """Authenticate the exact mounted manifest and every numerical blob."""
    fixture, directory = fixture or fixture_spec(), Path(directory)
    expected = fixture["files"]
    observed = {p.name for p in directory.iterdir() if p.is_file() and
                (p.name == "manifest.json" or p.suffix in (".fp16", ".bf16"))}
    if observed != set(expected):
        raise ValueError(f"export file set differs from pinned Tetraminx fixture: {directory}")
    for name, record in expected.items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Tetraminx artifact checksum mismatch: {path}")
    return {"directory": str(directory), "files": expected, "verified": True}


def _integer_vector(value):
    if isinstance(value, str):
        value = [int(token.strip()) for token in value.strip().strip("[]").split(",")]
    if not isinstance(value, list) or any(type(v) is not int for v in value):
        raise ValueError("expected a literal integer state/permutation vector")
    return value


def _unique_object(pairs):
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate JSON keys would change native textual generator order")
    return dict(pairs)


def load_inputs(input_root, fixture=None):
    """Find the pinned export plus one distinct N88/G24 competition contract."""
    from cayleypy import CayleyGraphDef
    fixture, root = fixture or fixture_spec(), Path(input_root)
    exports, contracts = [], []
    for path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(manifest, dict) and all(manifest.get(k) == v for k, v in fixture["shape"].items()):
            exports.append((path.parent, verify_export(path.parent, fixture)))
    if not exports:
        raise FileNotFoundError("mount kernel output trydotatwo/cayleypy-2xt4-tetraminx-example with its export directory")
    for path in sorted(root.rglob("puzzle_info.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
            center, generators = _integer_vector(data["central_state"]), data["generators"]
            if len(center) != fixture["shape"]["state_len"] or not isinstance(generators, dict) or len(generators) != fixture["move_count"]:
                continue
            # Python JSON insertion order matches the native reader's textual order.
            definition = CayleyGraphDef.create([_integer_vector(g) for g in generators.values()],
                generator_names=list(generators), central_state=center, name=fixture["name"])
        except (KeyError, TypeError, ValueError, AssertionError):
            continue
        table = path.with_name("test.csv")
        if not table.is_file():
            continue
        with table.open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row.get("initial_state_id") == "0"]
        if len(rows) != 1:
            raise ValueError(f"test.csv must contain exactly one initial_state_id=0: {table}")
        start = _integer_vector(rows[0]["initial_state"])
        if Counter(start) != Counter(center):
            raise ValueError(f"real puzzle 0 has the wrong state multiset: {table}")
        identity = json.dumps([center, list(generators.items()), start], separators=(",", ":"))
        contracts.append((identity, definition, start, {"puzzle_info": str(path), "test_csv": str(table),
            "puzzle_info_sha256": file_sha256(path), "test_csv_sha256": file_sha256(table)}))
    if not contracts:
        raise FileNotFoundError("mount competition cayley-py-professor-tetraminx-solve-optimally containing puzzle_info.json and test.csv")
    if len({row[0] for row in contracts}) != 1:
        raise ValueError("multiple different N88/G24 contracts found; provide an unambiguous input_root")
    _, definition, start, provenance = contracts[0]
    return definition, start, exports[0][0], {"export": exports[0][1], "contract": provenance,
        "fixture_checkpoint_sha256": fixture["checkpoint_sha256"],
        "graph_binding": "mounted competition contract; fixture authenticates weights, not training-graph provenance"}


def replay(start, path, generators):
    state = tuple(start)
    for move in path:
        if type(move) is not int or not 0 <= move < len(generators):
            raise ValueError(f"invalid generator id: {move!r}")
        state = tuple(state[index] for index in generators[move])
    return state


def make_cases(definition, real_start, max_steps=100):
    generators = definition.generators_permutations
    inverse = {i: next((j for j, g in enumerate(generators) if
        tuple(g[generators[i][k]] for k in range(len(g))) == tuple(range(len(g)))), None)
        for i in (0, 2, 4)}
    if any(value is None for value in inverse.values()):
        raise ValueError("shallow fixture generators need explicit one-move inverses")
    cases = [{"name": "real_puzzle_0", "start": list(real_start), "max_steps": max_steps,
              "source": "competition test.csv initial_state_id=0", "must_solve": False}]
    for scramble in ((0,), (0, 2), (0, 2, 4)):
        start = replay(definition.central_state, scramble, generators)
        if start == tuple(definition.central_state):
            raise ValueError("shallow fixture accidentally starts at the goal")
        cases.append({"name": "shallow_" + str(len(scramble)), "start": list(start),
                      "max_steps": len(scramble), "scramble": list(scramble),
                      "reference_solution": [inverse[move] for move in reversed(scramble)],
                      "source": "deterministic shallow scramble on the real graph; plumbing only", "must_solve": True})
    return cases


def validate_outcome(case, path, metadata, graph, devices, beam_width, artifact_hash):
    from .smoke_cases import rank_evidence
    from cayleypy_native.contracts import GraphContract
    if path is not None and len(path) > case["max_steps"]:
        raise ValueError("native solution exceeds the max_steps budget with touch_bfs_radius=0")
    if path is not None and replay(case["start"], path, graph.definition.generators_permutations) != tuple(graph.definition.central_state):
        raise ValueError("independent real-graph replay failed")
    if case["must_solve"] and path is None:
        raise ValueError(f"complete shallow cohort did not solve: {case['name']}")
    if metadata.get("devices") != list(devices) or metadata.get("requested_beam_width") != beam_width:
        raise ValueError("native devices or requested beam differ from the experiment")
    if metadata.get("model_hash", metadata.get("model_artifact_hash")) != artifact_hash:
        raise ValueError("model artifact changed between preparation and execution")
    if metadata.get("graph_hash") != GraphContract.from_graph(graph, case["start"]).graph_hash:
        raise ValueError("native graph fingerprint differs from the experiment")
    effective = metadata.get("effective_beam_width")
    if type(effective) is not int or effective < beam_width:
        raise ValueError("effective beam must be observed and at least the requested global beam")
    if metadata.get("build", {}).get("shape") != {"state_len": 88, "storage_len": 96, "alignment": 16, "move_count": 24}:
        raise ValueError("runner shape is not the N88/G24 Tetraminx fixture")
    evidence = rank_evidence(Path(metadata["log_path"]).read_text(encoding="utf-8", errors="replace"), devices)
    if not evidence["passed"]:
        raise ValueError("native worker rank/device startup evidence is incomplete")
    return {"path_found": path is not None, "path": path, "path_length": len(path) if path is not None else 0,
            "replay_valid": True if path is not None else None, "rank_evidence": evidence}


def run_real_acceptance(source_dir, cutlass_dir, input_root, output_dir, devices=(0, 1), *,
                        beam_width=65536, max_steps=100, warm_repeats=3,
                        timeout_seconds=180, build_timeout_seconds=1800):
    """Return/save fail-fast JSON evidence. Call from a dedicated, unpatched process.

    No Torch speedup is claimed: direct and wrapper calls use identical native
    artifacts/options. Direct worker-call time excludes its recorded preparation.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "passed": False, "cases": [], "torch_baseline": "not_run",
              "comparison": "public wrapper API wall versus prepared direct native call; not a Torch speedup",
              "devices": list(devices), "beam_width": beam_width, "touch_bfs_radius": 0,
              "parent_residency": "CPU CayleyGraph and mounted NativeModel; no parent neural model on GPU"}

    def save():
        temporary = output / "real_acceptance_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "real_acceptance_report.json")

    installed = False
    try:
        import torch
        from cayleypy import CayleyGraph
        from cayleypy_native import dispatch, NativeModel, NativeOptions
        from cayleypy_native.backend import prepare_runtime, run_native
        from cayleypy_native.contracts import GraphContract
        from cayleypy_native.models import prepare_model
        if dispatch._ORIGINAL is not None:
            raise RuntimeError("real acceptance requires an unpatched dedicated process")
        if type(warm_repeats) is not int or warm_repeats < 1 or type(beam_width) is not int or beam_width < 65536 or type(max_steps) is not int or max_steps < 1:
            raise ValueError("need warm_repeats>=1, beam_width>=65536 and max_steps>=1")
        if len(devices) != 2 or len(set(devices)) != 2 or not torch.cuda.is_available():
            raise RuntimeError("real acceptance requires two distinct CUDA devices")
        names = [torch.cuda.get_device_name(d) for d in devices]
        if any(name not in ("Tesla T4", "NVIDIA T4") for name in names):
            raise RuntimeError(f"real acceptance is a 2xT4 experiment, observed {names}")
        report["hardware"] = {"gpu_names": names, "torch": torch.__version__, "cuda": torch.version.cuda,
                              "free_total_bytes_before": [list(torch.cuda.mem_get_info(d)) for d in devices]}
        definition, start, weights, report["inputs"] = load_inputs(input_root)
        graph = CayleyGraph(definition, device="cpu", random_seed=1729)
        model = NativeModel.for_graph(graph, weights)
        options = NativeOptions(source_dir=source_dir, cutlass_dir=cutlass_dir,
            cache_dir=output / "native-cache", devices=tuple(devices), touch_bfs_radius=0,
            timeout_seconds=timeout_seconds, build_timeout_seconds=build_timeout_seconds)
        if list(options.cache_dir.glob("builds/**/native-build.json")):
            raise ValueError("cold/warm experiment requires a fresh output directory")
        real, *shallow = make_cases(definition, start, max_steps)
        contract = GraphContract.from_graph(graph, start)
        prepared = prepare_model(model, contract, options, output / "artifact-validation")
        report["graph_hash"], report["artifact_hash"] = contract.graph_hash, prepared.artifact_hash
        dispatch.enable_native(options, default_backend="native")
        installed = True

        def run(case, route, phase, repetition):
            record = {"case": case, "route": route, "phase": phase, "repetition": repetition,
                      "status": "running"}
            report["cases"].append(record)
            save()
            if route == "direct":
                direct_dir = output / f"direct-{len(report['cases']):02d}"
                current = GraphContract.from_graph(graph, case["start"])
                prepared_at = time.perf_counter()
                runtime = prepare_runtime(current, prepared, options, direct_dir, devices)
                record["preparation_seconds_outside_direct_timer"] = time.perf_counter() - prepared_at
            began = time.perf_counter()
            if route == "wrapper":
                result = graph.beam_search(backend="native", start_state=case["start"], predictor=model,
                    beam_width=beam_width, max_steps=case["max_steps"], beam_mode="simple", return_path=True)
                record["wall_seconds"] = time.perf_counter() - began
                path, metadata = result.path, getattr(result, "native_metadata", {})
            else:
                outcome = run_native(current, prepared, options, beam_width, case["max_steps"], direct_dir, devices, runtime=runtime)
                record["wall_seconds"] = time.perf_counter() - began
                path = None if outcome.path is None else list(outcome.path)
                metadata = dict(outcome.metadata, effective_beam_width=outcome.effective_beam_width,
                                elapsed_seconds=outcome.elapsed_seconds, run_dir=str(outcome.run_dir))
            # Retain worker evidence even if public-result or replay/rank/budget
            # validation rejects the outcome below.
            record["native_metadata"] = metadata
            if route == "wrapper" and (getattr(result, "backend", None) != "native"
                    or bool(result.path_found) != (result.path is not None)
                    or result.path_length != (len(result.path) if result.path is not None else 0)):
                raise ValueError("public API returned a fallback or inconsistent path length")
            record["result"] = validate_outcome(case, path, metadata, graph, devices, beam_width, prepared.artifact_hash)
            record["configure_ran_this_call"] = (Path(metadata["run_dir"]) / "cmake-configure.log").is_file()
            record["status"] = "passed"
            save()

        run(real, "wrapper", "cold_first_call", 0)
        for repetition in range(warm_repeats):
            for route in (("direct", "wrapper") if repetition % 2 == 0 else ("wrapper", "direct")):
                run(real, route, "warm_build_cache_fresh_worker", repetition + 1)
        for case in shallow:
            run(case, "wrapper", "shallow_replay_cohort", 0)
        report["warm_median_wall_seconds"] = {route: statistics.median(row["wall_seconds"] for row in report["cases"]
            if row["route"] == route and row["phase"] == "warm_build_cache_fresh_worker") for route in ("wrapper", "direct")}
        report["real_puzzle_0_found"] = [row["result"]["path_found"] for row in report["cases"] if row["case"]["name"] == "real_puzzle_0"]
        report["status"], report["passed"] = "passed", True
    except Exception as error:
        report["status"] = "failed"
        report["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if report["cases"] and report["cases"][-1]["status"] == "running":
            report["cases"][-1]["status"] = "failed"
    finally:
        if installed:
            dispatch.disable_native()
        save()
    return report
