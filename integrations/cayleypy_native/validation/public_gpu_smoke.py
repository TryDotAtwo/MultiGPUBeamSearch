"""Reproducible two-GPU installation smoke with synthetic models and no datasets.

Run from an installed adapter environment. This explicit script downloads pinned
public build sources, compiles two state sizes, launches workers and saves logs.
It is correctness/integration evidence, not learned-model or speedup evidence.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import platform
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--require-public-hook", action="store_true")
    args = parser.parse_args()
    import torch
    import cayleypy
    from cayleypy import CayleyGraph, PermutationGroups
    from cayleypy_native import setup_sources, NativeOptions, prepare_native, enable_native, disable_native
    from cayleypy_native.sources import NATIVE_SOURCE, CUTLASS_SOURCE
    if __package__:
        from .smoke_cases import run_acceptance, make_model, rank_evidence, replay
        from .prepared_acceptance import run_prepared_acceptance
    else:
        from smoke_cases import run_acceptance, make_model, rank_evidence, replay
        from prepared_acceptance import run_prepared_acceptance

    if platform.system() != "Linux" or torch.cuda.device_count() < 2:
        raise RuntimeError("this script requires Linux with two CUDA GPUs; it does not simulate CUDA")
    if args.require_public_hook and not hasattr(cayleypy, "register_beam_search_backend"):
        raise RuntimeError("install the CayleyPy backend-hook revision for this acceptance run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    options = setup_sources(options=NativeOptions(cache_dir=args.cache_dir, devices=(0, 1),
                                                  timeout_seconds=180, build_timeout_seconds=1200))
    report = {"status": "running", "source_setup_seconds": time.perf_counter() - started,
              "native_revision": NATIVE_SOURCE.revision, "native_archive_sha256": NATIVE_SOURCE.sha256,
              "cutlass_revision": CUTLASS_SOURCE.revision, "cutlass_archive_sha256": CUTLASS_SOURCE.sha256,
              "public_hook": hasattr(cayleypy, "register_beam_search_backend")}
    original = CayleyGraph.beam_search
    shallow = run_acceptance(source_dir=options.source_dir, cutlass_dir=options.cutlass_dir,
                             output_dir=args.output_dir / "lrx8", cache_dir=options.cache_dir)
    if not shallow["passed"]:
        raise RuntimeError("LRX8 integration failed; see acceptance_report.json")
    report["lrx8_cases"] = len(shallow["cases"])
    prepared = run_prepared_acceptance(options.source_dir, options.cutlass_dir,
                                       args.output_dir / "prepared", options.cache_dir)
    if prepared["status"] != "passed":
        raise RuntimeError("prepared integration failed; see prepared_acceptance_report.json")
    # Independent second state size: this is not the previous trained Tetraminx
    # fixture. It exercises N88/storage96 without private data or model access.
    graph = CayleyGraph(PermutationGroups.lrx(88), device="cpu", random_seed=1729)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        model = make_model()
        model.state_size = model.num_classes = 88
        model.input_layer = torch.nn.Linear(88 * 88, 32)
        model.eval()
    prepared88 = prepare_native(graph, model, native_options=options)
    enable_native(prepared88.options, default_backend="native")
    try:
        if report["public_hook"] and CayleyGraph.beam_search is not original:
            raise RuntimeError("public hook unexpectedly replaced CayleyGraph.beam_search")
        start = list(graph.definition.central_state)
        start[0], start[1] = start[1], start[0]
        result = graph.beam_search(start_state=start, predictor=prepared88.model,
                                   beam_width=7, max_steps=1, return_path=True)
        assert result.backend == "native" and result.path_found and result.path_length == 1
        assert replay(start, result.path, graph.definition.generators) == tuple(graph.definition.central_state)
        evidence = rank_evidence(Path(result.native_metadata["log_path"]).read_text(), (0, 1))
        assert evidence["passed"]
        report["lrx88"] = {"path": result.path, "metadata": result.native_metadata, "rank_evidence": evidence}
    finally:
        disable_native()
    assert CayleyGraph.beam_search is original
    assert setup_sources(options=replace(options, source_dir=None, cutlass_dir=None), offline=True) == options
    report.update(status="passed", elapsed_seconds=time.perf_counter() - started)
    (args.output_dir / "public_gpu_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
