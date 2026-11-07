"""Offline dispatch smoke; native requires genuine graph-compatible exported weights."""
import argparse
from pathlib import Path

from cayleypy import CayleyGraph, PermutationGroups, Predictor
from cayleypy_native import NativeModel, NativeOptions, enable_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "native", "torch"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--cutlass-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--q-model", action="store_true")
    args = parser.parse_args()
    graph = CayleyGraph(PermutationGroups.lrx(5), device=args.device, random_seed=42)
    predictor = Predictor(graph, "hamming")
    if args.weights_dir is not None:
        predictor = NativeModel.for_graph(graph, args.weights_dir, fallback=predictor)
    enable_native(NativeOptions(source_dir=args.source_dir, cutlass_dir=args.cutlass_dir, cache_dir=args.cache_dir))
    start = [1, 0, 2, 3, 4]
    search_options = {"use_child_scores": True} if args.q_model else {}
    result = graph.beam_search(start_state=start, predictor=predictor, backend=args.backend,
                              beam_width=1024, max_steps=4, return_path=True, **search_options)
    print(f"backend={getattr(result, 'backend', 'torch')} found={result.path_found} length={result.path_length}")
    if result.path_found:
        assert graph.apply_path(start, result.path).reshape(-1).tolist() == graph.central_state.tolist()
        print(f"path={result.get_path_as_string()} replay_valid=True")
    return 0 if result.path_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
