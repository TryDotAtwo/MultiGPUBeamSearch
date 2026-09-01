"""Exercise the installed distribution on CPU, including released CayleyPy 0.1.0."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import tempfile
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, action="append", default=[])
    args = parser.parse_args()
    sys.path[:0] = [str(path.resolve()) for path in args.site]
    import cayleypy
    from cayleypy import CayleyGraph, PermutationGroups
    import cayleypy_native
    from cayleypy_native import enable_native, disable_native, NativeOptions, NativeFallbackWarning

    distribution = importlib.metadata.distribution("cayleypy-native")
    assert Path(cayleypy_native.__file__).resolve().is_relative_to(Path(distribution.locate_file("")).resolve())
    assert any("LICENSE" in str(path) for path in distribution.files)
    assert callable(cayleypy_native.setup_sources)
    original = CayleyGraph.beam_search
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", random_seed=42)
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache"
        enable_native(NativeOptions(cache_dir=cache))
        try:
            assert not cache.exists()  # Import/enable never download, build or write.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", NativeFallbackWarning)
                result = graph.beam_search(start_state=[1, 0, 2, 3, 4], return_path=True)
            assert any(isinstance(w.message, NativeFallbackWarning) for w in caught)
            assert result.path == [2]
            assert graph.apply_path([1, 0, 2, 3, 4], result.path).reshape(-1).tolist() == [0, 1, 2, 3, 4]
        finally:
            disable_native()
    assert CayleyGraph.beam_search is original
    print(json.dumps({"status": "passed", "adapter": distribution.version,
                      "module": str(Path(cayleypy_native.__file__).resolve()),
                      "public_hook": hasattr(cayleypy, "register_beam_search_backend"),
                      "cpu_path": result.path}))


if __name__ == "__main__":
    main()
