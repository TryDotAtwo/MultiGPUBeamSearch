"""Exercise the production C++ GNN feature loader on both trained encoder checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="completed comparison seed directory")
    parser.add_argument("--cutlass", type=Path, required=True, help="CUTLASS checkout")
    args = parser.parse_args()
    paths = args.out.resolve()
    root = Path(__file__).resolve().parents[2]
    constants = ["BEAM_STATE_LOGICAL_BYTES=10", "BEAM_MOVE_COUNT=9",
                 "BEAM_STATE_PHYSICAL_BYTES=16", "BEAM_STATE_ALIGNMENT=16"]
    if os.name == "nt":
        flags = ["/std:c++20", "/O1", "/utf-8"] + ["/D" + value for value in constants]
    else:
        flags = ["-std=c++20", "-O1"] + ["-D" + value for value in constants]
    os.environ.setdefault("MAX_JOBS", "2")
    # Register native operators before importing the Python exporter.
    load(name="issue5_trained_gnn_probe", sources=[
        str(root / "tools/stream1_gnn_projection_libtorch.cpp"),
        str(root / "cuda/stream1_gnn_projection.cu"),
        str(root / "tests/stream1_gnn_native_probe.cpp")],
        extra_include_paths=[str(args.cutlass.resolve() / "include")], extra_cflags=flags,
        extra_cuda_cflags=["-O2", "-DBEAM_HAS_CUTLASS=1"], is_python_module=False, verbose=True)

    from compare import N, make_model, write_json
    from multigpubeamsearch.contracts import GraphContract
    from multigpubeamsearch.gnn_artifacts import export_gnn
    identity = tuple(range(N))
    moves = tuple(identity[:k][::-1] + identity[k:] for k in range(2, N + 1))
    contract = GraphContract(N, N, moves, tuple(map(str, range(2, N+1))), identity, identity,
                             hashlib.sha256(repr(moves).encode()).hexdigest())
    torch.manual_seed(105)
    states = torch.stack([torch.randperm(N) for _ in range(17)]).cuda()
    gpu = torch.cuda.get_device_capability()
    results = {}
    with tempfile.TemporaryDirectory(dir=paths) as cache, torch.inference_mode():
        for arm in ("value_graph", "transformer"):
            model = make_model(arm, 0).eval()
            model.load_state_dict(torch.load(paths/f"{arm}.pt", weights_only=True, map_location="cpu"))
            artifact = export_gnn(model, contract, Path(cache)/arm)
            model.half().cuda()
            expected = model.features(states)
            results[arm] = {}
            for cutlass in ([False, True] if gpu[0] >= 8 else [False]):
                actual = torch.ops.multigpubeamsearch_gnn.probe_features(
                    states, str(artifact.weights_dir), cutlass)
                torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
                results[arm]["cutlass" if cutlass else "libtorch"] = (actual-expected).abs().max().item()
    receipt = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, states=len(states),
                   feature_max_abs=results, atol=.02, rtol=.02)
    write_json(paths/"native-features.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
