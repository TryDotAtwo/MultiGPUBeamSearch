"""Check trained weights and both levels against the author's pinned model definitions."""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile

import torch

from compare import ARMS, DIM, HOPS, N, NEIGHBORS, make_model, write_json
from multigpubeamsearch import NativeOptions
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.models import prepare_model, verify_prepared_model


SOURCE_SHA = "fed40964a215d3b6f1f514fe01eb5221b9be3e92fcdca902e0db5a68e971ea75"


def reference_definitions(path):
    # The original module unconditionally starts a training job on import. Load its
    # exact class/function definitions in a test-only module, without executing main.
    source = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError("unexpected reference source")
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (
        ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    spec = importlib.util.spec_from_file_location("issue5_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = reference_definitions(args.author_source.resolve())
    torch.set_num_threads(2)
    torch.manual_seed(711)
    states = torch.stack([torch.randperm(N) for _ in range(17)])
    identity = tuple(range(N))
    generators = tuple(identity[:k][::-1] + identity[k:] for k in range(2, N + 1))
    names = tuple(str(k) for k in range(2, N + 1))
    contract = GraphContract(N, N, generators, names, identity, tuple(states[0].tolist()),
                             hashlib.sha256(repr(generators).encode()).hexdigest())
    results = {}
    with tempfile.TemporaryDirectory(dir=args.out) as workspace, torch.inference_mode():
        for arm in ARMS[1:]:
            weights = torch.load(args.out / f"{arm}.pt", weights_only=True, map_location="cpu")
            ours = make_model(arm, 0).eval()
            upstream = reference.PancakeRLModel(
                N, DIM, arm, num_layers=2, dropout=0.,
                mpnn_config=reference.MPNNConfig(max_neighbors_per_hop=NEIGHBORS, num_hops=HOPS),
            ).eval()
            ours.load_state_dict(weights, strict=True)
            upstream.load_state_dict(weights, strict=True)
            expected, _ = upstream(states, use_neighbors=True)
            actual = ours(states)
            element_ref = upstream.encode_state(states)
            element = ours.state_encoder(states)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)
            torch.testing.assert_close(element, element_ref, rtol=2e-5, atol=3e-5)
            prepared = prepare_model(ours, contract, NativeOptions(), Path(workspace)/arm)
            verify_prepared_model(prepared, contract)
            entry = prepared.manifest['ensemble']['models'][0]
            backbone = prepared.weights_dir / entry['weights_dir'] / 'backbone.pt'
            with backbone.open("rb") as artifact:
                exported = torch.jit.load(artifact).eval().cuda()
            eager_gpu = ours.cuda().half()(states.cuda())
            scripted_gpu = exported(states.cuda())
            torch.testing.assert_close(scripted_gpu, eager_gpu, rtol=.002, atol=.005)
            torch.testing.assert_close(scripted_gpu.float().cpu(), expected, rtol=.01, atol=.03)
            results[arm] = dict(
                strict_state_dict=True, tensors=len(weights),
                element_max_abs=(element-element_ref).abs().max().item(),
                value_max_abs=(actual-expected).abs().max().item(),
                exported_vs_eager_fp16_cuda_max_abs=(scripted_gpu-eager_gpu).abs().max().item(),
                exported_cuda_vs_author_fp32_max_abs=(scripted_gpu.float().cpu()-expected).abs().max().item(),
            )
    write_json(args.out / "reference-parity.json", dict(states=len(states), source_sha256=SOURCE_SHA,
               gpu=torch.cuda.get_device_name(), torch=torch.__version__, results=results))


if __name__ == "__main__":
    main()
