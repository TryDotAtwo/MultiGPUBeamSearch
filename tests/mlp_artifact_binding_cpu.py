"""Cross-language checks for the bytes actually consumed by the MLP loader.

Run against a C++ helper compiled with STATE_LEN=8, MOVE_COUNT=3. The fixture
uses Python's public artifact-hash byte order and intentionally changes valid,
finite tensors after computing that expected hash.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def fixture(directory, *, dtype, norm, blocks, hidden=8):
    directory.mkdir()
    manifest = dict(backend="mlp", dtype=dtype, state_len=8, num_classes=8,
                    hidden1=hidden, hidden2=hidden, residual_count=blocks, output_dim=3,
                    normalization=norm)
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (directory / "manifest.json").write_bytes(raw)
    counts = {"input_weight_hxk": 8 * 8 * hidden, "input_bias": hidden,
              "hidden_weight_hxk": hidden * hidden, "hidden_bias": hidden,
              "output_weight_hxk": hidden * 3, "output_bias": 3}
    layers = ["input", "hidden"]
    for block in range(blocks):
        for fc in (1, 2):
            prefix = f"residual{block}_fc{fc}"
            counts[prefix + "_weight_hxk"] = hidden * hidden
            counts[prefix + "_bias"] = hidden
            layers.append(prefix)
    if norm == "layernorm":
        for prefix in layers:
            counts[prefix + "_ln_gamma"] = counts[prefix + "_ln_beta"] = hidden
    graph = "a" * 64
    digest = hashlib.sha256(raw + graph.encode("ascii"))
    for index, (name, count) in enumerate(sorted(counts.items())):
        filename = f"{name}.{dtype}"
        # Both encodings interpret these bytes as finite values.
        data = bytes([index % 32, 0]) * count
        (directory / filename).write_bytes(data)
        digest.update(filename.encode("ascii"))
        digest.update(data)
    return graph, digest.hexdigest()


def invoke(binary, directory, graph=None, digest=None, mode=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("BEAM_MLP_GRAPH_HASH", "BEAM_MLP_ARTIFACT_HASH")}
    if graph is not None:
        env["BEAM_MLP_GRAPH_HASH"] = graph
    if digest is not None:
        env["BEAM_MLP_ARTIFACT_HASH"] = digest
    command = [str(binary), str(directory)] + ([mode] if mode else [])
    return subprocess.run(command, env=env,
                          text=True, capture_output=True, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--tmp", type=Path, required=True)
    args = parser.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        root = Path(tmp)
        for dtype, norm, blocks, hidden in (("fp16", "none", 1, 8),
                                            ("bf16", "layernorm", 12, 8),
                                            ("fp16", "layernorm", 1, 256)):
            weights = root / f"{dtype}-{norm}-{hidden}"
            graph, expected = fixture(weights, dtype=dtype, norm=norm, blocks=blocks, hidden=hidden)
            valid = invoke(args.binary, weights, graph, expected)
            assert valid.returncode == 0, valid.stderr
            assert invoke(args.binary, weights).returncode == 0, "legacy loading failed"

            tensor = weights / f"input_bias.{dtype}"
            original = tensor.read_bytes()
            owned = invoke(args.binary, weights, graph, expected, mode="owned")
            assert owned.returncode == 0 and "OWNED_BUFFER_BINDING_PASS" in owned.stdout, owned.stderr
            assert tensor.read_bytes() != original, "owned-buffer probe did not change the source file"
            tensor.write_bytes(original)
            tensor.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            changed = invoke(args.binary, weights, graph, expected)
            assert changed.returncode != 0 and "artifact hash mismatch" in changed.stderr, (
                "valid but changed tensor must be rejected", changed.stdout, changed.stderr)
            tensor.write_bytes(original)

            manifest = weights / "manifest.json"
            original_manifest = manifest.read_bytes()
            manifest.write_bytes(original_manifest + b" ")
            changed = invoke(args.binary, weights, graph, expected)
            assert changed.returncode != 0 and "artifact hash mismatch" in changed.stderr, (
                "changed manifest bytes must be rejected", changed.stdout, changed.stderr)
            manifest.write_bytes(original_manifest)

            for invalid_graph, invalid_digest in (("b" * 64, expected),
                                                   (graph, "c" * 64),
                                                   (graph, None), (None, expected),
                                                   (graph, "bad"), ("bad", expected),
                                                   (graph.upper(), expected), (graph, "")):
                rejected = invoke(args.binary, weights, invalid_graph, invalid_digest)
                assert rejected.returncode != 0, (invalid_graph, invalid_digest)
        print("MLP_ARTIFACT_BINDING_CPU_PASS")


if __name__ == "__main__":
    main()
