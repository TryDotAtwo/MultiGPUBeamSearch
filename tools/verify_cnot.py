"""Small, exhaustive reference for issue #6 receipts. Run directly or under torchrun."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import socket
import struct

import torch
import torch.distributed as dist

from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
from multigpubeamsearch.cnot_cli import initialize_world
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_scorer import TQPScorer


def rows_of(lanes, n):
    words = (n + 63) // 64
    return tuple(sum((lanes[i + j] % (1 << 64)) << (64 * j) for j in range(words))
                 for i in range(0, len(lanes), words))


def gate_pairs(n):
    return [(source, target) for source in range(n) for target in range(n) if target != source]


def apply(rows, pair):
    c, t = pair
    result = list(rows)
    result[t] ^= result[c]
    return tuple(result)


def float_value(bits):
    return struct.unpack("=d", struct.pack("=q", bits))[0]


def verify_layers(root, scorer, mode, width, k, layers, device):
    """Enumerate on Python integers. Compare the entire eligible union, path representatives,
    measured values, and every ordered survivor against the independent single-beam result."""
    n, closed = len(root), {root}
    pairs = gate_pairs(n)
    parents = [(root, ())]
    reports = []
    for layer in layers:
        if k < len(pairs):
            batch = torch.cat([pack_rows(rows, device=device) for rows, word in parents])
            head = scorer.parent_scores(batch, mode).tolist()
        eligible = {}
        for i, (rows, word) in enumerate(parents):
            actions = range(len(pairs))
            if k < len(pairs):
                actions = sorted(actions, key=lambda a: head[i][a], reverse=mode == "proposal")[:k]
            for action in actions:
                child = apply(rows, pairs[action])
                if child not in closed:
                    eligible.setdefault(child, (i, word + (action,)))
        measured = {}
        for bits, parent, _lex, _generation, route, *lanes in layer["candidates"]:
            child = rows_of(lanes, n)
            assert child not in measured and child in eligible, "duplicate or extraneous child"
            i, word = eligible[child]
            assert parent == i and (route & 65535) == word[-1], "wrong stable representative"
            measured[child] = float_value(bits)
        assert set(measured) == set(eligible), "candidate missing from the global union"
        if measured:
            children = torch.cat([pack_rows(rows, device=device) for rows in measured])
            actual = scorer.child_values(children).double().cpu()
            torch.testing.assert_close(actual, torch.tensor(list(measured.values()), dtype=torch.float64),
                                       rtol=1e-6, atol=1e-7)
        expected = sorted(eligible, key=lambda rows: (measured[rows], eligible[rows][1]))[:width]
        winners = [rows_of(item[5:], n) for item in layer["winners"]]
        assert winners == expected, "global top B differs"
        digest = hashlib.sha256(json.dumps(winners, separators=(",", ":")).encode()).hexdigest()
        reports.append(dict(depth=layer["depth"], distinct=len(eligible), retained=len(winners),
                            winners_sha256=digest, by_rank=layer["by_rank"]))
        parents = [(rows, eligible[rows][1]) for rows in expected]
        closed.update(winners)
    return reports


class WireFixture:
    """Diagnostic scores forcing one wide routed move; never used as the project model."""
    def __init__(self, undo):
        self.undo = undo

    def parent_scores(self, packed, mode):
        alphabet = packed.size(1) * (packed.size(1) - 1)
        q = torch.arange(alphabet, device=packed.device, dtype=torch.float64)
        q[self.undo] = -1
        return (q if mode == "qhat" else -q).expand(len(packed), -1)

    def child_values(self, children):
        # Popcount distance to the identity, using an independent byte lookup.
        n = children.size(1)
        eye = pack_rows([1 << j for j in range(n)], device=children.device)
        diff = (children ^ eye).contiguous().view(torch.uint8).reshape(len(children), -1).long()
        lut = torch.tensor([i.bit_count() for i in range(256)], device=children.device)
        return lut[diff].sum(1).double()


def wide_case():
    n = 144
    first, second = 130 * 143 + 2, 2 * 143 + 142
    return replay_cnot(n, (first, second)), WireFixture(second)


def validate_case(rows, scorer, mode, width, k, depth, device, require_cross_rank=False):
    world = World(device)
    result = search_cnot(pack_rows(rows, device=device), scorer, width=width, shortlist=k,
                         max_depth=depth, mode=mode, audit=True)
    def inspect():
        if world.rank:
            return None
        layers = verify_layers(rows, scorer, mode, width, k, result.layers, device)
        assert result.status == "solved"
        pairs = gate_pairs(len(rows))
        replay = tuple(1 << i for i in range(len(rows)))
        for action in result.gates:
            replay = apply(replay, pairs[action])
        assert replay == rows
        if require_cross_rank and world.size > 1:
            assert any(edge["move"] > 255 and len({edge["source"], edge["hash_owner"], edge["target"]}) > 1
                       for edge in result.path_routes), "large physical move never crossed a rank"
        assert any(layer["distinct"] > width and layer["retained"] == width and
                   (world.size == 1 or sum(rank[3] > 0 for rank in layer["by_rank"]) >= 2)
                   for layer in layers), "missing a saturated layer with participating owners"
        return dict(n=len(rows), width=width, shortlist=k, mode=mode, world=world.size,
                    scorer=type(scorer).__name__, model=getattr(scorer, "metadata", None),
                    layers=layers, physical_ids=list(result.gates), directed_gates=[pairs[i] for i in result.gates],
                    length=result.length, bitwise_replay=True, routes=result.path_routes)
    return world.checked("independent validation", inspect)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("trained", "wide"), required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--mode", choices=("qhat", "proposal"), default="qhat")
    parser.add_argument("--width", type=int)
    parser.add_argument("--library")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = initialize_world(args.backend)
    world = World(device)
    try:
        if args.case == "trained":
            rows, k, width = (1, 2, 7, 8), 11, 8 if args.width is None else args.width
            context = world.checked("load trained scorer", lambda:
                TQPScorer(args.library, args.manifest, n=len(rows), device=device, inference_batch=4))
        else:
            rows, scorer = wide_case()
            context, k, width = nullcontext(scorer), 3, 1 if args.width is None else args.width
        with context as scorer:
            output = validate_case(rows, scorer, args.mode, width, k, 3, device, args.case == "wide")
            hardware = dict(rank=world.rank, host=socket.gethostname(), device=str(device),
                            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
            devices = [None] * world.size
            if world.size > 1:
                dist.all_gather_object(devices, hardware)
            else:
                devices[0] = hardware
            if world.rank == 0:
                output.update(devices=devices, torch=str(torch.__version__), backend=args.backend)
                args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(output), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
