"""Matched Pancake-10 comparison using the upstream Pilgrim trainer and searcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE / "integrations/cayleypy_native"))
from multigpubeamsearch import NeighborConfig, PancakeGNN


PILGRIM_REV = "f02604fa7b665b82e5fbe5692b336a4fe4a01bdc"
TEST_SHA = "1250a897248a8731e775893e2f2c4910ef03a3c6ed9ac3b1cd210110d11c66bf"
N, DIM, HOPS, NEIGHBORS = 10, 32, 2, 3
EPOCHS, WALKERS, BATCH, K_MAX = 8, 256, 64, 15
BEAM, MAX_STEPS = 32, 30
ARMS = ("pilgrim", "value_graph", "transformer")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    if "rows" in data:
        # Keep each test start and its entire path together on a single line.
        fields = {key: value for key, value in data.items() if key != "rows"}
        header = json.dumps(fields, indent=2, allow_nan=False).removesuffix("\n}")
        rows = ",\n".join("    " + json.dumps(row, allow_nan=False) for row in data["rows"])
        payload = header + ',\n  "rows": [\n' + rows + "\n  ]\n}"
    else:
        payload = json.dumps(data, indent=2, allow_nan=False)
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def make_model(arm, seed):
    torch.manual_seed(seed)
    if arm == "pilgrim":
        from pilgrim import Pilgrim
        return Pilgrim(state_size=N, num_classes=N, hd1=1024, hd2=256, nrd=1, dropout_rate=0.0)
    neighbors = NeighborConfig(max_neighbors_per_hop=NEIGHBORS, num_hops=HOPS)
    model = PancakeGNN(N, DIM, 2, 0., neighbors, encoder_type=arm)
    if arm == "transformer":
        # The two Level-1 constructors consume different draws. Match all downstream tensors.
        common = make_model("value_graph", seed)
        for name in ("neighbor_agg", "value_head", "policy_head"):
            getattr(model, name).load_state_dict(getattr(common, name).state_dict())
    torch.nn.init.constant_(model.value_head[-1].bias, 9 * N / 11)
    return model


def train(out, seed, moves, inverse, goal, tests):
    from pilgrim import Trainer

    models = {arm: make_model(arm, seed).to(goal.device) for arm in ARMS}
    trainers = {arm: Trainer(net=model, num_epochs=EPOCHS, device=goal.device, batch_size=BATCH,
                             lr=.001, K_min=1, K_max=K_MAX, all_moves=moves,
                             inverse_moves=inverse, V0=goal, α=1) for arm, model in models.items()}
    test_indices = {tuple(row): i for i, row in enumerate(tests.tolist())}
    overlaps = set()
    hashes = []
    losses = {arm: [] for arm in ARMS}
    seconds = dict.fromkeys(ARMS, 0.0)
    for epoch in range(EPOCHS):
        torch.manual_seed(73000 + epoch)
        x, y = trainers["pilgrim"].generate_random_walks(k=WALKERS, K_min=1, K_max=K_MAX)
        digest = hashlib.sha256()
        for tensor in (x, y):
            digest.update(tensor.cpu().contiguous().numpy().tobytes())
        hashes.append(digest.hexdigest())
        for row in x.tolist():
            index = test_indices.get(tuple(row))
            if index is not None:
                overlaps.add(index)
        for arm, trainer in trainers.items():
            torch.manual_seed(seed + epoch)
            torch.cuda.synchronize()
            start = time.monotonic()
            loss = trainer._train_epoch(x, y.float())
            torch.cuda.synchronize()
            seconds[arm] += time.monotonic() - start
            losses[arm].append(loss)
        print("epoch", epoch + 1, {name: round(values[-1], 4) for name, values in losses.items()}, flush=True)
    steps = EPOCHS * WALKERS * K_MAX // BATCH
    arms = {}
    for arm, model in models.items():
        actual = {int(state["step"].item()) for state in trainers[arm].optimizer.state.values()}
        if actual != {steps}:
            raise RuntimeError(f"{arm}: expected {steps} Adam updates, observed {actual}")
        checkpoint = out / f"{arm}.pt"
        torch.save(model.cpu().state_dict(), checkpoint)
        arms[arm] = dict(parameters=sum(p.numel() for p in model.parameters()), optimizer_steps=steps,
                         losses=losses[arm], train_seconds=seconds[arm], weights_sha256=file_sha(checkpoint))
    write_json(out / "training.json", dict(seed=seed, pilgrim_commit=PILGRIM_REV, group="p044-t000",
               epochs=EPOCHS, walkers_per_depth=WALKERS, depths=[1, K_MAX], batch=BATCH, lr=.001,
               samples_per_arm=EPOCHS * WALKERS * K_MAX, neighbor_hops=HOPS, neighbors=NEIGHBORS,
               d_model=DIM, data_seeds=[73000 + e for e in range(EPOCHS)], data_sha256=hashes,
               test_indices_seen_in_training=sorted(overlaps), test_sha256=TEST_SHA,
               gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda, arms=arms))


def flipped(state, length):
    return state[:length][::-1] + state[length:]


def goal_ball():
    goal = tuple(range(N))
    distances = {goal: 0}
    layer = [goal]
    for depth in range(1, 6):
        following = []
        for state in layer:
            for length in range(2, N + 1):
                child = flipped(state, length)
                if child not in distances:
                    distances[child] = depth
                    following.append(child)
        layer = following
    return distances


def exact_distance(root, ball):
    layer, seen = {root}, {root}
    for depth in range(2 * N):
        hits = [ball[state] for state in layer if state in ball]
        if hits:
            return depth + min(hits)
        layer = {flipped(state, k) for state in layer for k in range(2, N + 1)} - seen
        seen.update(layer)
    raise RuntimeError("disconnected pancake graph")


def metrics(rows):
    solved = [row for row in rows if row["flips"] is not None]
    lengths = [len(row["flips"]) for row in solved]
    excess = [len(row["flips"]) - row["optimal"] for row in solved]
    return dict(instances=len(rows), solved=len(solved), replay_verified=len(solved),
                optimal_paths=excess.count(0), mean_length_solved=sum(lengths)/len(solved) if solved else None,
                mean_excess_solved=sum(excess)/len(solved) if solved else None)


def evaluate(out, seed, moves, names, goal, tests):
    from pilgrim import Searcher

    training = json.loads((out / "training.json").read_text(encoding="utf-8"))
    ball = goal_ball()
    starts = [tuple(row) for row in tests.tolist()]
    optimal = [exact_distance(state, ball) for state in starts]
    for arm in ARMS:
        checkpoint = out / f"{arm}.pt"
        if file_sha(checkpoint) != training["arms"][arm]["weights_sha256"]:
            raise ValueError(f"checkpoint digest mismatch: {checkpoint.name}")
        model = make_model(arm, seed)
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        model.eval().half().to(goal.device)
        if arm == "pilgrim":
            model.dtype = torch.float16
        torch.manual_seed(319)
        searcher = Searcher(model=model, all_moves=moves, V0=goal, device=goal.device)
        rows = []
        start = time.monotonic()
        with torch.inference_mode():
            for i, state in enumerate(starts):
                sequence, _ = searcher.get_solution(tests[i], B=BEAM, num_steps=MAX_STEPS, num_attempts=1)
                flips = None
                if sequence is not None:
                    flips = [int(names[j].rstrip("'")) for j in sequence.tolist()]
                    reconstructed = state
                    for length in flips:
                        if not 2 <= length <= N:
                            raise RuntimeError(f"illegal prefix length {length}")
                        reconstructed = flipped(reconstructed, length)
                    if reconstructed != tuple(range(N)) or len(flips) < optimal[i]:
                        raise RuntimeError(f"invalid returned path for {arm} instance {i}")
                rows.append(dict(index=i, state=state, optimal=optimal[i], flips=flips))
                if (i + 1) % 20 == 0:
                    print(arm, i + 1, metrics(rows)["solved"], flush=True)
        unseen = [r for r in rows if r["index"] not in training["test_indices_seen_in_training"]]
        write_json(out / f"{arm}-search.json", dict(seed=seed, beam=BEAM, max_steps=MAX_STEPS,
                   attempts=1, hash_seed=319, weights_sha256=file_sha(checkpoint),
                   test_sha256=TEST_SHA, first_ten=metrics(rows[:10]), all=metrics(rows),
                   state_disjoint=metrics(unseen), search_seconds=time.monotonic()-start, rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilgrim-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    root, out = args.pilgrim_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PILGRIM_REV:
        raise ValueError(f"expected Pilgrim {PILGRIM_REV}, got {commit}")
    sys.path.insert(0, str(root))
    from pilgrim import generate_inverse_moves
    torch.set_num_threads(2)
    device = torch.device("cuda")
    generators = json.loads((root/"generators/p044.json").read_text())
    names = generators["names"]
    moves = torch.tensor(generators["actions"], device=device)
    inverse = torch.tensor(generate_inverse_moves(names), device=device)
    goal = torch.load(root/"targets/p044-t000.pt", weights_only=True, map_location=device)
    dataset = root/"datasets/p044-t000-rnd.pt"
    if file_sha(dataset) != TEST_SHA:
        raise ValueError("unexpected p044 test data")
    tests = torch.load(dataset, weights_only=True, map_location=device)
    # Trainer creates logs/ and weights/ on construction. Keep its unused directories local to this run.
    original = Path.cwd()
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        try:
            os.chdir(scratch)
            if args.stage in ("train", "all"):
                train(out, args.seed, moves, inverse, goal, tests)
            if args.stage in ("evaluate", "all"):
                evaluate(out, args.seed, moves, names, goal, tests)
        finally:
            os.chdir(original)


if __name__ == "__main__":
    main()
