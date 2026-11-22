"""Recompute the aggregate from the nine saved searches and the public starts."""
import argparse
import json
from pathlib import Path

import torch

from compare import ARMS, N, TEST_SHA, exact_distance, file_sha, flipped, goal_ball, metrics, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    assert file_sha(args.dataset) == TEST_SHA
    starts = [tuple(s) for s in torch.load(args.dataset, weights_only=True, map_location="cpu").tolist()]
    assert len(starts) == len(set(starts)) == 100
    ball = goal_ball()
    distances = [exact_distance(s, ball) for s in starts]
    assert sum(distances[:10]) == 86 and sum(distances) == 857
    data_hashes = None
    per_seed = {}
    total = {arm: [] for arm in ARMS}
    paired = {arm: [] for arm in ARMS}
    for seed in (42, 43, 44):
        folder = args.results / f"seed{seed}"
        training = read(folder/"training.json")
        assert training["seed"] == seed and training["test_indices_seen_in_training"] == []
        if data_hashes is None:
            data_hashes = training["data_sha256"]
        assert training["data_sha256"] == data_hashes
        rows = {}
        for arm in ARMS:
            search = read(folder / f"{arm}-search.json")
            sha = file_sha(folder / f"{arm}.pt")
            assert training["arms"][arm]["weights_sha256"] == search["weights_sha256"] == sha
            assert search["beam"] == 32 and search["attempts"] == 1 and search["max_steps"] == 30
            assert search["seed"] == seed and search["test_sha256"] == TEST_SHA
            assert training["arms"][arm]["optimizer_steps"] == 480
            rows[arm] = search["rows"]
            assert len(rows[arm]) == 100
            for i, row in enumerate(rows[arm]):
                assert row["index"] == i and tuple(row["state"]) == starts[i]
                assert row["optimal"] == distances[i]
                if row["flips"] is None:
                    continue
                assert distances[i] <= len(row["flips"]) <= search["max_steps"]
                current = starts[i]
                for k in row["flips"]:
                    assert type(k) is int and 2 <= k <= N
                    current = flipped(current, k)
                assert current == tuple(range(N))
            assert metrics(rows[arm]) == search["all"]
            total[arm].extend(rows[arm])
        matched = [i for i in range(100) if all(rows[a][i]["flips"] is not None for a in ARMS)]
        for arm in ARMS:
            paired[arm].extend(rows[arm][i] for i in matched)
        per_seed[seed] = {arm: metrics(rows[arm]) for arm in ARMS}
    summary = dict(unique_test_states=100, model_seeds=[42,43,44], sum_optimal_distances=sum(distances),
                   train_test_overlap=0, per_seed=per_seed, aggregate={arm: metrics(total[arm]) for arm in ARMS},
                   solved_by_all_three={arm: metrics(paired[arm]) for arm in ARMS})
    write_json(args.results/"summary.json", summary)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
