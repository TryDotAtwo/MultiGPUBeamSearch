#!/usr/bin/env python3
"""CPU fixtures for reference-score ordering in CNOT acceptance."""
from __future__ import annotations

import hashlib
import json
import struct
import unittest

import torch

from verify_cnot import apply, gate_pairs, verify_layers


def score_bits(value):
    return struct.unpack("=q", struct.pack("=d", float(value)))[0]


def state_digest(states):
    payload = json.dumps(states, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TableScorer:
    def __init__(self, root, reference_scores):
        self.by_state = {}
        for action, pair in enumerate(gate_pairs(len(root))):
            self.by_state[apply(root, pair)] = float(reference_scores[action])

    def child_values(self, children):
        values = []
        for child in children:
            state = tuple(int(value) for value in child.reshape(-1).tolist())
            values.append(self.by_state[state])
        return torch.tensor(values, dtype=torch.float64, device=children.device)


def make_layer(root, reported_scores, reference_scores, width, candidate_order=None):
    pairs = gate_pairs(len(root))
    children = {action: apply(root, pair) for action, pair in enumerate(pairs)}
    order = list(range(len(pairs))) if candidate_order is None else list(candidate_order)
    candidates = [
        (score_bits(reported_scores[action]), 0, action, action, action, *children[action])
        for action in order
    ]
    winner_actions = sorted(range(len(pairs)), key=lambda action: (reported_scores[action], (action,)))[:width]
    winners = [(0, 0, 0, 0, 0, *children[action]) for action in winner_actions]
    layer = {"depth": 0, "candidates": candidates, "winners": winners, "by_rank": []}
    scorer = TableScorer(root, reference_scores)
    return children, scorer, layer, winner_actions


def inspect(root, reference_scores, reported_scores, width, candidate_order=None):
    children, scorer, layer, winner_actions = make_layer(
        root, reported_scores, reference_scores, width, candidate_order
    )
    report = verify_layers(
        root, scorer, "qhat", width, len(gate_pairs(len(root))), [layer], torch.device("cpu")
    )[0]
    winner_states = [children[action] for action in winner_actions]
    return report, state_digest(winner_states)


class CnotReferenceOrderingTests(unittest.TestCase):
    def test_exact_reference_ties_use_the_full_path_word_not_candidate_arrival_order(self):
        root = (1, 2, 4)
        scores = [5.0] * len(gate_pairs(len(root)))
        report, digest = inspect(root, scores, scores, width=2, candidate_order=reversed(range(6)))
        self.assertEqual(report["selection_oracle"], "reported_scores")
        self.assertEqual(report["reference_order_status"], "exact_tie_full_word")
        self.assertEqual(report["reference_ordered_winners_sha256"], digest)

    def test_overlapping_reference_boundary_is_reported_as_ambiguous(self):
        root = (1, 2, 4)
        reference = [10.0, 10.000001, 20.0, 21.0, 22.0, 23.0]
        reported = [10.0000008, 10.0000002, 20.0, 21.0, 22.0, 23.0]
        report, _ = inspect(root, reference, reported, width=1)
        self.assertEqual(report["reference_order_status"], "ambiguous_within_score_tolerance")
        self.assertAlmostEqual(report["reference_boundary_gap"], 0.000001)

    def test_separated_reference_boundary_is_recorded(self):
        root = (1, 2, 4)
        reference = [10.0, 11.0, 20.0, 21.0, 22.0, 23.0]
        report, digest = inspect(root, reference, reference, width=1)
        self.assertEqual(report["reference_order_status"], "reference_separated")
        self.assertEqual(report["reference_ordered_winners_sha256"], digest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
