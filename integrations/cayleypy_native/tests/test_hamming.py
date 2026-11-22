"""Independent scoring oracle for the fixed, untrained default heuristic."""
from itertools import permutations
from types import SimpleNamespace

import torch
import pytest
from cayleypy import CayleyGraph, PermutationGroups, Predictor
from cayleypy_native.models import prepare_model
from cayleypy_native.options import NativeOptions

from cayleypy_native.hamming import write_hamming_artifact


def test_hamming_matches_decoded_oracle_with_repeated_labels(tmp_path):
    center = (2, 0, 2, 1)
    contract = SimpleNamespace(state_len=4, num_classes=3, center=center, graph_hash="a" * 64)
    path = write_hamming_artifact(contract, tmp_path)
    states = torch.tensor(sorted(set(permutations(center))), dtype=torch.long)

    def read(name, shape):
        return torch.frombuffer(bytearray((path / (name + ".fp16")).read_bytes()),
                                dtype=torch.float16).reshape(shape).float()

    x = torch.nn.functional.one_hot(states, 4).flatten(1).float()
    x = torch.relu(x @ read("input_weight_hxk", (16, 8)) + read("input_bias", (8,)))
    x = torch.relu(x @ read("hidden_weight_hxk", (8, 8)) + read("hidden_bias", (8,)))
    r = torch.relu(x @ read("residual0_fc1_weight_hxk", (8, 8)) + read("residual0_fc1_bias", (8,)))
    x = torch.relu(x + r @ read("residual0_fc2_weight_hxk", (8, 8)) + read("residual0_fc2_bias", (8,)))
    actual = (x @ read("output_weight_hxk", (8, 1)) + read("output_bias", (1,))).flatten()
    expected = (states != torch.tensor(center)).sum(1)
    assert torch.equal(actual, expected.float())
    assert (actual == 0).sum().item() == 1


@pytest.mark.parametrize("explicit", [False, True])
def test_default_and_builtin_predictor_prepare_without_exporter(tmp_path, explicit):
    from cayleypy_native.contracts import GraphContract
    graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    contract = GraphContract.from_graph(graph, [1,0,2,3])
    predictor = Predictor(graph, "hamming") if explicit else None
    model = prepare_model(predictor, contract, NativeOptions(), tmp_path / "run")
    assert model.manifest["heuristic"] == "hamming"
    assert model.manifest["trained"] is False
