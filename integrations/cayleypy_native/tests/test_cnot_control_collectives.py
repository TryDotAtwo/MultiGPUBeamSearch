from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from multigpubeamsearch import cnot_stream5, rust_control
from multigpubeamsearch.cnot_collectives import World


class RecordingTransport:
    def __init__(self):
        self.calls = []

    def sum(self, tensor):
        self.calls.append(("sum", tensor))
        return tensor

    def gather_fixed(self, tensor):
        self.calls.append(("gather_fixed", tensor))
        return torch.stack((tensor, tensor + 10))

    def broadcast(self, tensor, source):
        self.calls.append(("broadcast", tensor, source))
        return tensor


def _two_rank_world():
    world = World("cpu")
    world.rank = 0
    world.size = 2
    return world


def test_control_collectives_use_the_selected_rust_transport():
    world = _two_rank_world()
    transport = RecordingTransport()
    world.transport = transport
    value = torch.tensor([1, 2], dtype=torch.int64)

    assert world.sum(value) is value
    assert torch.equal(world.gather_fixed(value), torch.tensor([[1, 2], [11, 12]]))
    assert world.broadcast(value, 1) is value

    assert [(call[0]) for call in transport.calls] == [
        "sum",
        "gather_fixed",
        "broadcast",
    ]
    assert transport.calls[0][1] is value
    assert transport.calls[1][1] is value
    assert transport.calls[2][1:] == (value, 1)


def test_control_collectives_keep_the_torch_fallback_without_rust_transport(monkeypatch):
    world = _two_rank_world()
    calls = []
    value = torch.tensor([3, 4], dtype=torch.int64)

    def all_reduce(tensor):
        calls.append(("sum", tensor))

    def all_gather(parts, tensor):
        calls.append(("gather_fixed", tensor))
        parts[0].copy_(tensor)
        parts[1].copy_(tensor + 5)

    def broadcast(tensor, source):
        calls.append(("broadcast", tensor, source))

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    assert world.sum(value) is value
    assert torch.equal(world.gather_fixed(value), torch.tensor([[3, 4], [8, 9]]))
    assert world.broadcast(value, 1) is value
    assert [call[0] for call in calls] == ["sum", "gather_fixed", "broadcast"]


def test_checked_stage_delegates_rank_error_coordination_to_rust(monkeypatch):
    world = _two_rank_world()
    calls = []
    expected = object()

    def run_native_checked(actual_world, stage, callback):
        calls.append((actual_world, stage))
        return callback()

    monkeypatch.setattr(
        rust_control,
        "extension",
        lambda: SimpleNamespace(run_cnot_native_checked=run_native_checked),
    )

    assert world.checked("prepare CNOT stage", lambda: expected) is expected
    assert calls == [(world, "prepare CNOT stage")]


def test_checked_stage_keeps_python_fallback_for_older_extensions(monkeypatch):
    world = _two_rank_world()
    expected = object()
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace())
    monkeypatch.setattr(world, "sum", lambda tensor: tensor)

    assert world.checked("prepare CNOT stage", lambda: expected) is expected


def test_rust_transport_requires_exchange_and_collective_abi(monkeypatch):
    class NativeLibrary:
        def __init__(self, _path):
            self.supports_exchange = True
            self.supports_collectives = False

    monkeypatch.setattr(
        rust_control,
        "extension",
        lambda: SimpleNamespace(CnotStream5Library=NativeLibrary),
    )
    native = SimpleNamespace(_name=str(Path(__file__).resolve()))

    assert cnot_stream5.RustStream5.compatible_library(native) is None


def test_rust_transport_is_available_when_both_abis_are_supported(monkeypatch):
    class NativeLibrary:
        def __init__(self, _path):
            self.supports_exchange = True
            self.supports_collectives = True

    monkeypatch.setattr(
        rust_control,
        "extension",
        lambda: SimpleNamespace(CnotStream5Library=NativeLibrary),
    )
    native = SimpleNamespace(_name=str(Path(__file__).resolve()))

    assert cnot_stream5.RustStream5.compatible_library(native) is not None

