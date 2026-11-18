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

    def exchange(self, rows, counts):
        self.calls.append(("exchange", rows, counts))
        return rows


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


def test_rust_world_collective_abi_is_exposed():
    native = rust_control.extension()
    assert all(callable(getattr(native, name, None)) for name in (
        "run_cnot_native_sum",
        "run_cnot_native_gather_fixed",
        "run_cnot_native_broadcast",
    ))


def test_world_control_collectives_delegate_to_rust_when_available(monkeypatch):
    world = _two_rank_world()
    world.transport = RecordingTransport()
    value = torch.tensor([5, 6], dtype=torch.int64)
    calls = []

    def sum_rows(actual_world, tensor):
        calls.append(("sum", actual_world, tensor))
        return tensor

    def gather_rows(actual_world, tensor):
        calls.append(("gather_fixed", actual_world, tensor))
        return torch.stack((tensor, tensor + 20))

    def broadcast_rows(actual_world, tensor, source):
        calls.append(("broadcast", actual_world, tensor, source))
        return tensor

    monkeypatch.setattr(
        rust_control,
        "extension",
        lambda: SimpleNamespace(
            run_cnot_native_sum=sum_rows,
            run_cnot_native_gather_fixed=gather_rows,
            run_cnot_native_broadcast=broadcast_rows,
        ),
    )

    assert world.sum(value) is value
    assert torch.equal(world.gather_fixed(value), torch.tensor([[5, 6], [25, 26]]))
    assert world.broadcast(value, 1) is value
    assert [call[0] for call in calls] == ["sum", "gather_fixed", "broadcast"]
    assert all(call[1] is world for call in calls)


def test_control_collectives_preserve_the_python_path_for_older_extensions(monkeypatch):
    world = _two_rank_world()
    transport = RecordingTransport()
    world.transport = transport
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace())
    value = torch.tensor([7, 8], dtype=torch.int64)

    assert world.sum(value) is value
    assert torch.equal(world.gather_fixed(value), torch.tensor([[7, 8], [17, 18]]))
    assert world.broadcast(value, 1) is value
    assert [call[0] for call in transport.calls] == ["sum", "gather_fixed", "broadcast"]


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


def test_rust_world_exchange_abi_is_exposed():
    native = rust_control.extension()
    assert callable(getattr(native, "run_cnot_native_world_exchange", None))


def test_world_exchange_delegates_to_rust_and_preserves_packet_flag(monkeypatch):
    world = _two_rank_world()
    world.transport = RecordingTransport()
    rows = torch.tensor([[5, 6]], dtype=torch.int64)
    targets = torch.tensor([1], dtype=torch.int64)
    expected = {
        False: torch.tensor([[10, 11]], dtype=torch.int64),
        True: torch.tensor([[20, 21]], dtype=torch.int64),
    }
    calls = []

    def exchange(actual_world, actual_rows, actual_targets, native_packet):
        calls.append((actual_world, actual_rows, actual_targets, native_packet))
        return expected[native_packet]

    monkeypatch.setattr(
        rust_control,
        "extension",
        lambda: SimpleNamespace(run_cnot_native_world_exchange=exchange),
    )

    assert world.exchange(rows, targets) is expected[False]
    assert world.exchange(rows, targets, native_packet=True) is expected[True]
    assert calls == [
        (world, rows, targets, False),
        (world, rows, targets, True),
    ]
    assert world.transport.calls == []


def test_world_exchange_preserves_python_fallback_for_older_extensions(monkeypatch):
    world = _two_rank_world()
    transport = RecordingTransport()
    world.transport = transport
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace())
    rows = torch.tensor([[30, 31], [10, 11], [20, 21]], dtype=torch.int64)
    targets = torch.tensor([1, 0, 1], dtype=torch.int64)

    result = world.exchange(rows, targets)

    assert result.tolist() == [[10, 11], [30, 31], [20, 21]]
    assert [call[0] for call in transport.calls] == ["exchange"]
    assert transport.calls[0][1].tolist() == [[10, 11], [30, 31], [20, 21]]
    assert transport.calls[0][2].tolist() == [1, 2]


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


def _configure_row_gather(world, local_rows, calls, remote_rows):
    world.rank = 0
    world.size = 2
    world.longs = lambda items: torch.as_tensor(items, dtype=torch.int64)

    def gather_fixed(tensor):
        calls.append(("gather_fixed", tensor.tolist()))
        counts = torch.tensor([len(local_rows), len(remote_rows)], dtype=tensor.dtype)
        return counts.unsqueeze(1)

    def broadcast(tensor, source):
        calls.append(("broadcast", source, tuple(tensor.shape)))
        if source == 1:
            tensor.copy_(remote_rows)
        return tensor

    world.gather_fixed = gather_fixed
    world.broadcast = broadcast


def test_rust_gather_rows_preserves_order_and_empty_rank():
    gather = getattr(rust_control.extension(), "run_cnot_native_gather_rows")
    for local_rows, expected_calls in (
        (torch.tensor([[10, 11], [12, 13]], dtype=torch.int64),
         [("broadcast", 0, (2, 2)), ("broadcast", 1, (1, 2))]),
        (torch.empty((0, 2), dtype=torch.int64),
         [("broadcast", 1, (1, 2))]),
    ):
        world = _two_rank_world()
        calls = []
        remote_rows = torch.tensor([[20, 21]], dtype=torch.int64)
        _configure_row_gather(world, local_rows, calls, remote_rows)

        result = gather(world, local_rows)

        assert result.tolist() == local_rows.tolist() + remote_rows.tolist()
        assert calls[0] == ("gather_fixed", [len(local_rows)])
        assert calls[1:] == expected_calls


def test_world_gather_rows_uses_the_rust_abi_when_available(monkeypatch):
    world = _two_rank_world()
    local_rows = torch.tensor([[1, 2]], dtype=torch.int64)
    native_calls = []
    python_calls = []
    _configure_row_gather(
        world, local_rows, python_calls, torch.tensor([[3, 4]], dtype=torch.int64)
    )
    expected = object()

    def gather(actual_world, rows):
        native_calls.append((actual_world, rows))
        return expected

    monkeypatch.setattr(
        rust_control,
        "extension",
        lambda: SimpleNamespace(run_cnot_native_gather_rows=gather),
    )

    assert world.gather_rows(local_rows) is expected
    assert native_calls == [(world, local_rows)]
    assert python_calls == []


def test_world_gather_rows_keeps_python_fallback_for_older_extensions(monkeypatch):
    world = _two_rank_world()
    local_rows = torch.tensor([[1, 2]], dtype=torch.int64)
    python_calls = []
    remote_rows = torch.tensor([[3, 4]], dtype=torch.int64)
    _configure_row_gather(world, local_rows, python_calls, remote_rows)
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace())

    assert world.gather_rows(local_rows).tolist() == [[1, 2], [3, 4]]
    assert python_calls == [
        ("gather_fixed", [1]),
        ("broadcast", 0, (1, 2)),
        ("broadcast", 1, (1, 2)),
    ]


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


def test_rust_world_start_transport_abi_is_exposed():
    native = rust_control.extension()
    assert callable(getattr(native, "run_cnot_native_start_transport", None))


def test_rust_world_native_stage_support_abi_is_exposed():
    native = rust_control.extension()
    assert callable(getattr(native, "run_cnot_native_all_ranks_support", None))


def test_world_native_stage_support_delegates_to_rust_when_available(monkeypatch):
    world = _two_rank_world()
    calls = []

    def support(actual_world, local_support):
        calls.append((actual_world, local_support))
        return local_support

    monkeypatch.setattr(
        rust_control,
        "extension",
        lambda: SimpleNamespace(run_cnot_native_all_ranks_support=support),
    )

    assert world.all_ranks_support(True) is True
    assert calls == [(world, True)]


def test_world_native_stage_support_keeps_python_fallback_for_older_extensions(monkeypatch):
    world = _two_rank_world()
    collectives = []

    def longs(values):
        collectives.append(("longs", list(values)))
        return torch.tensor(values, dtype=torch.int64)

    def sum(tensor):
        collectives.append(("sum", tensor.tolist()))
        return torch.tensor([1], dtype=torch.int64)

    monkeypatch.setattr(world, "longs", longs)
    monkeypatch.setattr(world, "sum", sum)
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace())

    assert world.all_ranks_support(False) is False
    assert collectives == [("longs", [0]), ("sum", [0])]


def test_rust_native_stage_support_requires_all_rank_votes(monkeypatch):
    native = rust_control.extension()
    support = getattr(native, "run_cnot_native_all_ranks_support", None)
    assert callable(support)

    world = _two_rank_world()
    votes = []
    totals = iter((1, 2, 0))

    def longs(values):
        votes.append(("longs", list(values)))
        return torch.tensor(values, dtype=torch.int64)

    def sum(tensor):
        votes.append(("sum", tensor.tolist()))
        return torch.tensor([next(totals)], dtype=torch.int64)

    monkeypatch.setattr(world, "longs", longs)
    monkeypatch.setattr(world, "sum", sum)

    assert support(world, False) is False
    assert support(world, True) is True
    assert support(world, True) is False
    assert votes == [
        ("longs", [0]), ("sum", [0]),
        ("longs", [1]), ("sum", [1]),
        ("longs", [1]), ("sum", [1]),
    ]


def test_world_start_transport_delegates_to_rust_when_available(monkeypatch):
    world = _two_rank_world()
    world._stream5_library = object()
    compatible = object()
    selected = object()
    calls = []

    class FakeRustStream5:
        @staticmethod
        def compatible_library(_library):
            return compatible

        def __new__(cls, _library, _world):
            return object()

    class FakeStream5:
        def __new__(cls, _library, _world):
            return object()

    def start(actual_world, prefer_rust):
        calls.append((actual_world, prefer_rust))
        actual_world.transport = selected

    monkeypatch.setattr(cnot_stream5, "RustStream5", FakeRustStream5)
    monkeypatch.setattr(cnot_stream5, "Stream5", FakeStream5)
    monkeypatch.setattr(world, "sum", lambda tensor: torch.full_like(tensor, world.size))
    monkeypatch.setattr(
        rust_control,
        "extension",
        lambda: SimpleNamespace(run_cnot_native_start_transport=start),
    )

    world.start_transport(prefer_rust=True)

    assert calls == [(world, True)]
    assert world.transport is selected


def test_world_start_transport_keeps_python_path_for_older_extensions(monkeypatch):
    world = _two_rank_world()
    world._stream5_library = object()
    compatible = object()
    selected = object()

    class FakeRustStream5:
        @staticmethod
        def compatible_library(_library):
            return compatible

        def __new__(cls, _library, _world):
            return selected

    class FakeStream5:
        def __new__(cls, _library, _world):
            return object()

    monkeypatch.setattr(cnot_stream5, "RustStream5", FakeRustStream5)
    monkeypatch.setattr(cnot_stream5, "Stream5", FakeStream5)
    monkeypatch.setattr(world, "sum", lambda tensor: torch.full_like(tensor, world.size))
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace())

    world.start_transport(prefer_rust=True)

    assert world.transport is selected


def test_rust_start_transport_preserves_rank_consensus_and_fallback(monkeypatch):
    native = rust_control.extension()
    start = getattr(native, "run_cnot_native_start_transport", None)
    assert callable(start)

    world = _two_rank_world()
    native_library = object()
    world._stream5_library = native_library
    library_by_rank = [None]
    consensus = [1]
    collectives = []
    selected = []

    class FakeRustStream5:
        @staticmethod
        def compatible_library(_library):
            return library_by_rank[0]

        def __new__(cls, library, actual_world):
            transport = object()
            selected.append(("rust", library, actual_world, transport))
            return transport

    class FakeStream5:
        def __new__(cls, library, actual_world):
            transport = object()
            selected.append(("python", library, actual_world, transport))
            return transport

    def longs(values):
        collectives.append(("longs", list(values)))
        return torch.tensor(values, dtype=torch.int64)

    def sum(tensor):
        collectives.append(("sum", tensor.tolist()))
        return torch.tensor([consensus[0]], dtype=torch.int64)

    monkeypatch.setattr(cnot_stream5, "RustStream5", FakeRustStream5)
    monkeypatch.setattr(cnot_stream5, "Stream5", FakeStream5)
    monkeypatch.setattr(world, "longs", longs)
    monkeypatch.setattr(world, "sum", sum)

    start(world, True)
    assert selected[-1][0] == "python"
    assert world.transport is selected[-1][3]
    assert collectives == [("longs", [0]), ("sum", [0])]

    library_by_rank[0] = object()
    consensus[0] = 2
    start(world, True)
    assert selected[-1][0] == "rust"
    assert selected[-1][1] is library_by_rank[0]
    assert world.transport is selected[-1][3]

    before = len(collectives)
    start(world, False)
    assert selected[-1][0] == "python"
    assert world.transport is selected[-1][3]
    assert len(collectives) == before

