"""C++ CNOT owner grouping preserves stable records and empty destinations."""
import pytest
import torch

from multigpubeamsearch import cnot_native
from multigpubeamsearch.cnot_collectives import World


class CaptureTransport:
    def exchange(self, send, counts):
        return send, counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("owners", [[], [0], [2, 0, 2, 0, 2, 1], [3, 3, 3]])
def test_native_exchange_packet_matches_reference(owners):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    targets = torch.tensor(owners, dtype=torch.int64, device=device)
    rows = torch.arange(len(owners) * 5, dtype=torch.int64, device=device).reshape(-1, 5)
    world = World(device)
    world.size, world.transport = 4, CaptureTransport()
    expected = world.exchange(rows, targets)
    actual = world.exchange(rows, targets, native_packet=True)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_old_operator_library_retains_reference_exchange(monkeypatch):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    rows = torch.arange(15, dtype=torch.int64, device=device).reshape(3, 5)
    targets = torch.tensor([2, 0, 2], dtype=torch.int64, device=device)
    world = World(device)
    world.size, world.transport = 3, CaptureTransport()
    expected = world.exchange(rows, targets)
    ops = cnot_native._cuda_ops()

    class OlderLibrary:
        def __getattr__(self, name):
            if name == "exchange_packet":
                raise AttributeError(name)
            return getattr(ops, name)

    monkeypatch.setattr(cnot_native, "_cuda_ops", lambda: OlderLibrary())
    actual = world.exchange(rows, targets, native_packet=True)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
