from types import SimpleNamespace
import pytest
import torch
from multigpubeamsearch import NativeOptions
from multigpubeamsearch.backend import runtime_devices


@pytest.mark.parametrize('graph,expected', [
    (SimpleNamespace(device='cpu'), (0, 1)),
    (SimpleNamespace(device_config=SimpleNamespace(devices=('cpu',))), (0, 1)),
    (SimpleNamespace(), (0, 1)),
    (SimpleNamespace(device='cuda:1'), (1,)),
])
def test_native_device_autoselection(monkeypatch, graph, expected):
    import multigpubeamsearch.backend as backend
    monkeypatch.setattr(backend.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'TORCHELASTIC_RUN_ID'):
        monkeypatch.delenv(key, raising=False)
    assert runtime_devices(graph, NativeOptions()) == expected
    assert runtime_devices(graph, NativeOptions(num_gpus=1)) == (0,)
    assert runtime_devices(graph, NativeOptions(devices=(1,))) == (1,)
