from types import SimpleNamespace
import torch
import pytest
from cayleypy_native.backend import prepare_runtime, runtime_devices
from cayleypy_native.options import NativeOptions
from cayleypy_native.errors import NativeUnavailable


@pytest.mark.parametrize("name,expected", [("Tesla T4","libtorch"), ("NVIDIA GeForce RTX 3060","cutlass")])
def test_automatic_executor_is_passed_to_build(monkeypatch, tmp_path, name, expected):
    import cayleypy_native.build as build
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (7,5) if "T4" in name else (8,6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: name)
    observed = []
    def ensure(contract, model, options, arch, run_dir):
        observed.append(options.inference_backend)
        return tmp_path / "runner", {"inference_backend":options.inference_backend}
    monkeypatch.setattr(build, "ensure_runner", ensure)
    model = SimpleNamespace(backend="mlp",manifest={"dtype":"fp16","output_dim":1})
    runtime = prepare_runtime(SimpleNamespace(move_count=3),model,NativeOptions(),tmp_path,(0,))
    assert observed == [expected]
    assert runtime.build_metadata["inference_backend"] == expected


def test_requested_gpu_count_never_silently_shrinks(monkeypatch):
    import cayleypy_native.backend as backend
    monkeypatch.setattr(backend.platform,"system",lambda:"Linux")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:2)
    for name in ("RANK","LOCAL_RANK","WORLD_SIZE","TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(name,raising=False)
    assert runtime_devices(object(),NativeOptions(num_gpus=2)) == (0,1)
    with pytest.raises(NativeUnavailable,match="exceeds"):
        runtime_devices(object(),NativeOptions(num_gpus=3))


@pytest.mark.parametrize('backend,capability,expected',[
    ('auto',(7,5),'libtorch'),('auto',(8,6),'cutlass'),('libtorch',(8,6),'libtorch'),
    ('cutlass',(8,6),'cutlass')])
def test_ensemble_backend_selection_is_explicit(monkeypatch,tmp_path,backend,capability,expected):
    import cayleypy_native.build as build
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda d:capability)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda d:'Test GPU')
    selected=[]
    def ensure(contract,model,options,architectures,run_dir):
        selected.append(options.inference_backend)
        return tmp_path/'runner',{'inference_backend':options.inference_backend}
    monkeypatch.setattr(build,'ensure_runner',ensure)
    model=SimpleNamespace(backend='ensemble',manifest={'dtype':'fp16','output_dim':1})
    prepare_runtime(SimpleNamespace(move_count=3),model,NativeOptions(inference_backend=backend),tmp_path,(0,))
    assert selected==[expected]


def test_explicit_ensemble_cutlass_rejects_t4(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda d:(7,5))
    model=SimpleNamespace(backend='ensemble',manifest={'dtype':'fp16','output_dim':1})
    with pytest.raises(NativeUnavailable,match='SM80'):
        prepare_runtime(SimpleNamespace(move_count=3),model,NativeOptions(inference_backend='cutlass'),tmp_path,(0,))
