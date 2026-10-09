import json
import os
from pathlib import Path
import pytest
from tools.cupti_replay_trace import validate_replay_trace, observe_injection_library

CAPTURE_WORLD = int(os.environ.get('CUPTI_CAPTURE_WORLD', '2'))
if CAPTURE_WORLD not in (2, 8):
    raise ValueError('CUPTI_CAPTURE_WORLD must be 2 or 8')

@pytest.fixture(params=range(CAPTURE_WORLD))
def capture(request):
    validation = os.environ.get('CUPTI_VALIDATION_ROOT')
    if validation:
        root = Path(validation) / f'device{request.param}'
        return (root/'runner.log').read_text(),json.loads(
            (root/'output/graph_kernel_inventory.json').read_text()),json.loads(
            (root/'output/graph_transfers.json').read_text()),request.param
    location=os.environ.get('CUPTI_CAPTURE_ROOT')
    logs=os.environ.get('CUPTI_CAPTURE_LOGS')
    if not location or not logs:
        pytest.skip('explicit actual remote CUPTI captures required')
    rank=request.param
    root=Path(location)/f'pilot_gpu{rank}'
    return (Path(logs)/f'cupti_replay_pilot_gpu{rank}.log').read_text(),json.loads(
        (root/'graph_kernel_inventory.json').read_text()),json.loads(
        (root/'graph_transfers.json').read_text()),rank

def test_actual_replay(capture):
    text,inventory,transfers,rank=capture
    result=validate_replay_trace(text,inventory,transfers,device_ordinal=rank)
    assert result['graph_launches']==8 and result['production_admitted'] is False

@pytest.mark.parametrize('mutation',['drop','error','launch','mutable','device','missing_node','static_shared','stream'])
def test_actual_corruption_rejected(capture,mutation):
    text,inventory,transfers,rank=capture
    if mutation=='drop':text+='\nDropped 1 records'
    elif mutation=='error':text+='\nCUPTI_ERROR_UNKNOWN'
    elif mutation=='launch':text=text.replace('cudaGraphLaunch_v10000','unobservedLaunch')
    elif mutation=='mutable':text+='\nRUNTIME "cudaGraphExecKernelNodeSetParams_v10000"'
    elif mutation=='device':rank=(rank+1)%CAPTURE_WORLD
    elif mutation=='static_shared':text=text.replace('sharedMemory (static 0,','sharedMemory (static 1,')
    elif mutation=='stream':
        import re
        text=re.sub(r'(streamId )\d+(, graphId [1-9]\d*, graphNodeId)',r'\g<1>999999\2',text,count=1)
    else:
        lines=text.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('CONCURRENT_KERNEL') and i+2<len(lines) and 'graphId 0,' not in lines[i+2]:
                del lines[i:i+3]
                break
        else:raise AssertionError('no executed graph kernel found')
        text='\n'.join(lines)
    with pytest.raises(ValueError):
        validate_replay_trace(text,inventory,transfers,device_ordinal=rank)

@pytest.mark.parametrize('ordinal', [-1, 8, True, False, '2', None])
def test_invalid_explicit_device_is_rejected(ordinal):
    with pytest.raises(ValueError, match='explicit device'):
        validate_replay_trace('', {}, {}, device_ordinal=ordinal)

def test_library_fingerprint_changes(tmp_path):
    path=tmp_path/'injection.so'
    path.write_bytes(b'first')
    first=observe_injection_library(path)
    path.write_bytes(b'second')
    assert observe_injection_library(path)!=first
    with pytest.raises(ValueError):observe_injection_library(tmp_path)

@pytest.mark.parametrize('problem',['no_inventory','no_transfers','changed_hash','ambient_injection'])
def test_invalid_cupti_request_never_starts_scorer(problem,tmp_path,monkeypatch):
    from tools.cube4_production_preflight import collect_fresh_candidate
    library=tmp_path/'injection.so';library.write_bytes(b'controlled-library')
    binding=observe_injection_library(library)
    observed=dict(expected={},reference_dir=str(tmp_path),reference_scores=[[0]],
        require_graph_inventory=True,require_graph_transfers=True,expected_cupti_replay=binding)
    if problem=='no_inventory':observed['require_graph_inventory']=False
    elif problem=='no_transfers':observed['require_graph_transfers']=False
    elif problem=='changed_hash':binding['sha256']='0'*64
    else:monkeypatch.setenv('CUDA_INJECTION64_PATH',str(library))
    started=[]
    monkeypatch.setattr('tools.cube4_production_preflight.run_scorer',lambda *a,**k:started.append(True))
    with pytest.raises(ValueError):
        collect_fresh_candidate([],tmp_path/'run',5,lambda:observed,[[0]])
    assert started==[]
    assert not (tmp_path/'run').exists()
