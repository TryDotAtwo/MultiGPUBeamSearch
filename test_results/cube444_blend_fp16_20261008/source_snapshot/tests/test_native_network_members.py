"""Opt-in actual scorer artifact, never a replacement for native mutation probes.

Missing native extraction must fail rather than silently skip once the evidence
directory is explicitly selected. Run only on the authorized remote test host.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path
import pytest
from tools.graph_network_contract import validate_graph_network_members


def test_actual_scorer_preserves_captured_consumed_network_members():
    selected = os.environ.get('BEAM_TEST_NETWORK_SCORE_OUTPUT')
    if not selected:
        pytest.skip('explicit actual native score output required')
    root = Path(selected)
    # Independent literal model: do not derive expected extents from sidecar.
    model = dict(dtype='fp16', d_model=256, ff_dim=1024, num_layers=4,
                 seq_len=57, output_dim=24, nhead=8, head_dim=32)
    observed = {}
    hashes = {}
    for name in ('execution.json', 'graph_kernel_inventory.json',
                 'graph_network_members.json'):
        path = root / name
        assert path.is_file(), f'real scorer did not publish {name}'
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        observed[name] = json.loads(raw)
    execution = observed['execution.json']
    inventory = observed['graph_kernel_inventory.json']
    members = observed['graph_network_members.json']
    assert execution['executor'] == 'native_cuda_graph'
    for artifact in (inventory, members):
        assert artifact['device'] == execution['device']
        assert artifact['execution_sha256'] == hashes['execution.json']
    assert members['inventory_sha256'] == hashes['graph_kernel_inventory.json']
    assert inventory['production_admitted'] is False
    profile = dict(execution,
                   expected_padded_seq_len=int(os.environ['BEAM_TEST_NETWORK_PADDED_SEQ']))
    result = validate_graph_network_members(members, inventory, model, profile,
                                            int(os.environ['BEAM_TEST_NETWORK_LN_SHARED']))
    assert result['consumed_network_members_checked'] is True
    assert result['production_admitted'] is False
    assert result['parameter_values_checked'] is False


def test_native_typed_network_extraction_mutations():
    if os.environ.get('BEAM_TEST_NATIVE_NETWORK_PROBE') != '1':
        pytest.skip('explicit authorized remote native probe required')
    root = Path(__file__).resolve().parents[1]
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        binary = Path(temporary) / 'network_probe'
        command = ['nvcc', '-std=c++17', '-arch=sm_86', '-I'+str(root/'src'),
                   '-I'+str(root/'cuda'), '-DBEAM_STATE_LOGICAL_BYTES=96',
                   '-DBEAM_STATE_PHYSICAL_BYTES=112', '-DBEAM_STATE_ALIGNMENT=16',
                   '-DBEAM_MOVE_COUNT=24', str(root/'tests/graph_network_native_probe.cu'),
                   '-o', str(binary)]
        build = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert build.returncode == 0, build.stdout + build.stderr
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr
        assert 'swapped roles/short extent preserved; foreign rejected' in run.stdout
