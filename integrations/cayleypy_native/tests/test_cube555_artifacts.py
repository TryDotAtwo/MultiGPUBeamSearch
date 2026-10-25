"""CPU model acceptance, independent of the native state150 boundary/GPU build."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.models import NativeEnsemble, NativeModel, prepare_model, verify_prepared_model
from multigpubeamsearch.options import NativeOptions
from tools.cube555.models import PieceTransformerQ555, ResMLPQ
from tools.cube555.export import export_native_models


@pytest.fixture
def bundle_inputs(tmp_path):
    torch.manual_seed(719)
    torch.set_num_threads(1)
    names = tuple(f'm{i}' for i in range(30))
    center = tuple(range(150))
    generators = tuple(tuple((j+i+1) % 150 for j in range(150)) for i in range(30))
    graph_data = dict(generators_type='PERMUTATION', generators=[list(g) for g in generators], central_state=list(center))
    graph_hash = hashlib.sha256(json.dumps(graph_data, sort_keys=True).encode()).hexdigest()
    contract = GraphContract(150, 150, generators, names, center, center, graph_hash)
    layout = dict(state_size=150, num_pieces=98, n_actions=30, head_facelet=list(range(98)),
        sticker_local=list(range(150)), sticker_orbit=[0]*150, sticker_vocab=150,
        piece_types=[0]*98, num_piece_types=1, orbit_sizes=[150], _move_names=list(names))
    tc = dict(model_class='PieceTransformerQ555', activation='silu', output_dim=30,
              d_model=8, nhead=2, ff_dim=16, num_layers=1)
    mc = dict(model_class='ResMLPQ', state_size=150, num_classes=150, output_dim=30,
              encoding='embedding', d_model=8, embed_dim=2, num_res_blocks=1)
    transformer, mlp = PieceTransformerQ555(tc, layout), ResMLPQ(mc)
    for model in (transformer, mlp):
        for parameter in model.parameters():
            torch.nn.init.uniform_(parameter, -.2, .2)
    for name, model, config in [('transformer', transformer, tc), ('mlp', mlp, mc)]:
        torch.save(dict(model_config=config, model=model.state_dict()), tmp_path/f'{name}.pt')
    (tmp_path/'layout.json').write_text(json.dumps(layout))
    info = dict(central_state=list(center), generators=dict(zip(names, map(list, generators))))
    (tmp_path/'puzzle_info.json').write_text(json.dumps(info))
    return tmp_path, contract, transformer.eval(), mlp.eval()


def export(inputs):
    path, contract, _, _ = inputs
    return export_native_models(path/'transformer.pt', path/'mlp.pt', path/'layout.json',
                                path/'puzzle_info.json', path/'export', contract=contract)


def test_two_explicit_members_private_snapshot_and_mutation(bundle_inputs):
    path, contract, _, _ = bundle_inputs
    members = export(bundle_inputs)
    assert tuple(m.backend for m in members) == ('cube555_transformer', 'cube555_mlp')
    ensemble = NativeEnsemble(members, (.8, .2))
    prepared = prepare_model(ensemble, contract, NativeOptions(), path/'run')
    assert [m['coefficient'] for m in prepared.manifest['ensemble']['models']] == [.8, .2]
    verify_prepared_model(prepared, contract)
    member = prepared.weights_dir/'member-0'/'weights'/'output_bias.fp32'
    member.write_bytes(bytes(member.stat().st_size))
    with pytest.raises(NativeBackendError, match='checksum'):
        verify_prepared_model(prepared, contract)


def test_script_features_and_fp32_readout(bundle_inputs):
    path, contract, transformer, mlp = bundle_inputs
    members = export(bundle_inputs)
    states = torch.stack([torch.randperm(150) for _ in range(5)]).to(torch.uint8)
    padded = torch.nn.functional.pad(states, (0, 10), value=255)
    outputs = []
    for member, original in zip(members, (transformer, mlp)):
        manifest = json.loads((member.weights_dir/'manifest.json').read_text())
        script = torch.jit.load(str(member.weights_dir/'features.ts'))
        with torch.inference_mode():
            feature = script(padded)
            torch.testing.assert_close(feature, original.half().features(states), rtol=0, atol=0)
            torch.testing.assert_close(script(states), feature, rtol=0, atol=0)
        assert feature.dtype == torch.float16
        width = manifest['feature_dim']
        weight = torch.frombuffer(bytearray((member.weights_dir/'output_weight.fp16').read_bytes()), dtype=torch.float16).reshape(width, 30)
        bias = torch.frombuffer(bytearray((member.weights_dir/'output_bias.fp32').read_bytes()), dtype=torch.float32)
        actual = feature.float() @ weight.float() + bias
        expected = torch.nn.functional.linear(feature.float(), original.q_head.weight.float(), original.q_head.bias.float())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        outputs.append(actual)
        with pytest.raises((RuntimeError, torch.jit.Error), match='uint8'):
            script(states.to(torch.int8))
    blended = .8*outputs[0] + .2*outputs[1]
    assert blended.shape == (5, 30) and blended.dtype == torch.float32
    assert torch.isfinite(blended).all()


@pytest.mark.parametrize('change', ['order', 'permutation', 'center', 'layout'])
def test_reject_graph_layout_mismatch_before_export(bundle_inputs, change):
    path, _, _, _ = bundle_inputs
    info = json.loads((path/'puzzle_info.json').read_text())
    layout = json.loads((path/'layout.json').read_text())
    if change == 'order':
        info['generators'] = dict(reversed(list(info['generators'].items())))
    elif change == 'permutation':
        info['generators']['m0'] = info['generators']['m1']
    elif change == 'center':
        info['central_state'][0], info['central_state'][1] = 1, 0
    else:
        layout['head_facelet'][0] = 150
    (path/'puzzle_info.json').write_text(json.dumps(info))
    (path/'layout.json').write_text(json.dumps(layout))
    with pytest.raises(ValueError):
        export(bundle_inputs)
    assert not (path/'export').exists()


def test_admission_declares_fp32_readout(bundle_inputs):
    path, contract, _, _ = bundle_inputs
    members = export(bundle_inputs)
    # The model-only admission is allowed, but silently choosing a coefficient is not.
    from multigpubeamsearch.cube555_artifacts import validate_bundle
    prepared = validate_bundle(members[0].weights_dir, contract, members[0].backend)
    assert prepared.manifest['score_semantics'] == 'distance_q'
    assert prepared.manifest['readout_semantics'] == 'fp16_features_fp32_accumulation'
    assert prepared.manifest['hidden1'] == prepared.manifest['hidden2'] == prepared.manifest['feature_dim']
    assert prepared.manifest['planning_descriptor_only'] is True


def test_manifest_duplicate_and_unsafe_file_rejected(bundle_inputs):
    path, contract, _, _ = bundle_inputs
    member = export(bundle_inputs)[0]
    target = member.weights_dir/'manifest.json'
    raw = target.read_text()
    target.write_text(raw.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    from multigpubeamsearch.cube555_artifacts import validate_bundle
    with pytest.raises(NativeBackendError, match='duplicate'):
        validate_bundle(member.weights_dir, contract, member.backend)
    manifest = json.loads(raw)
    manifest['files']['../escape'] = '0'*64
    target.write_text(json.dumps(manifest))
    with pytest.raises(NativeBackendError):
        validate_bundle(member.weights_dir, contract, member.backend)


@pytest.mark.parametrize('field,value', [('graph_hash', '0'*64), ('feature_dim', 16),
    ('hidden1', 16), ('normalization', 'layernorm'), ('readout_semantics', 'legacy_fp16_head')])
def test_reject_changed_execution_contract(bundle_inputs, field, value):
    _, contract, _, _ = bundle_inputs
    member = export(bundle_inputs)[0]
    target = member.weights_dir/'manifest.json'
    manifest = json.loads(target.read_text())
    manifest[field] = value
    target.write_text(json.dumps(manifest))
    from multigpubeamsearch.cube555_artifacts import validate_bundle
    with pytest.raises(NativeBackendError, match='contract'):
        validate_bundle(member.weights_dir, contract, member.backend)


@pytest.mark.skipif(os.environ.get('CUBE555_RUN_LOCAL_CUDA') != '1', reason='explicit opt-in local CUDA check')
def test_cuda_features_and_readout_synthetic(bundle_inputs):
    """One tiny CUDA model case; never a sweep or a target performance claim."""
    path, _, transformer, mlp = bundle_inputs
    assert torch.cuda.is_available(), 'opt-in CUDA check requires an available local GPU'
    members = export(bundle_inputs)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for member, original in zip(members, (transformer, mlp)):
            script = torch.jit.load(str(member.weights_dir/'features.ts'), map_location='cuda')
            original = original.half().cuda().eval()
            width = original.q_head.in_features
            weight = torch.frombuffer(bytearray((member.weights_dir/'output_weight.fp16').read_bytes()), dtype=torch.float16).reshape(width, 30).cuda()
            bias = torch.frombuffer(bytearray((member.weights_dir/'output_bias.fp32').read_bytes()), dtype=torch.float32).cuda()
            with torch.inference_mode():
                for batch in (1, 2):
                    states = torch.stack([torch.randperm(150) for _ in range(batch)]).to('cuda', torch.uint8)
                    features = script(torch.nn.functional.pad(states, (0, 10), value=255))
                    torch.testing.assert_close(features, original.features(states), rtol=0, atol=0)
                    actual = features.float() @ weight.float() + bias
                    reference = torch.nn.functional.linear(original.features(states).float(), original.q_head.weight.float(), original.q_head.bias.float())
                    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
            torch.cuda.synchronize()
            del script, original, weight, bias, features, actual, reference
        print(json.dumps(dict(device=torch.cuda.get_device_name(0), batches=[1, 2],
            scope='synthetic feature/readout CUDA parity only', peak_allocated=torch.cuda.max_memory_allocated())))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
