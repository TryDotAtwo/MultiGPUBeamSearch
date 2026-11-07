import json
from types import SimpleNamespace

import pytest

from multigpubeamsearch import models
from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.options import NativeOptions


def test_cube_heads_share_one_validated_private_snapshot(tmp_path,monkeypatch):
    calls=[]
    def validate(path,contract,family):
        calls.append((path.resolve(),family))
        return models.PreparedModel(path.resolve(),{'dtype':'fp16'},family,'a'*64,('weights.bin',))
    monkeypatch.setattr(models,'_validate_artifact',validate)
    def snapshot(prepared,contract,directory):
        directory.mkdir(parents=True)
        p=directory/'weights';p.mkdir();(p/'weights.bin').write_bytes(b'validated-test-only')
        return models.PreparedModel(p,prepared.manifest,prepared.backend,prepared.artifact_hash,prepared.artifact_files)
    monkeypatch.setattr(models,'_snapshot_artifact',snapshot)
    graph=SimpleNamespace(graph_hash='graph',state_len=96,move_count=24)
    sources=tuple(models.NativeModel(tmp_path/'original','graph',backend=family)
        for family in ('cube444_transformer','cube444_mlp'))
    model=models.prepare_model(models.NativeEnsemble(sources,(.6,.4)),graph,NativeOptions(),tmp_path/'run')
    # Original bundle validation once; each private ensemble validation once.
    assert len(calls)==2
    assert not (tmp_path/'run/weights/member-1').exists()
    entries=model.manifest['ensemble']['models']
    assert entries[0]['weights_dir']==entries[1]['weights_dir']
    assert [x['family'] for x in entries]==['cube444_transformer','cube444_mlp']
    models.verify_prepared_model(model,graph)
    assert len(calls)==3
    data=json.loads((model.weights_dir/'ensemble.json').read_text())
    data['models'][1]['artifact_hash']='b'*64
    (model.weights_dir/'ensemble.json').write_text(json.dumps(data))
    with pytest.raises(NativeBackendError,match='member changed'):
        models.verify_prepared_model(model,graph)


def test_shared_cube_member_still_checks_its_expected_artifact_hash(tmp_path,monkeypatch):
    graph=SimpleNamespace(graph_hash='graph',state_len=96,move_count=24)
    def validate(path,contract,family):
        return models.PreparedModel(path,{'dtype':'fp16'},family,'a'*64,())
    monkeypatch.setattr(models,'_validate_artifact',validate)
    def snapshot(model,contract,path):
        path.mkdir(parents=True)
        return models.PreparedModel(path,model.manifest,model.backend,model.artifact_hash,())
    monkeypatch.setattr(models,'_snapshot_artifact',snapshot)
    sources=(models.NativeModel(tmp_path,'graph',backend='cube444_transformer'),
        models.NativeModel(tmp_path,'graph',backend='cube444_mlp',expected_artifact_hash='b'*64))
    with pytest.raises(NativeBackendError,match='hash changed'):
        models.prepare_model(models.NativeEnsemble(sources,(.6,.4)),graph,NativeOptions(),tmp_path/'run')
