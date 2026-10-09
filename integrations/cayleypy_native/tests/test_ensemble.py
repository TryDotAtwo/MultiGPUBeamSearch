import json
from pathlib import Path
import pytest
from cayleypy import CayleyGraph, PermutationGroups
from multigpubeamsearch import NativeEnsemble, NativeModel, NativeOptions
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.models import prepare_model, verify_prepared_model
from multigpubeamsearch.errors import NativeBackendError


def contract():
    graph=CayleyGraph(PermutationGroups.lrx(4),device='cpu')
    return GraphContract.from_graph(graph,[1,0,2,3])


@pytest.mark.parametrize('coefficients', [(), (float('nan'),), (True,), (1e100,)])
def test_bad_coefficients(coefficients):
    with pytest.raises(ValueError): NativeEnsemble((None,),coefficients)


def test_arbitrary_members_and_private_snapshot(tmp_path):
    graph=contract()
    prepared=prepare_model(NativeEnsemble((None,)*5,(.2,)*5),graph,NativeOptions(),tmp_path/'run')
    assert prepared.backend=='ensemble'
    assert len(prepared.manifest['ensemble']['models'])==5
    verify_prepared_model(prepared,graph)
    pinned=NativeModel(prepared.weights_dir,graph.graph_hash,backend='ensemble',expected_artifact_hash=prepared.artifact_hash)
    copied=prepare_model(pinned,graph,NativeOptions(),tmp_path/'copied')
    assert copied.artifact_hash==prepared.artifact_hash
    member=copied.weights_dir/'member-0'/'weights'/'output_bias.fp16'
    member.write_bytes(b'\x00\x00')
    with pytest.raises(NativeBackendError):verify_prepared_model(copied,graph)
    verify_prepared_model(prepared,graph)


def test_graph_binding_and_coefficient_order(tmp_path):
    graph=contract()
    prepared=prepare_model(NativeEnsemble((None,None),(2.,-1.)),graph,NativeOptions(),tmp_path/'run')
    assert [x['coefficient'] for x in prepared.manifest['ensemble']['models']]==[2.,-1.]
    path=prepared.weights_dir/'ensemble.json'
    manifest=json.loads(path.read_text());manifest['graph_hash']='bad';path.write_text(json.dumps(manifest))
    with pytest.raises(NativeBackendError,match='contract'):verify_prepared_model(prepared,graph)
