import json
from types import SimpleNamespace
import pytest
from multigpubeamsearch.cube444_artifacts import validate_bundle, tensor_shapes
from multigpubeamsearch.errors import NativeBackendError


def manifest():
    return dict(backend='cube444_q_blend', dtype='fp32', state_len=96, num_classes=6,
                output_dim=24, transformer_weight=.6, mlp_weight=.4, files={})


@pytest.mark.parametrize('change', [{'state_len':120}, {'dtype':'fp16'},
                                  {'output_dim':1}, {'graph_hash':'other'}])
def test_wrong_cube_contract_rejected_before_blob_loading(tmp_path, change):
    data=dict(manifest(),**change)
    (tmp_path/'blend.json').write_text(json.dumps(data))
    graph=SimpleNamespace(state_len=96,move_count=24,num_classes=6,graph_hash='bound')
    with pytest.raises(NativeBackendError,match='contract mismatch'):
        validate_bundle(tmp_path,graph,'cube444_transformer')


def test_unsafe_bundle_binding_rejected(tmp_path):
    data=manifest();data['files']={'../weights.bin':'0'*64}
    (tmp_path/'blend.json').write_text(json.dumps(data))
    graph=SimpleNamespace(state_len=96,move_count=24,num_classes=6,graph_hash='bound')
    with pytest.raises(NativeBackendError,match='unsafe bundle binding'):
        validate_bundle(tmp_path,graph,'cube444_mlp')


def test_cube_schema_covers_both_readouts_and_all_backbone_layers():
    shapes=tensor_shapes()
    assert len(shapes)==122
    assert shapes['s3/blocks/3/ff2_w']==(1024,256)
    assert shapes['mlp/blocks/3/l2_w']==(2304,2304)
    assert shapes['s3/output_layer_w']==(256,24)
    assert shapes['mlp/out_w']==(2304,24)

