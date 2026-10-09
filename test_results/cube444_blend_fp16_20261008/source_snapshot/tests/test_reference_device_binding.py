import copy
import pytest
from tools.loaded_view_contract import validate_reference_device_binding

def fixtures():
    row=dict(device=0,sm=86,uuid_hex='1'*32)
    return (dict(schema_version=1,scope='observed_cuda_visible_devices_not_quality',
        devices=[row],production_quality_accepted=False),
        dict(row,schema_version=1,production_quality_accepted=False),
        dict(schema_version=1,scope='execution_shape_not_kernel_admission',
             outer_microbatch=8,transformer_microbatch=8,lanes=1,device=0,sm=86,production_quality_accepted=False))

def test_independent_map_match():
    assert validate_reference_device_binding(*fixtures())['reference_device_matches']

@pytest.mark.parametrize('kind', ['uuid','sm','ordinal','duplicate','zero','bool','missing','quality','shape'])
def test_wrong_binding_rejected(kind):
    p,i,s=copy.deepcopy(fixtures())
    if kind=='uuid': i['uuid_hex']='2'*32
    if kind=='sm': i['sm']=90
    if kind=='ordinal': i['device']=1
    if kind=='duplicate': p['devices'].append(dict(p['devices'][0],device=1))
    if kind=='zero': p['devices'][0]['uuid_hex']='0'*32
    if kind=='bool': p['devices'][0]['device']=False
    if kind=='missing': del i['uuid_hex']
    if kind=='quality': p['production_quality_accepted']=0
    if kind=='shape': s['sm']=90
    with pytest.raises(ValueError): validate_reference_device_binding(p,i,s)
