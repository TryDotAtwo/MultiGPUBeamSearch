from copy import deepcopy
import pytest
from tools.transfer_geometry_observation import validate_geometry_reference,validate_observed_pitched_geometry


def fixture():
    expected=dict(outer_microbatch=32,transformer_microbatch=13,
        device=dict(device=0,sm=86,uuid_hex='a'*32))
    cases=[]
    for size in (288,624):
        row=dict(bytes=size,source_pitched=dict(pitch=0,xsize=0,ysize=0),
                 destination_pitched=dict(pitch=0,xsize=0,ysize=0),
                 copy_geometry=dict(source_array_null=True,destination_array_null=True,
                    source_position=dict(x=0,y=0,z=0),destination_position=dict(x=0,y=0,z=0),
                    extent=dict(width=size,height=1,depth=1)))
        cases.append(dict(bytes=size,add_node=deepcopy(row),capture_async=deepcopy(row)))
    table=dict(schema_version=1,scope='independent_cuda_1d_transfer_geometry_not_admission',
        production_admitted=False,runtime_version=12080,device=deepcopy(expected['device']),cases=cases,
        memset_reference={route:dict(pitch=0,width=4096,height=1,element_size=1,value=0)
            for route in ('add_node','capture_async')})
    reference=dict(table=table,binary_sha256='b'*64,source_sha256={})
    value=dict(lanes=[dict(operations=[dict(kind='memcpy',**deepcopy(cases[0]['add_node'])),
        dict(kind='memset',bytes=4,element_size=1,memset_geometry=dict(pitch=0,width=4,height=1))])])
    return expected,reference,value


def test_reference_two_public_api_routes_bind_expected_profile():
    expected,reference,value=fixture()
    validate_geometry_reference(reference['table'],expected)
    validate_observed_pitched_geometry(value,reference,expected)

@pytest.mark.parametrize('section,field',[
    ('source_position','x'),('source_position','y'),('source_position','z'),
    ('destination_position','x'),('destination_position','y'),('destination_position','z'),
    ('extent','width'),('extent','height'),('extent','depth')])
def test_complete_copy_geometry_rejects_mutation(section,field):
    expected,reference,value=fixture()
    value['lanes'][0]['operations'][0]['copy_geometry'][section][field]+=1
    with pytest.raises(ValueError):validate_observed_pitched_geometry(value,reference,expected)

@pytest.mark.parametrize('field',['source_array_null','destination_array_null'])
@pytest.mark.parametrize('value',[False,1,None])
def test_array_endpoint_exclusions_require_typed_null_proof(field,value):
    expected,reference,capture=fixture()
    capture['lanes'][0]['operations'][0]['copy_geometry'][field]=value
    with pytest.raises(ValueError):validate_observed_pitched_geometry(capture,reference,expected)

@pytest.mark.parametrize('field',['pitch','width','height'])
def test_memset_geometry_cannot_be_omitted_or_changed(field):
    expected,reference,capture=fixture()
    del capture['lanes'][0]['operations'][1]['memset_geometry'][field]
    with pytest.raises(ValueError):validate_observed_pitched_geometry(capture,reference,expected)


@pytest.mark.parametrize('endpoint',['source_pitched','destination_pitched'])
@pytest.mark.parametrize('field',['pitch','xsize','ysize'])
@pytest.mark.parametrize('value',[1,0.0,False])
def test_all_six_captured_fields_reject_value_or_type_mutation(endpoint,field,value):
    expected,reference,captured=fixture()
    captured['lanes'][0]['operations'][0][endpoint][field]=value
    with pytest.raises(ValueError):validate_observed_pitched_geometry(captured,reference,expected)


@pytest.mark.parametrize('mutation',['runtime','uuid','size','api_disagreement','extra','missing'])
def test_reference_drift_cannot_supply_expected_geometry(mutation):
    expected,reference,captured=fixture();table=reference['table']
    if mutation=='runtime':table['runtime_version']=12090
    elif mutation=='uuid':table['device']['uuid_hex']='c'*32
    elif mutation=='size':table['cases'][0]['bytes']=48
    elif mutation=='api_disagreement':table['cases'][0]['capture_async']['source_pitched']['xsize']=1
    elif mutation=='extra':table['extra']=0
    else:del table['production_admitted']
    with pytest.raises(ValueError):validate_observed_pitched_geometry(captured,reference,expected)
