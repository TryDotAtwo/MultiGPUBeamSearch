import pytest
from tools.loaded_view_contract import validate_reference_execution_shape

def fixture():
    return dict(schema_version=1, scope='execution_shape_not_kernel_admission',
        outer_microbatch=8, transformer_microbatch=8, lanes=1, device=0, sm=86,
        production_quality_accepted=False)

def test_shape_is_only_reference_admission():
    assert validate_reference_execution_shape(fixture(), device=0, sm=86) == {
        'reference_shape_matches': True, 'production_quality_accepted': False}

@pytest.mark.parametrize('key,value', [('device',1),('sm',90),('lanes',4),
    ('outer_microbatch',1024),('transformer_microbatch',32),('schema_version',True),
    ('lanes',True),('production_quality_accepted',0),('scope','production')])
def test_wrong_field_rejected(key,value):
    shape=fixture(); shape[key]=value
    with pytest.raises(ValueError): validate_reference_execution_shape(shape, device=0, sm=86)

def test_unknown_or_missing_fields_rejected():
    for shape in ({},dict(fixture(),extra=1)):
        with pytest.raises(ValueError): validate_reference_execution_shape(shape, device=0, sm=86)

@pytest.mark.parametrize('device,sm', [(True,86),(0,True),(-1,86),(0,74)])
def test_expected_identity_must_be_valid(device,sm):
    with pytest.raises(ValueError): validate_reference_execution_shape(fixture(), device=device, sm=sm)
