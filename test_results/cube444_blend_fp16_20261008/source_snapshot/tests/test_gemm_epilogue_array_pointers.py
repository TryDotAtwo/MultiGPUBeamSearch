from copy import deepcopy
import pytest
from tools.gemm_launch_values import validate_epilogue_override_endpoints

def fixture():return {name:dict(role='null',offset=0,available_bytes=0) for name in
                      ('alpha','beta','alpha_array','beta_array')}

def test_regular_requires_four_override_endpoints():
    validate_epilogue_override_endpoints(fixture(),broadcast=False)

@pytest.mark.parametrize('field',['alpha_array','beta_array'])
@pytest.mark.parametrize('mutation',['missing','nonnull','bool_offset','bool_extent','nonzero_extent'])
def test_regular_array_override_mutations_reject(field,mutation):
    value=deepcopy(fixture());endpoint=value[field]
    if mutation=='missing':del value[field]
    elif mutation=='nonnull':endpoint['role']='lane_context'
    elif mutation=='bool_offset':endpoint['offset']=False
    elif mutation=='bool_extent':endpoint['available_bytes']=False
    else:endpoint['available_bytes']=8
    with pytest.raises(ValueError):validate_epilogue_override_endpoints(value,broadcast=False)

def test_broadcast_has_no_array_pointer_members():
    with pytest.raises(ValueError):validate_epilogue_override_endpoints(fixture(),broadcast=True)
    value=fixture();del value['alpha_array'];del value['beta_array']
    validate_epilogue_override_endpoints(value,broadcast=True)
