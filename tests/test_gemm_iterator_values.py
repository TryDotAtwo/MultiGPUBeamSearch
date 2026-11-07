"""Independent literal iterator fixture; catches wrong increments/advances."""
import copy
import pytest

DESC = dict(shape=dict(row=8,group=4,cluster=1,tile=1),
    iterations=dict(row=1,group=1),delta=dict(row=8,group=32,cluster=128),
    count=dict(row=2,group=1))
VALUES = dict(stride=48,increment_row=384,increment_group=1536,
    increment_cluster=6144,advance_row=384,advance_group=2304,
    advance_cluster=3072,advance_tile=1536)

def test_literal_iterator_values():
    from tools.gemm_iterator_values import validate_epilogue_iterator
    assert validate_epilogue_iterator(VALUES, row_elements=24, descriptor=DESC)

@pytest.mark.parametrize('field',tuple(VALUES))
def test_each_public_member_corruption_rejected(field):
    from tools.gemm_iterator_values import validate_epilogue_iterator
    changed=copy.deepcopy(VALUES);changed[field]+=17
    with pytest.raises(ValueError):
        validate_epilogue_iterator(changed,row_elements=24,descriptor=DESC)

@pytest.mark.parametrize('row_elements',[True,0,-1,2**63])
def test_invalid_row_geometry_rejected(row_elements):
    from tools.gemm_iterator_values import validate_epilogue_iterator
    with pytest.raises(ValueError):
        validate_epilogue_iterator(VALUES,row_elements=row_elements,descriptor=DESC)

def test_boolean_member_and_unknown_field_rejected():
    from tools.gemm_iterator_values import validate_epilogue_iterator
    for changed in (dict(VALUES,stride=True),dict(VALUES,extra=0)):
        with pytest.raises(ValueError):
            validate_epilogue_iterator(changed,row_elements=24,descriptor=DESC)

def test_empty_and_zero_descriptor_rejected():
    from tools.gemm_iterator_values import validate_epilogue_iterator
    zero=copy.deepcopy(DESC);zero['shape']['row']=0
    for descriptor in ({},zero):
        with pytest.raises(ValueError):
            validate_epilogue_iterator(VALUES,row_elements=24,descriptor=descriptor)
