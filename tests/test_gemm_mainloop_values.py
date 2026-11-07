"""Literal independent mainloop arithmetic,not CUTLASS constructor mirroring."""
import copy
import pytest
from tools import gemm_iterator_values

FIXTURES=[
    (256,dict(element_bits=16,advance_rank=0,shape_contiguous=32,shape_strided=128,
        iterations_strided=4,delta_strided=4),
     dict(stride=256,increment_strided=2048,increment_next=-6080,advance=64)),
    (768,dict(element_bits=16,advance_rank=1,shape_contiguous=64,shape_strided=32,
        iterations_strided=4,delta_strided=4),
     dict(stride=768,increment_strided=6144,increment_next=30720,advance=49152))]
FIELDS=('stride','increment_strided','increment_next','advance')
def validator():
    fn=getattr(gemm_iterator_values,'validate_mainloop_iterator',None)
    assert callable(fn),'independent A/B mainloop validator missing'
    return fn

@pytest.mark.parametrize('row,descriptor,values',FIXTURES)
def test_literal_mainloop_values(row,descriptor,values):
    assert validator()(values,row_elements=row,descriptor=descriptor)

@pytest.mark.parametrize('row,descriptor,values',FIXTURES)
@pytest.mark.parametrize('field',FIELDS)
def test_each_consumed_mainloop_field_mutation_rejected(row,descriptor,values,field):
    changed=copy.deepcopy(values);changed[field]+=17
    with pytest.raises(ValueError):validator()(changed,row_elements=row,descriptor=descriptor)

@pytest.mark.parametrize('row',[True,0,-1,2**63])
def test_invalid_mainloop_row_rejected(row):
    _,descriptor,values=FIXTURES[0]
    with pytest.raises(ValueError):validator()(values,row_elements=row,descriptor=descriptor)

@pytest.mark.parametrize('field',FIELDS)
def test_bool_mainloop_field_rejected(field):
    row,descriptor,values=FIXTURES[0];changed=dict(values);changed[field]=True
    with pytest.raises(ValueError):validator()(changed,row_elements=row,descriptor=descriptor)

@pytest.mark.parametrize('mutation',['missing','extra','bool_descriptor','wrong_dtype','wrong_rank'])
def test_incomplete_or_unsupported_mainloop_contract_rejected(mutation):
    row,descriptor,values=FIXTURES[0];descriptor=dict(descriptor);values=dict(values)
    if mutation=='missing':values.pop('advance')
    if mutation=='extra':values['other']=0
    if mutation=='bool_descriptor':descriptor['delta_strided']=True
    if mutation=='wrong_dtype':descriptor['element_bits']=8
    if mutation=='wrong_rank':descriptor['advance_rank']=2
    with pytest.raises(ValueError):validator()(values,row_elements=row,descriptor=descriptor)
