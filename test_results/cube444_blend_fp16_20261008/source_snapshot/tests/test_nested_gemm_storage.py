from copy import deepcopy
import pytest
from tools.public_parameter_storage import nested_reference,validate_nested_gemm_storage

@pytest.mark.parametrize('mutation',['pointer_offset','stride_offset','stride_bytes','roundtrip','wrapper_bytes','epilogue_missing','epilogue_offset','missing_family'])
def test_nested_layout_corruption_rejects(mutation):
    value=deepcopy(nested_reference());row=value['entries'][0]
    if mutation=='missing_family':value['entries'].pop()
    elif mutation=='wrapper_bytes':row['mainloop_A'][0]['bytes']+=8
    elif mutation=='epilogue_missing':row['epilogue_C']['fields'].pop()
    elif mutation=='epilogue_offset':row['epilogue_C']['fields'][0]['offset']=False
    elif mutation=='roundtrip':row['ref_A']['accessor_roundtrip_checked']=False
    else:row['ref_A'][mutation]+=1
    with pytest.raises(ValueError):validate_nested_gemm_storage(value)

def test_six_nested_family_table_preserves_nonadmission():
    value=nested_reference();validate_nested_gemm_storage(value)
    assert len(value['entries'])==6 and value['production_admitted'] is False
