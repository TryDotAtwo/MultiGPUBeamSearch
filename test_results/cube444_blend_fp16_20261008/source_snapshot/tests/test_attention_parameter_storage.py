from copy import deepcopy
import pytest
from tools.attention_parameter_storage import validate_attention_parameter_storage


def table():
    entries=[dict(architecture=sm,queries_per_block=q,keys_per_block=64,max_k=k,
        aligned=a,parameter_size=224,parameter_alignment=8) for sm in (75,80)
        for q,k,a in ((64,64,True),(32,64,True),(64,32,True),
                     (32,32,True),(64,64,False),(64,32,False))]
    return dict(schema_version=1,scope='compile_time_attention_parameter_storage_not_admission',entries=entries)


def test_all_compiled_specializations_present():
    assert len(validate_attention_parameter_storage(table()))==12


@pytest.mark.parametrize('kind',['missing','duplicate','bool_size','bad_align','unknown_arch','too_large'])
def test_malformed_parameter_tables_reject(kind):
    t=deepcopy(table());r=t['entries'][0]
    if kind=='missing':t['entries'].pop()
    elif kind=='duplicate':t['entries'][1]=deepcopy(r)
    elif kind=='bool_size':r['parameter_size']=True
    elif kind=='bad_align':r['parameter_alignment']=3
    elif kind=='unknown_arch':r['architecture']=90
    else:r['parameter_size']=32769
    with pytest.raises(ValueError):validate_attention_parameter_storage(t)
