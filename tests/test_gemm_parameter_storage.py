from copy import deepcopy
import pytest
from tools.gemm_parameter_storage import validate_gemm_parameter_storage


def table():
    return dict(schema_version=1,scope='compiled_SM75_baseline_gemm_parameters_not_admission',
        entries=[dict(family=f,tile=[128,64,32],warp=[64,32,32],instruction=[16,8,8],
            stages=2,swizzle=1,parameter_size=368 if f=='gemm_pipelined' else 440,
            parameter_alignment=8) for f in ('gemm_pipelined','gemm_bias','gemm_relu')])


def test_all_supported_baseline_families_present():
    assert len(validate_gemm_parameter_storage(table()))==3


@pytest.mark.parametrize('kind',['missing','duplicate','bool_size','bad_align','unknown_tile','unknown_family'])
def test_malformed_tables_rejected(kind):
    t=deepcopy(table());r=t['entries'][0]
    if kind=='missing':t['entries'].pop()
    elif kind=='duplicate':t['entries'][1]=deepcopy(r)
    elif kind=='bool_size':r['parameter_size']=True
    elif kind=='bad_align':r['parameter_alignment']=3
    elif kind=='unknown_tile':r['tile'][0]=64
    else:r['family']='gemm_unknown'
    with pytest.raises(ValueError):validate_gemm_parameter_storage(t)
