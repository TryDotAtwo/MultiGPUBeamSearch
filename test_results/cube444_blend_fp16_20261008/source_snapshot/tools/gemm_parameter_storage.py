"""Compiled classic GEMM Params sizes; no fields/values or admission."""
import json


def descriptor_key(description):
    return json.dumps(description,sort_keys=True,separators=(',',':'),allow_nan=False)


def validate_gemm_parameter_storage(value):
    if (type(value) is not dict or set(value)!={'schema_version','scope','entries'} or
            type(value['schema_version']) is not int or value['schema_version']!=1 or
            value['scope'] not in ('compiled_SM75_baseline_gemm_parameters_not_admission',
                                  'compiled_SM80_classic_gemm_parameters_not_admission') or
            type(value['entries']) is not list):
        raise ValueError('invalid GEMM parameter table')
    sm80=value['scope']=='compiled_SM80_classic_gemm_parameters_not_admission'
    instruction=(16,8,16) if sm80 else (16,8,8)
    expected={('gemm_residual' if sm80 else 'gemm_pipelined',(128,64,32),(64,32,32),instruction,3 if sm80 else 2,1),
              ('gemm_bias',(128,64,32),(64,32,32),instruction,3 if sm80 else 2,1),
              ('gemm_relu',(128,64,32),(64,32,32),instruction,3 if sm80 else 2,1)}
    if sm80:
        expected.add(('gemm_bias',(128,128,32),(64,64,32),instruction,3,1))
        expected.add(('gemm_pipelined',(128,64,32),(64,32,32),(16,8,8),2,1))
    if len(value['entries'])!=len(expected):raise ValueError('incomplete GEMM parameter table')
    result={};seen=set()
    fields={'family','tile','warp','instruction','stages','swizzle','parameter_size','parameter_alignment'}
    for row in value['entries']:
        if type(row) is not dict or set(row)!=fields:raise ValueError('malformed GEMM parameter record')
        if type(row['family']) is not str:raise ValueError('invalid GEMM family')
        for name in ('tile','warp','instruction'):
            if type(row[name]) is not list or len(row[name])!=3 or any(type(v) is not int for v in row[name]):
                raise ValueError('unsupported GEMM parameter specialization')
        if (type(row['stages']) is not int or
                type(row['swizzle']) is not int or row['swizzle']!=1 or
                type(row['parameter_alignment']) is not int or row['parameter_alignment']!=8 or
                type(row['parameter_size']) is not int or not 0<row['parameter_size']<=32768 or row['parameter_size']%8):
            raise ValueError('invalid GEMM parameter storage')
        key=(row['family'],tuple(row['tile']),tuple(row['warp']),tuple(row['instruction']),row['stages'],row['swizzle'])
        if key not in expected or key in seen:raise ValueError('unknown or duplicate GEMM descriptor')
        seen.add(key)
        desc={k:row[k] for k in ('family','tile','warp','instruction','stages','swizzle')}
        result[descriptor_key(desc)]=[dict(index=0,offset=0,size=row['parameter_size'])]
    if seen!=expected:raise ValueError('incomplete GEMM descriptor coverage')
    return result
