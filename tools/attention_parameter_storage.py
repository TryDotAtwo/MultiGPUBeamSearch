"""Compiled FMHA Params sizes, independently supplied; no values or admission."""


def validate_attention_parameter_storage(value):
    if (type(value) is not dict or set(value)!={'schema_version','scope','entries'} or
        type(value['schema_version']) is not int or value['schema_version']!=1 or
        value['scope']!='compile_time_attention_parameter_storage_not_admission' or
        type(value['entries']) is not list or len(value['entries'])!=12):
        raise ValueError('invalid attention parameter storage table')
    expected={(sm,q,64,k,a) for sm in (75,80) for q,k,a in
        ((64,64,True),(32,64,True),(64,32,True),(32,32,True),(64,64,False),(64,32,False))}
    result={}
    fields={'architecture','queries_per_block','keys_per_block','max_k','aligned',
            'parameter_size','parameter_alignment'}
    for row in value['entries']:
        if type(row) is not dict or set(row)!=fields or type(row['aligned']) is not bool:
            raise ValueError('malformed attention parameter record')
        if any(type(row[k]) is not int for k in fields-{'aligned'}):
            raise ValueError('invalid attention parameter integer')
        key=tuple(row[k] for k in ('architecture','queries_per_block','keys_per_block','max_k','aligned'))
        size,alignment=row['parameter_size'],row['parameter_alignment']
        if key not in expected or key in result or alignment not in (4,8,16) or not 0<size<=32768 or size%alignment:
            raise ValueError('unsupported or duplicate attention parameter storage')
        result[key]=[dict(index=0,offset=0,size=size)]
    if set(result)!=expected:raise ValueError('incomplete attention parameter table')
    return result
