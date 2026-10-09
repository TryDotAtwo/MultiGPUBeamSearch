"""Strict compile-time probe table; provenance and graph admission are separate."""


def validate_attention_storage_table(value):
    if (type(value) is not dict or set(value)!={'schema_version','scope','entries'}
            or type(value['schema_version']) is not int or value['schema_version']!=1
            or value['scope']!='compile_time_attention_storage_not_admission'
            or type(value['entries']) is not list or len(value['entries'])!=12):
        raise ValueError('invalid compile-time attention storage table')
    required={'architecture','queries_per_block','keys_per_block','max_k',
              'aligned','block','dynamic_shared_bytes'}
    expected={(sm,q,64,k,a) for sm in (75,80)
              for q,k,a in ((64,64,True),(32,64,True),(64,32,True),
                            (32,32,True),(64,64,False),(64,32,False))}
    result={}
    for row in value['entries']:
        if (type(row) is not dict or set(row)!=required or
                any(type(row[k]) is not int for k in
                    ('architecture','queries_per_block','keys_per_block','max_k','dynamic_shared_bytes'))
                or type(row['aligned']) is not bool):
            raise ValueError('invalid attention storage record')
        key=tuple(row[k] for k in ('architecture','queries_per_block','keys_per_block','max_k','aligned'))
        if key not in expected or key in result:
            raise ValueError('unsupported or duplicate attention storage specialization')
        block=row['block']
        if (type(block) is not list or len(block)!=3 or
                any(type(x) is not int for x in block) or
                block!=[32,4 if row['queries_per_block']==64 else 2,1] or
                not 0<row['dynamic_shared_bytes']<=98304):
            raise ValueError('invalid attention launch storage geometry')
        result[key]={'block':tuple(block),'dynamic_shared_bytes':row['dynamic_shared_bytes']}
    if set(result)!=expected:
        raise ValueError('incomplete attention storage table')
    return result
