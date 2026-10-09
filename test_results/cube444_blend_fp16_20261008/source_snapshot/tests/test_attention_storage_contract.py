import copy
import pytest
from tools.attention_storage_contract import validate_attention_storage_table


def table():
    entries=[]
    for sm in (75,80):
        for q,k,aligned in ((64,64,True),(32,64,True),(64,32,True),
                            (32,32,True),(64,64,False),(64,32,False)):
            entries.append(dict(architecture=sm,queries_per_block=q,
                keys_per_block=64,max_k=k,aligned=aligned,
                block=[32,4 if q==64 else 2,1],
                dynamic_shared_bytes=18944 if q==64 else 13568))
    return dict(schema_version=1,scope='compile_time_attention_storage_not_admission',entries=entries)


def test_complete_table_has_twelve_distinct_expectations():
    assert len(validate_attention_storage_table(table()))==12


@pytest.mark.parametrize('mutation', ['duplicate','missing','extra','boolean_sm',
    'integer_alignment','zero_shared','wrong_block','unknown_field','wrong_scope'])
def test_malformed_table_rejected(mutation):
    value=copy.deepcopy(table())
    if mutation=='duplicate': value['entries'][1]=value['entries'][0]
    elif mutation=='missing': value['entries'].pop()
    elif mutation=='extra': value['entries'].append(value['entries'][0])
    elif mutation=='boolean_sm': value['entries'][0]['architecture']=True
    elif mutation=='integer_alignment': value['entries'][0]['aligned']=1
    elif mutation=='zero_shared': value['entries'][0]['dynamic_shared_bytes']=0
    elif mutation=='wrong_block': value['entries'][0]['block']=[64,2,1]
    elif mutation=='unknown_field': value['entries'][0]['admitted']=True
    else: value['scope']='production_admitted'
    with pytest.raises(ValueError): validate_attention_storage_table(value)
