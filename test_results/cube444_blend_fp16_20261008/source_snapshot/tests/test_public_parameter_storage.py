"""Adversarial corruption of independently compiled public-member coverage."""
from copy import deepcopy
import pytest
from tools.public_parameter_storage import reference,validate_table

@pytest.mark.parametrize('kind',['gemm','attention'])
@pytest.mark.parametrize('mutation',['missing_family','missing_field','duplicate_field','bool_offset','shift_offset','short_member','gap_claim','admission'])
def test_corrupt_compiled_field_tables_reject(kind,mutation):
    value=deepcopy(reference()[kind]);row=value['entries'][0]
    if mutation=='missing_family':value['entries'].pop()
    elif mutation=='missing_field':row['fields'].pop()
    elif mutation=='duplicate_field':row['fields'].append(deepcopy(row['fields'][0]))
    elif mutation=='bool_offset':row['fields'][0]['offset']=False
    elif mutation=='shift_offset':row['fields'][0]['offset']+=1
    elif mutation=='short_member':row['fields'][0]['bytes']-=1
    elif mutation=='gap_claim':row['unclassified_storage_gaps']=[]
    else:value['production_admitted']=True
    with pytest.raises(ValueError):validate_table(value,kind)

@pytest.mark.parametrize('kind',['gemm','attention'])
def test_source_bound_public_table_preserves_nonadmission(kind):
    value=reference()[kind];validate_table(value,kind)
    assert value['production_admitted'] is False

@pytest.mark.parametrize('kind',['gemm','attention','nested'])
def test_missing_public_table_stops_transaction_before_process(kind,tmp_path,monkeypatch):
    from tools.cube4_production_preflight import collect_fresh_candidate
    import tools.cube4_production_preflight as preflight
    def forbidden(*args,**kwargs):raise AssertionError('scorer must not start')
    monkeypatch.setattr(preflight,'run_scorer',forbidden)
    tables={name+'_public_fields':value for name,value in reference().items()}
    from tools.public_parameter_storage import nested_reference
    tables['gemm_nested_storage']=nested_reference()
    tables.pop('gemm_nested_storage' if kind=='nested' else kind+'_public_fields')
    observation=dict(expected={},reference_dir='fixture',reference_scores={},compiled_probe_bundle=dict(tables=tables))
    with pytest.raises(ValueError):
        collect_fresh_candidate(['forbidden'],tmp_path/'run',1,lambda:observation,{})
    assert not (tmp_path/'run').exists()
