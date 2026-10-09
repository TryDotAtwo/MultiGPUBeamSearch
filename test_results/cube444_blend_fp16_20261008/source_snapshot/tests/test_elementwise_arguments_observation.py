from copy import deepcopy
import pytest
from tools.elementwise_arguments_observation import validate_empty_arguments_table,observe_headers,HEADERS

def table():
    return dict(schema_version=1,scope='compiled_canonical_empty_elementwise_arguments_not_admission',
                entries=[dict(activation=name,empty=True,pinned_empty_type=True,storage_bytes=1,alignment=1)
                         for name in ('Identity','ReLu')])

def test_independent_empty_arguments_table():
    assert validate_empty_arguments_table(table())==table()

@pytest.mark.parametrize('field,value',[('activation','Scale'),('empty',False),('empty',1),
    ('pinned_empty_type',False),('storage_bytes',2),('storage_bytes',1.0),('alignment',2),('alignment',True)])
def test_nonempty_unpinned_or_wrong_typed_arguments_reject(field,value):
    changed=deepcopy(table());changed['entries'][0][field]=value
    with pytest.raises(ValueError):validate_empty_arguments_table(changed)

@pytest.mark.parametrize('mutation',['missing','extra','duplicate','scope','version_type'])
def test_incomplete_exclusion_rejects(mutation):
    changed=table()
    if mutation=='missing':changed['entries'].pop()
    elif mutation=='extra':changed['entries'][0]['hidden']=0
    elif mutation=='duplicate':changed['entries'][1]=deepcopy(changed['entries'][0])
    elif mutation=='scope':changed['scope']='production_admitted'
    else:changed['schema_version']=True
    with pytest.raises(ValueError):validate_empty_arguments_table(changed)

@pytest.mark.parametrize('name',list(HEADERS))
def test_header_drift_is_fail_closed(tmp_path,name,monkeypatch):
    monkeypatch.setattr('tools.elementwise_arguments_observation.HEADERS',{name:HEADERS[name]})
    folder=tmp_path/'include/cutlass/epilogue/thread';folder.mkdir(parents=True)
    for entry in HEADERS:(folder/entry).write_text('drift')
    (tmp_path/'CMakeCache.txt').write_text('CUTLASS_DIR:PATH='+str(tmp_path)+'\n')
    with pytest.raises(ValueError):observe_headers(tmp_path)
