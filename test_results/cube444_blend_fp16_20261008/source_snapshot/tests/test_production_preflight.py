from copy import deepcopy
import pytest
from tools.cube4_production_preflight import inspect_candidate
import os
import sys
import time
from tools.cube4_production_preflight import run_scorer
from tools.cube4_production_preflight import collect_fresh_candidate
import json
import hashlib
from tools.resolved_execution_contract import canonical_resolved_recipe, SELECTOR_NAMES, PREFIX


def fixture():
    reference = [[float(i) for i in range(24)]]
    scores = {'schema_version': 1, 'jobs': []}
    for case, base, count in [('full', 0, 2), ('partial', 1, 1),
                              ('singleton', 2, 1), ('zero', 0, 0)]:
        scores['jobs'].append(dict(lane=0, case=case, parent_base=base,
            active_rows=count, reference_rows=[0] * count,
            raw_scores=[reference[0][:] for _ in range(count)]))
    expected = dict(reference_sha256='a' * 64, manifest_sha256='b' * 64,
        outer_microbatch=2, transformer_microbatch=1, lanes=1,
        executor='native_eager', device=dict(schema_version=1, device=0, sm=86,
            uuid_hex='c' * 32, production_quality_accepted=False),
        loaded_tensor_bindings={'a.fp16': {'size_bytes': 2, 'sha256': 'd' * 64}},
        requested_launch_policies={PREFIX+name:None for name in SELECTOR_NAMES})
    execution = deepcopy(expected)
    execution.update(schema_version=1, scope='actual_runner_raw_score_candidate_not_admission',
        production_quality_accepted=False, resolved_execution_contract_complete=False,
        loaded_tensor_content_attestation_complete=True, device_upload_bytes_verified=True,
        communicator_initialized=False, search_buffers_allocated=False,
        lanes_exercised_serially=True, distinct_reference_rows=1, loaded_view={},
        host_kernel_observations=[], kernel_coverage_complete=False,
        cohort_generalization_proven=False, scores_before_quantization=True)
    return reference, scores, execution, expected


def test_current_numerically_valid_candidate_cannot_admit_beam_ranks():
    result = inspect_candidate(*fixture())
    assert result['status'] == 'numeric_pass_unadmitted'
    assert result['production_admitted'] is False
    assert result['comparison']['elements_compared'] == 96


def test_candidate_reports_missing_acceptance_not_absent_launcher():
    # The launcher calls preflight but cannot admit this candidate. Returning
    # an absent-integration diagnosis sends the operator to the wrong repair.
    result = inspect_candidate(*fixture())
    assert result['production_admitted'] is False
    assert 'mandatory pre-rank launcher integration' not in result['missing_admission']
    assert 'complete independently validated production admission' in result['missing_admission']


@pytest.mark.skipif(os.name != 'posix', reason='remote POSIX process execution')
@pytest.mark.parametrize('requirement',[
    'expected_graph_attention_members_profile','expected_graph_network_profile',
    'expected_graph_dims_profile','expected_graph_pointer_profile',
    'expected_graph_non_gemm_scalar_profile'])
def test_contradictory_graph_requirements_reject_before_scorer_start(requirement,tmp_path):
    # Catches moving request validation after the expensive scorer invocation.
    # Controlled subprocess protocol only, not neural/GPU correctness evidence.
    reference,scores,execution,expected=fixture()
    marker=tmp_path/'scorer_started'
    code=('import pathlib,json,sys\n'
        f'pathlib.Path({str(marker)!r}).write_text("started")\n'
        'output=pathlib.Path(sys.argv[3]);output.mkdir()\n'
        f'(output/"execution.json").write_text({json.dumps(execution)!r})\n'
        f'(output/"raw_scores.json").write_text({json.dumps(scores)!r})\n')
    def observe():
        return dict(expected=expected,reference_dir=str(tmp_path),reference_scores=reference,
            require_graph_inventory=False,**{requirement:{}})
    with pytest.raises(ValueError,match='require.*graph inventory'):
        collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',5,observe,reference)
    assert not marker.exists(), 'invalid requirements must not start scorer'


@pytest.mark.parametrize('field', ['reference_sha256', 'manifest_sha256', 'device',
    'outer_microbatch', 'transformer_microbatch', 'lanes', 'executor',
    'loaded_tensor_bindings', 'requested_launch_policies'])
def test_candidate_must_match_independently_observed_execution(field):
    reference, scores, execution, expected = fixture()
    del execution[field]
    with pytest.raises(ValueError):
        inspect_candidate(reference, scores, execution, expected)


@pytest.mark.parametrize('field', ['resolved_execution_contract_complete',
    'kernel_coverage_complete', 'production_quality_accepted'])
def test_self_claimed_promotion_cannot_bypass_missing_resolved_validator(field):
    reference, scores, execution, expected = fixture()
    execution[field] = True
    with pytest.raises(ValueError):
        inspect_candidate(reference, scores, execution, expected)


def test_numeric_failure_cannot_be_admitted_by_a_pass_receipt():
    reference, scores, execution, expected = fixture()
    scores['jobs'][2]['raw_scores'][0][23] += 4
    with pytest.raises(ValueError):
        inspect_candidate(reference, scores, execution, expected)


@pytest.mark.parametrize('mutation', ['ordinal_bool', 'schema_bool', 'quality_integer', 'tensor_float'])
def test_nested_json_type_coercion_cannot_match_device_or_tensor_binding(mutation):
    reference, scores, execution, expected = fixture()
    if mutation == 'ordinal_bool': execution['device']['device'] = False
    elif mutation == 'schema_bool': execution['device']['schema_version'] = True
    elif mutation == 'quality_integer': execution['device']['production_quality_accepted'] = 0
    elif mutation == 'tensor_float': execution['loaded_tensor_bindings']['a.fp16']['size_bytes'] = 2.0
    with pytest.raises(ValueError):
        inspect_candidate(reference, scores, execution, expected)


@pytest.mark.skipif(os.name != 'posix', reason='remote POSIX process-group execution')
def test_scorer_is_actually_executed_and_exit_code_preserved(tmp_path):
    log = tmp_path / 'scorer.log'
    result = run_scorer([sys.executable, '-c', 'print("actual-child"); raise SystemExit(7)'], log, 2)
    assert result == 7
    assert 'actual-child' in log.read_text()


@pytest.mark.skipif(os.name != 'posix', reason='remote POSIX process-group execution')
def test_scorer_binds_explicit_working_directory_and_environment(tmp_path):
    directory=tmp_path/'owned-cwd'
    directory.mkdir()
    log=tmp_path/'scorer.log'
    code='import os; print(os.getcwd()); print(os.environ.get("TASK_TEST_FLAG"))'
    assert run_scorer([sys.executable,'-c',code],log,2,
                      cwd=directory,env={'TASK_TEST_FLAG':'owned-test'})==0
    assert log.read_text().splitlines()==[str(directory),'owned-test']


@pytest.mark.skipif(os.name != 'posix', reason='remote POSIX process-group execution')
def test_scorer_timeout_is_not_a_success(tmp_path):
    with pytest.raises(TimeoutError):
        run_scorer([sys.executable, '-c', 'import time; time.sleep(10)'], tmp_path / 'log', .05)


@pytest.mark.skipif(os.name != 'posix', reason='remote POSIX process-group execution')
def test_exited_parent_cannot_leave_background_descendant(tmp_path):
    marker = tmp_path / 'late-marker'
    started = tmp_path / 'child-started'
    code = ('import os,time,pathlib\npid=os.fork()\n'
            f'if pid:\n while not pathlib.Path({str(started)!r}).exists(): time.sleep(.005)\n os._exit(0)\n'
            f'pathlib.Path({str(started)!r}).write_text("started")\n'
            f'time.sleep(.3)\npathlib.Path({str(marker)!r}).write_text("orphan")\n')
    assert run_scorer([sys.executable, '-c', code], tmp_path / 'log', 2) == 0
    assert started.exists(), 'test must exercise an actual background child'
    time.sleep(.5)
    assert not marker.exists()


@pytest.mark.skipif(os.name != 'posix', reason='remote POSIX process execution')
@pytest.mark.parametrize('mode', ['valid', 'drift', 'bad_scores', 'missing_output', 'exit_failure',
                                'symlink_output', 'reference_mismatch', 'missing_resolved',
                                'resolved_valid','resolved_device','resolved_hash','resolved_promotion',
                                'resolved_specialized_disagreement', 'missing_graph_inventory',
                                'graph_valid','graph_device','graph_hash','graph_promotion',
                                'graph_schema','graph_scope','graph_missing_functions',
                                'graph_attention_valid','graph_attention_malformed','graph_attention_reject',
                                'graph_non_gemm_valid','graph_non_gemm_malformed','graph_non_gemm_reject',
                                'graph_abi_valid','graph_abi_missing','graph_abi_device',
                                'graph_abi_execution_hash','graph_abi_inventory_hash','graph_abi_reject',
                                'graph_abi_attention_valid','graph_abi_attention_reject',
                                'graph_abi_gemm_valid','graph_abi_gemm_reject',
                                'graph_transfer_valid','graph_transfer_missing',
                                'graph_transfer_bad_copy','graph_transfer_device',
                                'graph_transfer_execution_hash','graph_transfer_inventory_hash',
                                'graph_dependency_valid','graph_dependency_missing','graph_dependency_cycle'])
def test_fresh_transaction_executes_owned_scorer_and_cannot_promote(mode, tmp_path, monkeypatch):
    reference, scores, execution, expected=fixture()
    if mode.startswith('graph_'):
        expected['executor']=execution['executor']='native_cuda_graph'
    if mode=='bad_scores': scores['jobs'][0]['raw_scores'][0][0]+=10
    code=('import json,pathlib,sys,os\n'
          'assert sys.argv[1]=="--validate-stream1-scores"\n'
          'output=pathlib.Path(sys.argv[3]); output.mkdir()\n'
          f'pathlib.Path({str(tmp_path / "executed")!r}).write_text("actual")\n')
    if mode!='missing_output':
        code+=(f'(output/"raw_scores.json").write_text({json.dumps(scores)!r})\n'
               f'(output/"execution.json").write_text({json.dumps(execution)!r})\n')
    if mode.startswith('resolved_'):
        resolved=deepcopy(expected)
        resolved.pop("executor")
        resolved.update(canonical_resolved_recipe(expected), specialized_gemm_observations=[])
        resolved.update(schema_version=1,scope='resolved_host_choices_not_kernel_admission',
            production_admitted=False,kernel_coverage_complete=False,
            raw_scores_sha256=hashlib.sha256(json.dumps(scores).encode()).hexdigest())
        if mode=='resolved_device': resolved['device']['uuid_hex']='f'*32
        if mode=='resolved_hash': resolved['raw_scores_sha256']='0'*64
        if mode=='resolved_promotion': resolved['production_admitted']=True
        if mode=='resolved_specialized_disagreement':
            resolved['specialized_gemm_observations']=[{'family':'linear_bias_strided'}]
        code+=f'(output/"resolved_execution.json").write_text({json.dumps(resolved)!r})\n'
    if mode.startswith('graph_'):
        inventory=dict(schema_version=1,scope='captured_graph_kernel_inventory_not_admission',
            production_admitted=False,kernel_coverage_complete=False,
            device=deepcopy(expected['device']),
            execution_sha256=hashlib.sha256(json.dumps(execution).encode()).hexdigest())
        inventory.update(functions=[dict(mangled_name='_Z4testv',attributes=dict(
            binary_version=86,ptx_version=86,max_threads_per_block=1024,num_regs=12,
            static_shared_bytes=0,local_bytes=0,constant_bytes=0,max_dynamic_shared_bytes=49152))],
            lanes=[dict(lane=0,node_count=3,node_types=dict(kernel=1,memcpy=1,memset=1),
                kernels=[dict(function_id=0,grid=[2,1,1],block=[128,1,1],dynamic_shared_bytes=0)])])
        if mode=='graph_missing_functions': del inventory['functions']
        if mode=='graph_device': inventory['device']['uuid_hex']='f'*32
        if mode=='graph_hash': inventory['execution_sha256']='0'*64
        if mode=='graph_promotion': inventory['production_admitted']=True
        if mode=='graph_schema': inventory['schema_version']=True
        if mode=='graph_scope': inventory['scope']='full_admission'
        if mode.startswith('graph_dependency_'):
            inventory['lanes'][0].update(node_count=6,node_types=dict(kernel=2,memcpy=2,memset=2))
            inventory['lanes'][0]['kernels']*=2
        code+=f'(output/"graph_kernel_inventory.json").write_text({json.dumps(inventory)!r})\n'
        if mode.startswith(('graph_transfer_','graph_dependency_')) and mode!='graph_transfer_missing':
            transfer=dict(schema_version=1,scope='captured_graph_transfers_not_admission',
                production_admitted=False,device=deepcopy(expected['device']),
                execution_sha256=hashlib.sha256(json.dumps(execution).encode()).hexdigest(),
                inventory_sha256=hashlib.sha256(json.dumps(inventory).encode()).hexdigest(),
                lanes=[dict(lane=0,operations=[
                    dict(kind='memset',destination='raw_scores',destination_offset=0,
                         bytes=96,value=0,element_size=1),
                    dict(kind='memset',destination='numeric_error',destination_offset=0,
                         bytes=4,value=0,element_size=1),
                    dict(kind='memcpy',source='lane_logits',source_offset=0,
                         destination='raw_scores',destination_offset=0,bytes=48,
                         direction='device_to_device'),
                    dict(kind='memcpy',source='lane_logits',source_offset=0,
                         destination='raw_scores',destination_offset=48,bytes=48,
                         direction='device_to_device')])])
            if mode=='graph_transfer_bad_copy':transfer['lanes'][0]['operations'][-1]['bytes']=24
            if mode=='graph_transfer_device':transfer['device']['uuid_hex']='0'*32
            if mode=='graph_transfer_execution_hash':transfer['execution_sha256']='0'*64
            if mode=='graph_transfer_inventory_hash':transfer['inventory_sha256']='0'*64
            if mode.startswith('graph_dependency_') and mode!='graph_dependency_missing':
                lane=transfer['lanes'][0]
                lane.update(node_count=6,kernel_node_ids=[2,4],dependency_edges=[[0,1],[1,2],[2,3],[3,4],[4,5]])
                for op,node in zip(lane['operations'],[0,1,3,5]):op['node_id']=node
                if mode=='graph_dependency_cycle':lane['dependency_edges'].append([5,0])
            code+=f'(output/"graph_transfers.json").write_text({json.dumps(transfer)!r})\n'
        if mode.startswith('graph_abi_') and mode!='graph_abi_missing':
            abi=dict(device=deepcopy(expected['device']),
                execution_sha256=hashlib.sha256(json.dumps(execution).encode()).hexdigest(),
                inventory_sha256=hashlib.sha256(json.dumps(inventory).encode()).hexdigest())
            if mode=='graph_abi_device':abi['device']['uuid_hex']='f'*32
            if mode=='graph_abi_execution_hash':abi['execution_sha256']='0'*64
            if mode=='graph_abi_inventory_hash':abi['inventory_sha256']='0'*64
            code+=f'(output/"graph_parameter_layout.json").write_text({json.dumps(abi)!r})\n'
    if mode=='exit_failure': code+='raise SystemExit(7)\n'
    if mode=='symlink_output':
        code+='old=output.parent/"old"; output.rename(old); output.symlink_to(old,target_is_directory=True)\n'
    calls=[]
    attention_calls=[]
    def validate_attention(inventory, model, profile, storage):
        attention_calls.append(1)
        assert model=={'trusted_model':1} and storage=={'trusted_storage':1}
        assert profile['expected_padded_seq_len']==57
        assert profile['device']==expected['device']
        if mode=='graph_attention_reject':raise ValueError('attention mismatch')
    monkeypatch.setattr('tools.graph_attention_profile.validate_graph_attention_profile',validate_attention)
    non_gemm_calls=[]
    def validate_non_gemm(inventory,model,profile,shared):
        non_gemm_calls.append(1)
        assert model=={'trusted_model':1} and shared==16
        assert profile['expected_padded_seq_len']==57
        assert profile['device']==expected['device']
        if mode=='graph_non_gemm_reject':raise ValueError('non-GEMM mismatch')
    monkeypatch.setattr('tools.graph_non_gemm_profile.validate_graph_non_gemm_profile',validate_non_gemm)
    abi_calls=[]
    def validate_abi(payload,inventory,profile,storage,attention=None,gemm_storage=None):
        abi_calls.append(1)
        assert storage=={'trusted_storage':1} and profile['device']==expected['device']
        if mode=='graph_abi_reject':raise ValueError('parameter ABI mismatch')
        if mode.startswith('graph_abi_attention_'):
            assert attention=={'trusted_attention_storage':1}
            if mode=='graph_abi_attention_reject':raise ValueError('attention ABI mismatch')
        if mode.startswith('graph_abi_gemm_'):
            assert gemm_storage=={'trusted_gemm_storage':1}
            if mode=='graph_abi_gemm_reject':raise ValueError('GEMM ABI mismatch')
    monkeypatch.setattr('tools.graph_parameter_layout.validate_non_gemm_parameter_layout',validate_abi)
    def observe():
        calls.append(1)
        result=dict(expected=deepcopy(expected), reference_dir=str(tmp_path), runner_sha256='e'*64,
                    reference_scores=deepcopy(reference))
        if mode=='missing_resolved' or mode.startswith('resolved_'):
            result['expected_resolved']=canonical_resolved_recipe(expected)
        if mode=='missing_graph_inventory' or mode.startswith('graph_'):
            result['require_graph_inventory']=True
        if mode.startswith(('graph_transfer_','graph_dependency_')):
            result['require_graph_transfers']=True
        if mode.startswith('graph_dependency_'):result['require_graph_dependencies']=True
        if mode.startswith('graph_attention_'):
            result['expected_graph_attention_profile']=dict(model={'trusted_model':1},
                padded_seq_len=57,storage={'trusted_storage':1})
            if mode=='graph_attention_malformed':del result['expected_graph_attention_profile']['storage']
        if mode.startswith('graph_non_gemm_'):
            result['expected_graph_non_gemm_profile']=dict(model={'trusted_model':1},
                padded_seq_len=57,ln_shared_bytes=16)
            if mode=='graph_non_gemm_malformed':del result['expected_graph_non_gemm_profile']['ln_shared_bytes']
        if mode.startswith('graph_abi_'):
            result['expected_graph_parameter_layout']={'trusted_storage':1}
        if mode.startswith('graph_abi_attention_'):
            result['expected_graph_attention_parameter_storage']={'trusted_attention_storage':1}
        if mode.startswith('graph_abi_gemm_'):
            result['expected_graph_gemm_parameter_storage']={'trusted_gemm_storage':1}
        if mode=='reference_mismatch': result['reference_scores'][0][0]+=1
        if mode=='drift' and len(calls)>1: result['runner_sha256']='f'*64
        return result
    if mode in ('valid','resolved_valid','graph_valid','graph_attention_valid','graph_non_gemm_valid','graph_abi_valid','graph_abi_attention_valid','graph_abi_gemm_valid','graph_transfer_valid','graph_dependency_valid'):
        result=collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',2,observe,reference)
        assert result['production_admitted'] is False
        assert result['status']=='numeric_pass_unadmitted'
        assert len(calls)==2 and (tmp_path/'run/comparison.json').exists()
        if mode=='resolved_valid': assert result['resolved_bindings_checked'] is True
        if mode=='graph_valid': assert result['graph_inventory_bindings_checked'] is True
        if mode=='graph_transfer_valid':assert result['graph_transfer_arguments_checked'] is True
        if mode=='graph_dependency_valid':assert result['graph_dependency_order_checked'] is True
        if mode=='graph_attention_valid':
            assert result['graph_attention_profile_checked'] is True
            assert attention_calls==[1]
        if mode=='graph_non_gemm_valid':
            assert result['graph_non_gemm_profile_checked'] is True
            assert non_gemm_calls==[1]
        if mode=='graph_abi_valid':
            assert result['graph_non_gemm_parameter_layout_checked'] is True
            assert abi_calls==[1]
        if mode=='graph_abi_attention_valid':
            assert result['graph_attention_parameter_layout_checked'] is True
            assert abi_calls==[1]
        if mode=='graph_abi_gemm_valid':
            assert result['graph_gemm_parameter_layout_checked'] is True
            assert abi_calls==[1]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',2,observe,reference)
        assert not (tmp_path/'run/comparison.json').exists()
    if mode=='reference_mismatch':
        assert not (tmp_path/'executed').exists(), 'reject reference mismatch before scorer startup'
    else:
        assert (tmp_path/'executed').exists(), 'must actually execute scorer, not inspect old receipt'
