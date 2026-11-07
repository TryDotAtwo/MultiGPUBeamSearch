"""Reject mutations of a real completed remote admission transaction."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import pytest
from tools.cube4_production_admission import validate,read

FIXTURE=os.environ.get('BEAM_TEST_PRODUCTION_ADMISSION_FIXTURE')
pytestmark=pytest.mark.skipif(not FIXTURE,reason='fresh completed remote transaction required')


def inputs():
    value=json.loads(Path(FIXTURE).read_text())
    return value,value['source'],value['runner'],value['build_receipt'],value['transaction']


def test_actual_transaction_has_positive_scoped_admission():
    _,source,runner,build,tx=inputs()
    decision=validate(source,runner,build,tx)
    assert decision['production_admitted'] is True
    assert decision['solved_per_profile']==64
    assert decision['arbitrary_input_quality_accepted'] is False
    assert decision['full_kernel_argument_bytes_attested'] is False
    assert decision['training_disjointness_proven'] is False


@pytest.mark.parametrize('mutation',['binary','world','world_bool','missing_mode','duplicate_mode',
    'reordered_modes','policy','reference','partial_cases','wrong_case_id','wrong_row',
    'unsolved','missing_numeric','shape','selector'])
def test_actual_transaction_mutation_rejects_before_promotion(mutation,tmp_path):
    _,source,runner,build,path=inputs();tx=copy.deepcopy(read(path))
    if mutation=='binary':tx['runner_sha256']='0'*64
    elif mutation=='world':tx['world']=4
    elif mutation=='world_bool':tx['world']=True
    elif mutation=='missing_mode':tx['modes']=tx['modes'][:1]
    elif mutation=='duplicate_mode':tx['modes'][1]['mode']='baseline'
    elif mutation=='reordered_modes':tx['modes'].reverse()
    elif mutation=='policy':tx['policy_sha256']='0'*64
    elif mutation=='reference':tx['reference_sha256']='0'*64
    elif mutation=='partial_cases':tx['modes'][0]['cases'].pop()
    elif mutation=='wrong_case_id':tx['modes'][0]['cases'][0]['puzzle_id']+=1
    elif mutation=='wrong_row':tx['modes'][0]['cases'][0]['row']=1
    elif mutation=='unsolved':tx['modes'][0]['cases'][0]['status']='unsolved'
    elif mutation=='missing_numeric':tx['modes'][0]['numeric_dir']=str(tmp_path/'missing')
    elif mutation=='shape':tx['modes'][0]['profile']['BEAM_B_MICRO']='1024'
    elif mutation=='selector':tx['modes'][0]['profile']['BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY']='1'
    destination=tmp_path/'transaction.json';destination.write_text(json.dumps(tx))
    with pytest.raises((ValueError,OSError,KeyError,TypeError)):validate(source,runner,build,destination)


def test_corrupted_raw_inventory_rejects_even_with_rewritten_hash_and_pass_receipt(tmp_path):
    _,source,runner,build,path=inputs();tx=copy.deepcopy(read(path))
    original=Path(tx['modes'][0]['numeric_dir']);directory=tmp_path/'numeric'
    shutil.copytree(original,directory)
    tx['modes'][0]['numeric_dir']=str(directory)
    candidate=directory/'device0';target=candidate/'output/graph_kernel_inventory.json'
    value=read(target);value['lanes'][0]['kernels'][0]['function_id']=True
    target.write_text(json.dumps(value))
    report=read(candidate/'comparison.json')
    report['candidate_content_hashes']['graph_kernel_inventory.json']=hashlib.sha256(target.read_bytes()).hexdigest()
    (candidate/'comparison.json').write_text(json.dumps(report))
    summary=read(directory/'summary.json');summary['devices'][0]=report
    (directory/'summary.json').write_text(json.dumps(summary))
    destination=tmp_path/'transaction.json';destination.write_text(json.dumps(tx))
    with pytest.raises((ValueError,OSError,KeyError,TypeError)):validate(source,runner,build,destination)


@pytest.mark.parametrize('mutation',['rank_failure','wrong_rank','bool_rank','bool_exit','missing_status','missing_sanitizer',
    'missing_csv','duplicate_csv','wrong_csv_id','wrong_replay','wrong_initial','changed_request_profile'])
def test_actual_rank_and_result_mutations_reject(mutation,tmp_path):
    _,source,runner,build,path=inputs();tx=copy.deepcopy(read(path))
    original=Path(tx['modes'][0]['cases'][0]['directory']);directory=tmp_path/'case'
    shutil.copytree(original,directory,ignore=shutil.ignore_patterns('history','history-disk*'))
    tx['modes'][0]['cases'][0]['directory']=str(directory)
    if mutation in ('rank_failure','wrong_rank','bool_rank','bool_exit'):
        target=directory/'rank-status/rank-0.json';value=read(target)
        if mutation=='rank_failure':value['exit_code']=9
        elif mutation=='wrong_rank':value['rank']=1
        elif mutation=='bool_rank':value['rank']=False
        else:value['exit_code']=False
        target.write_text(json.dumps(value))
    elif mutation=='missing_status':(directory/'rank-status/rank-0.json').unlink()
    elif mutation=='missing_sanitizer':(directory/'rank0.log').write_text('ERROR SUMMARY: 1 error\n')
    elif mutation in ('missing_csv','duplicate_csv','wrong_csv_id','wrong_replay'):
        target=next((directory/'test_results').glob('submit_p954000_*.csv'))
        if mutation=='missing_csv':target.unlink()
        elif mutation=='duplicate_csv':shutil.copyfile(target,target.with_name('submit_p954000_duplicate.csv'))
        elif mutation=='wrong_csv_id':target.write_text(target.read_text().replace('954000','954001'))
        else:target.write_text('initial_state_id,path\n954000,\n')
    else:
        target=directory/'request.json';value=read(target)
        if mutation=='wrong_initial':value['initial_state'][0]=(value['initial_state'][0]+1)%6
        else:value['profile']['BEAM_B_MICRO']='1'
        target.write_text(json.dumps(value))
    destination=tmp_path/'transaction.json';destination.write_text(json.dumps(tx))
    with pytest.raises((ValueError,OSError,KeyError,TypeError)):validate(source,runner,build,destination)
