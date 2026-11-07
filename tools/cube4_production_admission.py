"""Positive admission of an exact observed 2/8 RTX3060 profile/cohort.

This decision never claims arbitrary input quality, training disjointness,
complete kernel argument bytes, or acceptance of a different build/profile.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.cube4_profile import unique
from tools.cube4_numeric_gate import validate_reference,validate_checkpoint_binding,POLICY
from tools.cube4_production_preflight import inspect_candidate,validate_saved_candidate
from tools.beam_smoke_fixture import verify_submission
from tools.cube4_bundle_contract import validate_cube4_weights
from tools.nccl_library_observation import observe as observe_nccl

REQUIRED = ('inputs_unchanged','resolved_bindings_checked','executed_graph_replay_checked',
    'graph_inventory_bindings_checked','graph_inventory_structure_checked',
    'graph_gemm_public_iterators_checked','graph_gemm_launch_values_checked',
    'graph_gemm_mainloop_iterators_checked','graph_gemm_profile_checked',
    'graph_attention_profile_checked','graph_non_gemm_profile_checked',
    'graph_non_gemm_parameter_layout_checked','graph_attention_parameter_layout_checked',
    'graph_gemm_parameter_layout_checked','graph_transfer_arguments_checked',
    'graph_transfer_pitched_geometry_checked','graph_dependency_order_checked',
    'graph_non_gemm_u32_values_checked','graph_known_pointer_roles_checked',
    'graph_embedded_dims_checked','graph_consumed_network_members_checked',
    'graph_attention_members_checked','canonical_empty_elementwise_arguments_checked',
    'public_parameter_field_storage_checked','nested_gemm_storage_checked')


def raw(path, limit=64*1024*1024):
    path=Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('artifact must be regular, not a symlink')
    with path.open('rb') as stream: value=stream.read(limit+1)
    if len(value)>limit:raise ValueError('artifact too large')
    return value


def digest(path):return hashlib.sha256(raw(path,512*1024*1024)).hexdigest()


def read(path):
    value=json.loads(raw(path),object_pairs_hook=unique)
    if not isinstance(value,dict):raise ValueError('object required')
    return value


def validate(source,runner,build_receipt,transaction):
    source=Path(source).resolve(strict=True);runner=Path(runner).resolve(strict=True)
    build=read(build_receipt);tx=read(transaction);identity=digest(runner)
    if type(build.get('schema_version')) is not int or build['schema_version']!=1 or build.get('status')!='pass' or build.get('runner_sha256')!=identity:
        raise ValueError('missing/stale build transaction')
    directory=Path(build_receipt).parent
    for name,key in (('build.log','build_log_sha256'),('CMakeCache.snapshot.txt','cmake_cache_sha256'),
                     ('ninja-deps.snapshot.bin','ninja_deps_sha256')):
        if digest(directory/name)!=build[key]:raise ValueError('build provenance artifact changed')
    if digest(build['compiler_path'])!=build['compiler_sha256'] or digest(build['nccl_header_path'])!=build['nccl_header_sha256']:
        raise ValueError('compiler/header binding changed')
    if digest(build['host_compiler_path'])!=build['host_compiler_sha256']:
        raise ValueError('host compiler changed')
    if observe_nccl(runner)!=build['runtime_library']:raise ValueError('loaded NCCL changed')
    cutlass=Path(build['cutlass_path']).resolve(strict=True)
    if not build['cutlass_header_hashes']:raise ValueError('missing external header provenance')
    for name,sha in build['cutlass_header_hashes'].items():
        path=(cutlass/name).resolve(strict=True)
        if not path.is_relative_to(cutlass) or digest(path)!=sha:raise ValueError('CUTLASS headers changed')
    if digest(source/'source_manifest.json')!=build['source_manifest_sha256']:
        raise ValueError('source manifest changed after build')
    if read(source/'source_manifest.json')!=build['source_hashes']:
        raise ValueError('build source hashes differ from manifest')
    for name,sha in build['source_hashes'].items():
        target=(source/name).resolve(strict=True)
        if not target.is_relative_to(source) or digest(target)!=sha:raise ValueError('stale/escaping source')
    if type(tx.get('schema_version')) is not int or tx['schema_version']!=1 or tx.get('runner_sha256')!=identity or type(tx.get('world')) is not int or tx['world'] not in (2,8):
        raise ValueError('unsupported or stale search transaction')
    cohort=Path(tx['cohort']).resolve(strict=True);world=tx['world']
    hardware=build['hardware']
    if len(hardware)!=world or any('RTX 3060' not in row['name'] or row['memory_mib']!=12288 for row in hardware):
        raise ValueError('wrong GPU count/model/VRAM')
    policy=read(cohort/'policy.json');reference=read(cohort/'reference.json')
    if digest(cohort/'policy.json')!=tx['policy_sha256'] or digest(cohort/'reference.json')!=tx['reference_sha256']:
        raise ValueError('policy/reference changed')
    if policy.get('numeric')!=POLICY or policy.get('reference_count')!=64 or policy.get('solve')!={
            'beam':4096,'depth_limit':12,'minimum_solved':64,'per_case_timeout_seconds':180}:
        raise ValueError('unapproved or weakened acceptance policy')
    if any(policy.get(key) is not True for key in ('strict_rank_memcheck_required',
            'immutable_checkpoint_required','baseline_and_optimized_required')):
        raise ValueError('missing mandatory preregistered gate')
    scores=validate_reference(reference)
    if len(scores)!=64:raise ValueError('partial cohort')
    if policy['pair_contract_sha256']!=digest(cohort/'pairs.json'):raise ValueError('pair contract changed')
    for name,sha in read(cohort/'immutable.json').items():
        if digest(cohort/name)!=sha:raise ValueError('immutable fixture changed')
    manifest=read(cohort/'weights_fp16/manifest.json')
    validate_checkpoint_binding(reference,manifest);validate_cube4_weights(cohort/'weights_fp16')
    modes=tx.get('modes')
    if not isinstance(modes,list) or [m.get('mode') for m in modes]!=['baseline','optimized']:
        raise ValueError('baseline/optimized transaction missing or reordered')
    puzzle=read(cohort/'data/puzzle_info.json');evidence_hashes={}
    for mode in modes:
        numeric=Path(mode['numeric_dir']).resolve(strict=True)
        summary=read(numeric/'summary.json')
        if summary.get('status')!='numeric_pass_unadmitted' or len(summary.get('devices',[]))!=world:
            raise ValueError('missing per-device numeric evidence')
        evidence_hashes[str(numeric/'summary.json')]=digest(numeric/'summary.json')
        for rank,reported in enumerate(summary['devices']):
            candidate=numeric/f'device{rank}'
            report=read(candidate/'comparison.json')
            if report!=reported or any(report.get(key) is not True for key in REQUIRED):
                raise ValueError('partial/stale per-device checks')
            bindings=report['observed_bindings']
            if validate_saved_candidate(candidate,bindings,scores)!=report:
                raise ValueError('saved numeric/graph/argument receipt fails independent recomputation')
            expected=bindings['expected']
            if bindings['runner_sha256']!=identity or expected['device']['device']!=rank or expected['device']['sm']!=86:
                raise ValueError('wrong runner/device profile')
            uuid=hardware[rank]['uuid'].removeprefix('GPU-').replace('-','').lower()
            if expected['device']['uuid_hex']!=uuid:raise ValueError('device UUID differs from hardware observation')
            if expected['reference_sha256']!=tx['reference_sha256']:
                raise ValueError('different numeric reference')
            for filename,sha in report['candidate_content_hashes'].items():
                path=candidate/'output'/filename
                if path.parent!=candidate/'output' or digest(path)!=sha:raise ValueError('numeric output changed')
                evidence_hashes[str(path)]=sha
            log_path=candidate/'runner.log'
            if digest(log_path)!=report['executed_graph_replay']['trace_sha256']:
                raise ValueError('executed replay trace changed')
            evidence_hashes[str(log_path)]=digest(log_path)
            actual=read(candidate/'output/raw_scores.json');execution=read(candidate/'output/execution.json')
            recomputed=inspect_candidate(scores,actual,execution,expected)
            if recomputed['comparison']!=report['comparison']:raise ValueError('numeric comparison differs')
            coverage=report.get('graph_argument_coverage',{})
            if any(type(coverage.get(key)) is not int or coverage[key]<=0 for key in
                   ('non_gemm_nodes','gemm_nodes','attention_nodes','covered_argument_slots')):
                raise ValueError('missing composed public member coverage')
            # Keep the original narrow byte-coverage statement, even on pass.
            if coverage.get('parameter_values_checked') is not False:
                raise ValueError('unsupported full argument-byte promotion')
            profile=mode['profile']
            for key,value in expected['requested_launch_policies'].items():
                if profile.get(key)!=value:raise ValueError('search/scorer selector differs')
            for key,name in (('BEAM_B_MICRO','outer_microbatch'),
                ('BEAM_STREAM1_TRANSFORMER_MICRO','transformer_microbatch'),('BEAM_STREAM1_CONCURRENCY','lanes')):
                if profile.get(key)!=str(expected[name]):raise ValueError('search/scorer shape differs')
        cases=mode.get('cases')
        if not isinstance(cases,list) or len(cases)!=64:raise ValueError('missing search rows')
        for row,case in enumerate(cases):
            pid=954000+row;directory=Path(case['directory']).resolve(strict=True)
            request=read(directory/'request.json')
            if (case.get('puzzle_id')!=pid or case.get('row')!=row or case.get('status')!='solved' or
                request.get('profile')!=mode['profile'] or request.get('world')!=world or
                request.get('runner_sha256')!=identity or request.get('puzzle_id')!=pid or
                request.get('policy_sha256')!=tx['policy_sha256'] or
                request.get('reference_sha256')!=tx['reference_sha256'] or
                request.get('initial_state')!=reference['states'][row]):raise ValueError('wrong search identity/row/profile')
            for rank in range(world):
                status_path=directory/'rank-status'/f'rank-{rank}.json'
                status=read(status_path);log=raw(directory/f'rank{rank}.log').decode('utf-8')
                if (any(type(status.get(key)) is not int for key in ('rank','puzzle_id','exit_code')) or
                    status.get('rank')!=rank or status.get('puzzle_id')!=pid or status.get('exit_code')!=0 or
                    status.get('status')!='solved' or log.count('ERROR SUMMARY: 0 errors')!=1 or
                    '0 bytes leaked in 0 allocations' not in log):raise ValueError('rank failure/missing sanitizer')
                evidence_hashes[str(status_path)]=digest(status_path)
                evidence_hashes[str(directory/f'rank{rank}.log')]=digest(directory/f'rank{rank}.log')
            submissions=list((directory/'test_results').glob(f'submit_p{pid}_*.csv'))
            if len(submissions)!=1:raise ValueError('ambiguous/missing solution')
            replay=verify_submission(submissions[0],dict(puzzle_id=pid,initial=reference['states'][row],
                central=puzzle['central_state'],minimum_depth=0),
                dict(move_names=list(puzzle['generators']),moves=list(puzzle['generators'].values())),12)
            if replay!=case.get('replay'):raise ValueError('independent replay disagrees')
            evidence_hashes[str(submissions[0])]=digest(submissions[0])
    return dict(schema_version=1,status='pass',production_admitted=True,
        scope='exact_source_bound_3060_profile_and_64_state_validation_cohort',
        world=world,runner_sha256=identity,build_receipt_sha256=digest(build_receipt),
        transaction_sha256=digest(transaction),policy_sha256=tx['policy_sha256'],
        reference_sha256=tx['reference_sha256'],source_manifest_sha256=build['source_manifest_sha256'],
        profiles={mode['mode']:mode['profile'] for mode in modes},
        solved_per_profile=64,strict_rank_memchecks=128*world,
        production_quality_accepted=True,quality_scope='preregistered_reference_and_replay_cohort_only',
        arbitrary_input_quality_accepted=False,training_disjointness_proven=False,
        full_kernel_argument_bytes_attested=False,generalization_to_1000_puzzles_proven=False,
        evidence_hashes=evidence_hashes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','runner','build-receipt','transaction','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        result=validate(args.source_root,args.runner,args.build_receipt,args.transaction)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps({key:value for key,value in result.items() if key not in ('evidence_hashes','profiles')}))
        return 0
    except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as error:
        print('production admission rejected: '+str(error),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
