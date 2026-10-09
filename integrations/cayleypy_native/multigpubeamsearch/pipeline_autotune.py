"""Automatic downstream calibration after the measured inference winner."""
from __future__ import annotations
import json
from pathlib import Path
import time

from .calibration_frontier import write_frontiers
from .models import verify_prepared_model
from .pipeline_probe import NativePipelineProbe
from .pipeline_profiles import tune_pipeline
from .errors import NativeBackendError


def tune_downstream(contract, model, runtime, options, devices, beam_width,
                    run_dir, environment, runner, inference):
    if runtime.build_metadata.get('plan_calibration_protocol')=='json-session-v1':
        from .plan_session import NativePlanSession
        with NativePlanSession(runner,environment,len(devices),Path(run_dir)/'plan-session',
                deadline=time.monotonic()+options.calibration_pipeline_seconds) as session:
            return _tune_downstream(contract,model,runtime,options,devices,beam_width,
                run_dir,environment,runner,inference,session)
    return _tune_downstream(contract,model,runtime,options,devices,beam_width,
                run_dir,environment,runner,inference,None)


def _tune_downstream(contract, model, runtime, options, devices, beam_width,
                    run_dir, environment, runner, inference, session):
    directory=Path(run_dir)/'pipeline-calibration';directory.mkdir()
    deadline=time.monotonic()+options.calibration_pipeline_seconds
    search_deadline=deadline-min(30.0,options.calibration_pipeline_seconds*.1)
    world=len(devices);storage=runtime.build_metadata['shape']['storage_len']
    micro=inference['parent_batch']
    baseline={'BEAM_B_MICRO':str(micro),'BEAM_ENSEMBLE_INFERENCE_MICRO':str(micro)}
    native_single=model.backend=='mlp' if model is not None else False
    if native_single:
        baseline['BEAM_B_MICRO']=str(micro*(contract.move_count if model.manifest['output_dim']==1 else 1))
    fixtures=[directory/'frontiers'/f'frontier-rank-{i}.bin' for i in range(world)]
    # The calibration fixture has no solved-neighborhood shortcut. Ordinary
    # execution retains its original touch-BFS settings and requested beam.
    probe_env=dict(environment,BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',
                   BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_SOLVE_BUCKET_MODE='0')
    def verify(texts,plans):
        verify_prepared_model(model,contract)
        for text,plan in zip(texts,plans):
            fields={}
            for line in text.splitlines():
                if '=' in line and ' ' not in line:
                    key,value=line.split('=',1);fields[key]=value
            if fields.get('completed_depths')!='6':return False
            if int(fields.get('last_final_frontier_size','-1'))!=plan['frontier_state_capacity']:return False
            if 'production_runner_error=' in text:return False
        return inference.get('phase')=='inference_verified'
    actual=NativePipelineProbe(runner,probe_env,beam_width,world,directory/'actual-plans',
        fixtures,storage,deadline=search_deadline,verify=verify)
    actual.planning_session=session
    # Failure to admit the real requested beam is an execution error, never a
    # reason to benchmark a smaller beam and silently run that instead.
    try:initial_plans=actual.admit(baseline)
    except ValueError as error:
        raise NativeBackendError('requested beam failed all-rank native admission') from error
    if runtime.build_metadata.get('component_calibration_protocol')=='exact-capacity-v1':
        from .beam_geometry import Shape,memory_shortlist
        from .component_autotune import tune_components
        initial=initial_plans[0]
        baseline.update(BEAM_SHARD_COUNT=str(initial['SHARD_COUNT']),
            BEAM_STREAM3_RING_SLOTS=str(initial['STREAM3_RING_SLOTS']),
            BEAM_STREAM4_ACTIVE_SORT_SLOTS=str(initial['STREAM4_ACTIVE_SORT_SLOTS']),
            BEAM_STREAM4_BATCH_CANDIDATES=str(initial['STREAM4_BATCH_CANDIDATES']))
        shape=Shape(beam_width,world,int(baseline['BEAM_B_MICRO']),contract.move_count,storage,
            alignment=initial['STREAM4_BATCH_ALIGNMENT'],
            capacity_ppm=int(environment.get('BEAM_SHARD_CAPACITY_SCALE_PPM','1250000')),
            receive_ppm=int(environment.get('BEAM_GLOBAL_SPILL_SCALE_PPM','2000000')))
        rows=memory_shortlist(shape,effective_beam=initial['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
            staging_slots=initial['STREAM3_RING_SLOTS'],sort_slots=initial['STREAM4_ACTIVE_SORT_SLOTS'],
            budget_bytes=min(p['gpu_budget_bytes'] for p in initial_plans))
        candidates=[('shards-'+str(r['shards']),dict(baseline,BEAM_SHARD_COUNT=str(r['shards'])))
            for r in rows if r['shards']!=initial['SHARD_COUNT']][:2]
        if len(candidates)<2:
            ring=4 if initial['STREAM3_RING_SLOTS']!=4 else 2
            candidates.append(('staging-'+str(ring),dict(baseline,BEAM_STREAM3_RING_SLOTS=str(ring))))
        data=tune_components(actual,session,initial_plans,baseline,candidates,
            moves=contract.move_count,inference=inference)
        if session is not None:session.close()
        actual.planning_session=None
        verify_prepared_model(model,contract)
        data['requested_beam_width']=beam_width
        if options.calibration_full_frontier:
            from dataclasses import replace
            from .calibration_stats import estimate
            chosen=next(r for r in data['tested'] if r['name']==data['selection'])
            final_plans=chosen['plans']
            actual.deadline=deadline
            receipt=write_frontiers(contract,[p['frontier_state_capacity'] for p in final_plans],
                storage,directory/'frontiers',deadline=deadline,
                max_states=options.calibration_frontier_max_states or final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'])
            samples=[replace(r,profile='selected') for r in actual.measure(data['environment'],final_plans)]
            # Component envelopes omit graph-dependent rejection and pressure.
            # On inexpensive exact frontiers, verify the proxy winner against
            # the baseline rather than certifying a known avoidable slowdown.
            full_comparison=[]
            if data['selection']!='baseline' and final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']<=1048576:
                from .calibration_stats import select
                base=next(r for r in data['tested'] if r['name']=='baseline')
                if base['plans'][0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']!=final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']:
                    raise NativeBackendError('full comparison changed exact frontier')
                baseline_samples=[replace(r,profile='baseline') for r in actual.measure(base['environment'],base['plans'])]
                winner,rejections=select(baseline_samples+samples,baseline='baseline',min_improvement=.02)
                full_comparison=[r.__dict__ for r in baseline_samples+samples]
                data['full_comparison_rejections']=rejections
                if winner.profile=='baseline':
                    data.update(environment=base['environment'],selection='baseline-full-verified')
                    final_plans=base['plans'];samples=baseline_samples
            data.update(phase='pipeline_measured',pipeline_verified=True,fixture=receipt,
                measurement_scope='full_requested_frontier',
                estimate=estimate(samples[0].profile,samples).__dict__,
                measurements=[r.__dict__ for r in samples])
            if full_comparison:data['full_comparison_measurements']=full_comparison
            if runtime.build_metadata.get('calibration_protocol')=='json-session-v1':
                from .matched_probe import measure_matched
                data['matched_stream1']=measure_matched(contract,model,runtime,probe_env,
                    directory/'matched-stream1',receipt,data,micro,deadline=deadline,devices=devices)
        (directory/'selection.json').write_text(json.dumps(data,indent=2))
        return data
    # A measured small-frontier profile cannot certify or optimize a larger
    # frontier. If an explicit fixture budget is exceeded, decline calibration.
    workload=beam_width
    probe=NativePipelineProbe(runner,probe_env,workload,world,directory/'measurements',
        fixtures,storage,deadline=search_deadline,verify=verify)
    probe.planning_session=session
    try:
        plans=probe.admit(baseline)
        receipt=write_frontiers(contract,[row['frontier_state_capacity'] for row in plans],
            storage,directory/'frontiers',deadline=deadline,
            max_states=options.calibration_frontier_max_states or plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'])
    except ValueError as error:
        data={'phase':'not_measured','reason':str(error),'requested_beam_width':beam_width,
              'pipeline_verified':False,'cache_hit':False}
        (directory/'selection.json').write_text(json.dumps(data,indent=2))
        return data
    def admit(env):
        return actual.admit(env)
    fast={}
    if session is not None:
        from .beam_geometry import Shape,memory_shortlist
        initial=initial_plans[0]
        shape=Shape(beam_width,world,int(baseline['BEAM_B_MICRO']),contract.move_count,storage,
            alignment=initial['STREAM4_BATCH_ALIGNMENT'],
            capacity_ppm=int(environment.get('BEAM_SHARD_CAPACITY_SCALE_PPM','1250000')),
            receive_ppm=int(environment.get('BEAM_GLOBAL_SPILL_SCALE_PPM','2000000')))
        baseline['BEAM_SHARD_COUNT']=str(initial['SHARD_COUNT'])
        baseline['BEAM_STREAM3_RING_SLOTS']=str(initial['STREAM3_RING_SLOTS'])
        baseline['BEAM_STREAM4_ACTIVE_SORT_SLOTS']=str(initial['STREAM4_ACTIVE_SORT_SLOTS'])
        rows=memory_shortlist(shape,effective_beam=initial['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
            staging_slots=initial['STREAM3_RING_SLOTS'],sort_slots=initial['STREAM4_ACTIVE_SORT_SLOTS'],
            budget_bytes=min(p['gpu_budget_bytes'] for p in initial_plans))
        candidates=[('shards-'+str(r['shards']),dict(baseline,BEAM_SHARD_COUNT=str(r['shards'])))
            for r in rows if r['shards']!=initial['SHARD_COUNT']][:2]
        if len(candidates)<2:
            ring=4 if initial['STREAM3_RING_SLOTS']!=4 else 2
            candidates.append(('staging-'+str(ring),dict(baseline,BEAM_STREAM3_RING_SLOTS=str(ring))))
        def finish_planning():
            session.close();actual.planning_session=None;probe.planning_session=None
        fast=dict(candidate_profiles=candidates,finish_planning=finish_planning)
    try:
        selection=tune_pipeline(micro,baseline,admit=admit,measure=probe.measure,
            deadline=search_deadline,max_outer=max(micro,min(65536,options.calibration_max_batch*8)),
            tune_outer=not native_single,**fast)
    except ValueError as error:
        raise NativeBackendError('full-pipeline baseline calibration failed: '+str(directory)) from error
    actual.deadline=deadline
    final_plans=actual.admit(selection['environment'])
    verify_prepared_model(model,contract)
    if selection['workload_parents']!=final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']:
        raise NativeBackendError('pipeline calibration attempted to transfer a smaller-frontier profile')
    data=dict(selection,phase='pipeline_measured',requested_beam_width=beam_width,
        requested_beam_effective=final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
        fixture=receipt,pipeline_verified=True,cache_hit=False,
        measurement_scope='full_requested_frontier' if selection['workload_parents']==final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'] else 'bounded_legal_frontier',
        correctness_scope='fixture legality, accepted model readout, native completion; normal solution replay checked by run_native')
    if runtime.build_metadata.get('calibration_protocol')=='json-session-v1':
        from .matched_probe import measure_matched
        try:
            data['matched_stream1']=measure_matched(contract,model,runtime,probe_env,
                directory/'matched-stream1',receipt,selection,micro,deadline=deadline,devices=devices)
        except (ValueError,RuntimeError,TimeoutError) as error:
            data['matched_stream1']={'measurement_scope':'unavailable','reason':str(error)}
    (directory/'selection.json').write_text(json.dumps(data,indent=2))
    return data
