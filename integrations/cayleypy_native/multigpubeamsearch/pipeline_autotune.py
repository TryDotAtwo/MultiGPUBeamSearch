"""Automatic downstream calibration after the measured inference winner."""
from __future__ import annotations
import json
from pathlib import Path
import time

from .calibration_frontier import write_frontiers
from .models import verify_prepared_model
from .pipeline_probe import NativePipelineProbe
from .pipeline_profiles import tune_pipeline, rank_frontier_counts
from .errors import NativeBackendError


def tune_downstream(contract, model, runtime, options, devices, beam_width,
                    run_dir, environment, runner, inference, *, planning_session=None):
    if planning_session is not None:
        return _tune_downstream(contract,model,runtime,options,devices,beam_width,
            run_dir,environment,runner,inference,planning_session)
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
    if session is not None:session.set_deadline(deadline)
    search_deadline=deadline-min(30.0,options.calibration_pipeline_seconds*.1)
    world=len(devices);storage=runtime.build_metadata['shape']['storage_len']
    micro=inference['parent_batch']
    baseline={'BEAM_B_MICRO':str(micro),'BEAM_ENSEMBLE_INFERENCE_MICRO':str(micro)}
    native_single=model.backend=='mlp' if model is not None else False
    if native_single:
        baseline['BEAM_B_MICRO']=str(micro*(contract.move_count if model.manifest['output_dim']==1 else 1))
    grow=runtime.build_metadata.get('frontier_calibration_protocol')=='natural-growth-v1'
    fixtures=[directory/'frontiers'/f'frontier-rank-{i}.bin' for i in range(world)]
    # The calibration fixture has no solved-neighborhood shortcut. Ordinary
    # execution retains its original touch-BFS settings and requested beam.
    probe_env=dict(environment,BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',
                   BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_SOLVE_BUCKET_MODE='0')
    def verify(texts,plans):
        verify_prepared_model(model,contract)
        for text,count in zip(texts,rank_frontier_counts(plans)):
            fields={}
            for line in text.splitlines():
                if '=' in line and ' ' not in line:
                    key,value=line.split('=',1);fields[key]=value
            if fields.get('calibration_saturated_steps' if grow else 'completed_depths')!='6':return False
            if int(fields.get('last_final_frontier_size','-1'))!=count:return False
            if 'production_runner_error=' in text:return False
        return inference.get('phase')=='inference_verified'
    actual=NativePipelineProbe(runner,probe_env,beam_width,world,directory/'actual-plans',
        fixtures,storage,deadline=search_deadline,verify=verify,grow=grow)
    actual.planning_session=session
    # Failure to admit the real requested beam is an execution error, never a
    # reason to benchmark a smaller beam and silently run that instead.
    try:initial_plans=actual.admit(baseline)
    except ValueError as error:
        raise NativeBackendError('requested beam failed all-rank native admission') from error
    if runtime.build_metadata.get('component_calibration_protocol')=='exact-capacity-v1':
        from .pipeline_cache import cache_identity,cache_path,read_profile,write_profile
        from . import adaptive_profiles as adaptive
        adaptive_context=adaptive.context(devices,initial_plans,inference)
        adaptive_context["outer_rows_per_parent"]=(contract.move_count if native_single and model.manifest["output_dim"]==1 else 1)
        identity=cache_identity(inference,environment,beam=beam_width,world=world,
            full_frontier=getattr(options,'calibration_full_frontier',False))
        if identity is not None:
            identity['runtime_metadata']=runtime.build_metadata
            identity['adaptive_context']=adaptive_context
        path=cache_path(options.cache_dir,identity) if identity is not None else None
        if path is not None:
            cached=read_profile(path,identity,actual.admit)
            if cached is not None:
                verify_prepared_model(model,contract)
                (directory/'selection.json').write_text(json.dumps(cached,indent=2))
                return cached
        # GitHub records are proposals, not current-host timing receipts. Reuse
        # requires fresh all-rank admission and exact-capacity component probes.
        from .shared_profiles import lookup, pipeline_environment
        shared = lookup(contract, model, runtime, options, devices, beam_width)
        shared_env = pipeline_environment(shared, micro) if shared else None
        if (shared_env is not None and not options.calibration_full_frontier
                and session is not None and session.supports_components):
            from .component_autotune import tune_components
            try:
                shared_plans = actual.admit(shared_env)
                data = tune_components(actual, session, shared_plans, shared_env, [],
                    moves=contract.move_count, inference=inference,
                    candidate_factory=lambda *args: [])
            except ValueError as error:
                (directory/'shared-profile-rejected.json').write_text(json.dumps({
                    'sha256':shared['registry_sha256'], 'reason':str(error)}))
                initial_plans = actual.admit(baseline)
            else:
                verify_prepared_model(model, contract)
                data.update(requested_beam_width=beam_width,
                    shared_profile_sha256=shared['registry_sha256'],
                    shared_profile_validated=True,
                    search_policy='verified GitHub candidate; fresh exact-capacity validation')
                if path is not None:write_profile(path, identity, data)
                (directory/'selection.json').write_text(json.dumps(data, indent=2))
                session.close()
                actual.planning_session = None
                return data
        if options.calibration_full_frontier:
            from .working_capacity import verification_cost
            cost=verification_cost(beam_width,inference,max(.001,deadline-time.monotonic()),
                                   include_reference=False)
            (directory/'verification-budget.json').write_text(json.dumps(cost,indent=2))
            if not cost['fits']:
                raise NativeBackendError(
                    f"full-frontier verification for {beam_width:,} parents has projected "
                    f"inference time {cost['predicted_inference_seconds']:.0f}s versus "
                    f"remaining calibration budget {cost['budget_seconds']:.0f}s; "
                    "no full-frontier benchmark started and no smaller beam substituted")
        from .beam_geometry import Shape,memory_shortlist,ring_pool_limit
        from .component_autotune import tune_components
        initial=initial_plans[0]
        baseline.update(BEAM_SHARD_COUNT=str(initial['SHARD_COUNT']),
            BEAM_STREAM3_RING_SLOTS=str(initial['STREAM3_RING_SLOTS']),
            BEAM_STREAM4_ACTIVE_SORT_SLOTS=str(initial['STREAM4_ACTIVE_SORT_SLOTS']),
            BEAM_STREAM4_BATCH_CANDIDATES=str(initial['STREAM4_BATCH_CANDIDATES']))
        shape=Shape(beam_width,world,int(initial.get('B_MICRO',baseline['BEAM_B_MICRO'])),contract.move_count,storage,
            alignment=initial['STREAM4_BATCH_ALIGNMENT'],
            capacity_ppm=int(environment.get('BEAM_SHARD_CAPACITY_SCALE_PPM','1250000')),
            receive_ppm=int(environment.get('BEAM_GLOBAL_SPILL_SCALE_PPM','2000000')))
        rows=memory_shortlist(shape,effective_beam=initial['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
            staging_slots=initial['STREAM3_RING_SLOTS'],sort_slots=initial['STREAM4_ACTIVE_SORT_SLOTS'],
            budget_bytes=min(p['gpu_budget_bytes'] for p in initial_plans),
            physical_ring_limit=ring_pool_limit(environment))
        candidates=[('shards-'+str(r['shards']),dict(baseline,BEAM_SHARD_COUNT=str(r['shards'])))
            for r in rows if r['shards']!=initial['SHARD_COUNT']][:2]
        if len(candidates)<2:
            ring=4 if initial['STREAM3_RING_SLOTS']!=4 else 2
            candidates.append(('staging-'+str(ring),dict(baseline,BEAM_STREAM3_RING_SLOTS=str(ring))))
        search_trace=dict(context=adaptive_context,rounds=[],stop_reason=None)
        def candidate_factory(rows,plan,service):
            from .component_autotune import service_envelope
            from .profile_heuristics import narrow
            proposed,trace=adaptive.proposals(shape,baseline,adaptive_context,
                service=service,ring_limit=ring_pool_limit(environment))
            pressure=service_envelope(rows,plan,moves=contract.move_count,
                inference_seconds=inference['estimate']['median'],parent_batch=micro)
            proposed,heuristic=narrow(proposed,baseline,pressure,plan=plan,moves=contract.move_count)
            search_trace['heuristic']=heuristic
            search_trace['coarse']=trace
            return [('coarse-'+name,env) for name,env in proposed]
        data=tune_components(actual,session,initial_plans,baseline,candidates,
            moves=contract.move_count,inference=inference,candidate_factory=candidate_factory)
        data['adaptive_search']=search_trace
        data['search_policy']=adaptive.POLICY
        if session is not None:session.close()
        actual.planning_session=None
        verify_prepared_model(model,contract)
        data['requested_beam_width']=beam_width
        if path is not None and not options.calibration_full_frontier:
            write_profile(path,identity,data)
        (directory/'selection.json').write_text(json.dumps(data,indent=2))
        if options.calibration_full_frontier:
            from dataclasses import replace
            from .calibration_stats import select
            # Component timings propose the finite grid. Only full-depth times
            # select its winner, at every width (including widths above 1M).
            from .profile_heuristics import full_challengers
            eligible=adaptive.shortlist(data['tested'],
                count=full_challengers(search_trace.get('heuristic',{})))
            search_trace['coarse_full_candidates']=[r['name'] for r in eligible]
            search_trace['coarse_full_tested']=[]
            base=next(r for r in eligible if r['name']=='baseline')
            final_plans=base['plans']
            actual.deadline=deadline
            receipt=dict(kind='native-natural-growth-v1', requested_beam=beam_width) if grow else write_frontiers(contract,rank_frontier_counts(final_plans),
                storage,directory/'frontiers',deadline=deadline,
                max_states=options.calibration_frontier_max_states or final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'])
            samples=[];completed=[]
            for row in eligible:
                if rank_frontier_counts(row['plans']) != rank_frontier_counts(final_plans):
                    raise NativeBackendError('full comparison changed exact frontier')
                try:
                    measured=[replace(r,profile=row['name'])
                        for r in actual.measure(row['environment'],row['plans'])]
                except ValueError as error:
                    # Baseline correctness and a complete timing cohort remain mandatory.
                    # A failed optional challenger must not discard an accepted baseline.
                    if row['name']=='baseline':raise
                    search_trace.setdefault('coarse_full_rejected',[]).append(
                        dict(name=row['name'],reason=str(error)))
                    search_trace['stop_reason']=('time_budget' if time.monotonic()>=deadline
                        else 'coarse_measurement_failed')
                    break
                samples.extend(measured);completed.append(row)
                search_trace['coarse_full_tested'].append(row['name'])
            eligible=completed
            winner,rejections=select(samples,baseline='baseline',min_improvement=.02)
            if winner.profile=='baseline' and search_trace['stop_reason'] is None:
                search_trace['stop_reason']='no_statistically_separated_gain'
            seen={adaptive.key(r['environment']) for r in data['tested'] if 'environment' in r}
            from .component_autotune import service_envelope
            for round_index in range(adaptive.MAX_ROUNDS):
                if search_trace['stop_reason'] is not None:break
                center=next(r for r in eligible if r['name']==winner.profile)
                proposed,trace=adaptive.proposals(shape,center['environment'],adaptive_context,
                    fine=True,round_index=round_index,ring_limit=ring_pool_limit(environment),seen=seen)
                trace.update(center=winner.profile,tested=[],full_tested=[])
                search_trace['rounds'].append(trace)
                fine_rows=[]
                for name,env in proposed:
                    if time.monotonic()>=deadline:
                        search_trace['stop_reason']='time_budget';break
                    name=f'fine-{round_index+1}-'+name
                    seen.add(adaptive.key(env))
                    try:
                        target=actual.admit(env)
                        if rank_frontier_counts(target)!=rank_frontier_counts(final_plans):
                            raise NativeBackendError('adaptive refinement changed exact frontier')
                        component_rows=actual.measure_components(env,target,move_count=contract.move_count)
                        estimate=service_envelope(component_rows,target[0],moves=contract.move_count,
                            inference_seconds=inference['estimate']['median'],parent_batch=micro)
                        row=dict(name=name,environment=env,plans=target,
                            rank_measurements=component_rows,estimate=estimate)
                        fine_rows.append(row);data['tested'].append(row);trace['tested'].append(name)
                    except ValueError as error:
                        data['tested'].append(dict(name=name,rejected=str(error)))
                previous=winner.profile
                for row in adaptive.shortlist(fine_rows,baseline=False,count=adaptive.FULL_PER_ROUND):
                    if time.monotonic()>=deadline:
                        search_trace['stop_reason']='time_budget';break
                    try:
                        measured=[replace(r,profile=row['name']) for r in actual.measure(row['environment'],row['plans'])]
                    except ValueError as error:
                        trace.setdefault('full_rejected',[]).append(dict(name=row['name'],reason=str(error)))
                        continue
                    samples.extend(measured);eligible.append(row);trace['full_tested'].append(row['name'])
                winner,rejections=select(samples,baseline='baseline',min_improvement=.02)
                trace['winner']=winner.profile
                if search_trace['stop_reason']=='time_budget':break
                if fine_rows and not trace['full_tested']:
                    search_trace['stop_reason']='no_valid_neighbor_measurements';break
                if not proposed or not fine_rows:
                    search_trace['stop_reason']='no_new_admissible_neighbors';break
                local_winner,_=select(samples,baseline=previous,min_improvement=.02)
                if local_winner.profile==previous:
                    search_trace['stop_reason']='no_statistically_separated_gain';break
            if search_trace['stop_reason'] is None:search_trace['stop_reason']='round_limit'
            search_trace['search_complete']=search_trace['stop_reason'] in {
                'round_limit','no_new_admissible_neighbors','no_statistically_separated_gain'}
            search_trace['scope']='best statistically separated measured profile in bounded hardware-derived domain'
            data['full_tested']=[r['name'] for r in eligible]
            chosen=next(r for r in eligible if r['name']==winner.profile)
            final_plans=chosen['plans']
            data.update(phase='pipeline_measured',pipeline_verified=True,fixture=receipt,
                environment=chosen['environment'],selection=winner.profile,
                measurement_scope='full_requested_frontier',
                selection_objective='fastest statistically separated full step in hardware-adaptive coarse-to-fine search',
                full_grid_complete=not search_trace.get('coarse_full_rejected'),
                full_comparison_rejections=rejections,
                estimate=winner.__dict__,
                measurements=[r.__dict__ for r in samples if r.profile==winner.profile],
                full_comparison_measurements=[r.__dict__ for r in samples])
            # Preserve the measured winner and the explicit finite-search stop reason.
            (directory/'selection.json').write_text(json.dumps(data,indent=2))
        if path is not None and options.calibration_full_frontier:
            write_profile(path,identity,data)
        (directory/'selection.json').write_text(json.dumps(data,indent=2))
        return data
    # A measured small-frontier profile cannot certify or optimize a larger
    # frontier. If an explicit fixture budget is exceeded, decline calibration.
    workload=beam_width
    probe=NativePipelineProbe(runner,probe_env,workload,world,directory/'measurements',
        fixtures,storage,deadline=search_deadline,verify=verify,grow=grow)
    probe.planning_session=session
    try:
        plans=probe.admit(baseline)
        receipt=dict(kind='native-natural-growth-v1', requested_beam=beam_width) if grow else write_frontiers(contract,rank_frontier_counts(plans),
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
        from .beam_geometry import Shape,memory_shortlist,ring_pool_limit
        initial=initial_plans[0]
        shape=Shape(beam_width,world,int(initial.get('B_MICRO',baseline['BEAM_B_MICRO'])),contract.move_count,storage,
            alignment=initial['STREAM4_BATCH_ALIGNMENT'],
            capacity_ppm=int(environment.get('BEAM_SHARD_CAPACITY_SCALE_PPM','1250000')),
            receive_ppm=int(environment.get('BEAM_GLOBAL_SPILL_SCALE_PPM','2000000')))
        baseline['BEAM_SHARD_COUNT']=str(initial['SHARD_COUNT'])
        baseline['BEAM_STREAM3_RING_SLOTS']=str(initial['STREAM3_RING_SLOTS'])
        baseline['BEAM_STREAM4_ACTIVE_SORT_SLOTS']=str(initial['STREAM4_ACTIVE_SORT_SLOTS'])
        rows=memory_shortlist(shape,effective_beam=initial['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
            staging_slots=initial['STREAM3_RING_SLOTS'],sort_slots=initial['STREAM4_ACTIVE_SORT_SLOTS'],
            budget_bytes=min(p['gpu_budget_bytes'] for p in initial_plans),
            physical_ring_limit=ring_pool_limit(environment))
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
    if grow:
        data['matched_stream1']={'measurement_scope':'unavailable',
            'reason':'natural search does not export its evolving frontier; isolated inference is a separate workload'}
    elif runtime.build_metadata.get('calibration_protocol')=='json-session-v1':
        from .matched_probe import measure_matched
        try:
            data['matched_stream1']=measure_matched(contract,model,runtime,probe_env,
                directory/'matched-stream1',receipt,selection,micro,deadline=deadline,devices=devices)
        except (ValueError,RuntimeError,TimeoutError) as error:
            data['matched_stream1']={'measurement_scope':'unavailable','reason':str(error)}
    (directory/'selection.json').write_text(json.dumps(data,indent=2))
    return data
