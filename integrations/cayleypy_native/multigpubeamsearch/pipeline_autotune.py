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
    directory=Path(run_dir)/'pipeline-calibration';directory.mkdir()
    deadline=time.monotonic()+options.calibration_pipeline_seconds
    search_deadline=deadline-min(30.0,options.calibration_pipeline_seconds*.1)
    world=len(devices);storage=runtime.build_metadata['shape']['storage_len']
    micro=inference['parent_batch']
    baseline={'BEAM_B_MICRO':str(micro),'BEAM_ENSEMBLE_INFERENCE_MICRO':str(micro)}
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
    # Failure to admit the real requested beam is an execution error, never a
    # reason to benchmark a smaller beam and silently run that instead.
    try:actual.admit(baseline)
    except ValueError as error:
        raise NativeBackendError('requested beam failed all-rank native admission') from error
    workload=min(beam_width,options.calibration_frontier_max_states)
    probe=NativePipelineProbe(runner,probe_env,workload,world,directory/'measurements',
        fixtures,storage,deadline=search_deadline,verify=verify)
    try:
        plans=probe.admit(baseline)
        receipt=write_frontiers(contract,[row['frontier_state_capacity'] for row in plans],
            storage,directory/'frontiers',deadline=deadline,
            max_states=options.calibration_frontier_max_states)
    except ValueError as error:
        data={'phase':'not_measured','reason':str(error),'requested_beam_width':beam_width,
              'pipeline_verified':False,'cache_hit':False}
        (directory/'selection.json').write_text(json.dumps(data,indent=2))
        return data
    def admit(env):
        actual.admit(env)
        return probe.admit(env)
    try:
        selection=tune_pipeline(micro,baseline,admit=admit,measure=probe.measure,
            deadline=search_deadline,max_outer=max(micro,min(65536,options.calibration_max_batch*8)))
    except ValueError as error:
        raise NativeBackendError('full-pipeline baseline calibration failed: '+str(directory)) from error
    actual.deadline=deadline
    final_plans=actual.admit(selection['environment'])
    verify_prepared_model(model,contract)
    data=dict(selection,phase='pipeline_measured',requested_beam_width=beam_width,
        requested_beam_effective=final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'],
        fixture=receipt,pipeline_verified=True,cache_hit=False,
        measurement_scope='full_requested_frontier' if selection['workload_parents']==final_plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'] else 'bounded_legal_frontier',
        correctness_scope='fixture legality, accepted model readout, native completion; normal solution replay checked by run_native')
    (directory/'selection.json').write_text(json.dumps(data,indent=2))
    return data

