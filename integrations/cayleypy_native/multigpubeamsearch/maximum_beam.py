"""Maximum aligned frontier admission for the current model and GPU cohort."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import math
from collections import Counter
from .beam_capacity import CapacityRejected,find_capacity
from .plan_session import NativePlanSession


def capacity_seed(calibration):
    """Reserve the fastest verified inference cohort before sizing the frontier."""
    from .inference_admission import measured_candidates
    candidates=measured_candidates(calibration)
    if not candidates:raise CapacityRejected('no verified inference batch for maximum frontier')
    seed=min(candidates,key=lambda row:row['estimate']['median'])
    return dict(calibration,**seed,capacity_bootstrap=dict(
        objective='fastest verified inference first; frontier uses remaining memory',
        unconstrained_batch=calibration['parent_batch']))


def refine_maximum(initial_capacity,initial_inference,*,tune,discover,max_rounds=3):
    """Bootstrap an unknown width, then bind inference to its exact width.

    A changed inference batch/reserve invalidates admission and requires a new
    capacity search. Never label the bootstrap's small-beam cache as exact.
    """
    capacity=initial_capacity;previous=initial_inference;history=[]
    for round_index in range(max_rounds):
        beam=capacity['effective_beam'];calibration=tune(beam,round_index)
        if calibration.get('signature',{}).get('requested_beam_width')!=beam:
            raise ValueError('maximum inference receipt is not bound to the exact frontier')
        rejection=calibration.get('rejected',{}).get(str(previous['parent_batch']), '')
        if 'GPU telemetry' in rejection:
            raise RuntimeError('cannot confirm inference winner: previous fastest batch '
                'has missing, incomplete or throttled GPU telemetry; maximum frontier '
                'must not expand by treating that cohort as a slower batch')
        changed=any(calibration[key]!=previous[key] for key in ('parent_batch','reserve_bytes'))
        if changed:capacity=discover(calibration,round_index)
        history.append(dict(beam=beam,parent_batch=calibration['parent_batch'],
                            reserve_bytes=calibration['reserve_bytes'],readmitted=changed))
        if capacity['effective_beam']==beam:
            capacity['inference_refinement']=history
            return capacity,calibration
        previous=calibration
    raise RuntimeError('maximum frontier/inference calibration did not converge')


def discover_capacity(contract,runtime,devices,environment,runner,run_dir,*,seconds):
    import torch
    if runtime.build_metadata.get('plan_calibration_protocol')!='json-session-v1':
        raise RuntimeError('maximum beam requires the verified persistent native planner')
    world=len(devices);storage=runtime.build_metadata['shape']['storage_len']
    ceiling=min(torch.cuda.get_device_properties(d).total_memory for d in devices)*world//storage
    state_bound=math.factorial(contract.state_len)
    for count in Counter(contract.center).values():state_bound//=math.factorial(count)
    # This is a mathematical upper bound, not a claim that the generators reach
    # the entire orbit. Exclude the same solved states as the calibration fixture.
    excluded={contract.center}
    for generator in contract.generators:
        inverse=[0]*contract.state_len
        for index,value in enumerate(generator):inverse[value]=index
        excluded.add(tuple(contract.center[index] for index in inverse))
    ceiling=min(ceiling,max(0,state_bound-len(excluded)))
    alignment=int(environment.get('BEAM_STREAM4_BATCH_ALIGNMENT','1024'))
    deadline=time.monotonic()+seconds;started=time.monotonic()
    best=None;probes=0;complete=True;rejections=[]
    with NativePlanSession(runner,environment,world,Path(run_dir)/'capacity-session',deadline=deadline) as session:
        for shards in range(1,129):
            if time.monotonic()>=deadline:
                complete=False;break
            profile=dict(BEAM_B_MICRO=environment['BEAM_B_MICRO'],BEAM_SHARD_COUNT=str(shards),
                BEAM_STREAM3_RING_SLOTS='2',BEAM_STREAM4_ACTIVE_SORT_SLOTS='1',
                BEAM_STREAM4_BATCH_CANDIDATES=str(alignment*8))
            stride=world*shards*alignment
            if stride>ceiling:continue
            try:
                result=find_capacity([profile],admit=session.admit,upper_bound=ceiling,
                    alignment=stride,deadline=deadline)
            except CapacityRejected as error:
                rejections.append(dict(shards=shards,reason=str(error)));continue
            probes+=result.probes
            complete=complete and result.search_complete
            if best is None or result.effective_beam>best.effective_beam:best=result
            if not result.search_complete:break
    if best is None:raise CapacityRejected('no model/GPU frontier was admitted')
    data=asdict(best)
    data.update(probes=probes,search_complete=complete,wall_seconds=time.monotonic()-started,
        profiles='shards 1..128, two staging slots, one active sort slot',
        upper_bound_source='minimum of smallest GPU VRAM / state storage and multiset permutation bound',
        capacity_scope='native memory admission; full-step verification pending',rejected=rejections)
    (Path(run_dir)/'maximum-beam.json').write_text(json.dumps(data,indent=2))
    return data
