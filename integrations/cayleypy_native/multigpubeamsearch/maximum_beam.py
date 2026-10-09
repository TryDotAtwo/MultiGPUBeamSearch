"""Maximum aligned frontier admission for the current model and GPU cohort."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import math
from collections import Counter
from .beam_capacity import CapacityRejected,find_capacity
from .plan_session import NativePlanSession


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
