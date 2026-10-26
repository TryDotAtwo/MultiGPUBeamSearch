"""Separate native admission, full-step evidence and measured efficiency."""
from dataclasses import replace
import json
import math
from pathlib import Path


def maximum_options(options):
    return replace(options,calibration_full_frontier=True)


def require_full_step(beam,pipeline):
    if (pipeline.get('phase')!='pipeline_measured'
        or pipeline.get('pipeline_verified') is not True
        or pipeline.get('measurement_scope')!='full_requested_frontier'
        or pipeline.get('requested_beam_effective')!=beam):
        raise ValueError('maximum frontier requires a verified full step at its exact width')
    value=pipeline.get('estimate',{}).get('median',0)
    if not math.isfinite(value) or value<=0:
        raise ValueError('maximum frontier has no valid full-step timing')
    return value


def assess_efficiency(beam,seconds_per_parent,reference):
    result=dict(tested_beam=beam,full_step_verified=True,
        full_step_seconds=seconds_per_parent*beam,
        parents_per_second=1/seconds_per_parent,
        efficiency_verified=False,efficient_working_beam=None,
        scope='largest suitable measured width; not an exhaustive efficient maximum')
    if reference is None:
        result['reason']='no comparable smaller full-step receipt'
        return result
    retention=reference['seconds_per_parent']/seconds_per_parent
    near_linear=retention>=.8
    result.update(efficiency_verified=True,reference=reference,
        throughput_retention=retention,near_linear=near_linear,
        efficient_working_beam=beam if near_linear else reference['beam'],
        efficiency_policy='retain at least 80% of smaller measured-width throughput')
    return result


def record_evidence(capacity,pipeline,inference,cache_dir):
    """Read only validated cache receipts for an identical execution cohort."""
    from .pipeline_cache import valid_receipt
    beam=capacity['effective_beam'];value=require_full_step(beam,pipeline)
    signature=inference['signature'];matches=[]
    identity_keys=('graph','models','inference_backend','runner','probe','gpu_uuids',
        'world_size','torch','cuda','precision','topology_sha256','hardware_sha256')
    for path in sorted((Path(cache_dir)/'pipeline-profiles').glob('*.json'),
                       key=lambda p:p.stat().st_mtime,reverse=True)[:256]:
        try:
            row=json.loads(path.read_text());identity=row['cache_identity'];other=identity['signature']
            if not identity.get('full_frontier') or not valid_receipt(row,identity):continue
            if identity['parent_batch']!=inference['parent_batch']:continue
            if any(k not in signature or signature[k]!=other.get(k) for k in identity_keys):continue
            width=row['requested_beam_effective']
            if not 0<width<beam:continue
            seconds=require_full_step(width,row)
            matches.append(dict(beam=width,seconds_per_parent=seconds,
                profile=row['selection'],receipt=path.name))
        except (OSError,ValueError,KeyError,TypeError):continue
    reference=max(matches,key=lambda r:r['beam']) if matches else None
    return assess_efficiency(beam,value,reference)


def measure_reference(contract,model,runtime,options,devices,runner,environment,
                      pipeline,beam,run_dir):
    """One smaller-width cohort; no profile grid or inference retuning."""
    import time
    from dataclasses import asdict,replace
    from .pipeline_probe import NativePipelineProbe
    from .pipeline_profiles import rank_frontier_counts
    from .calibration_frontier import write_frontiers
    from .calibration_stats import estimate
    from .models import verify_prepared_model
    directory=Path(run_dir)/('working-reference-'+str(beam));directory.mkdir()
    deadline=time.monotonic()+options.calibration_pipeline_seconds
    storage=runtime.build_metadata['shape']['storage_len'];world=len(devices)
    env=dict(environment,BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',
             BEAM_STREAM2_SUFFIX_RADIUS='0',BEAM_SOLVE_BUCKET_MODE='0')
    fixtures=[directory/'frontiers'/f'frontier-rank-{rank}.bin' for rank in range(world)]
    def verify(texts,plans):
        verify_prepared_model(model,contract)
        for text,count in zip(texts,rank_frontier_counts(plans)):
            fields=dict(line.split('=',1) for line in text.splitlines() if '=' in line and ' ' not in line)
            if fields.get('completed_depths')!='6' or int(fields.get('last_final_frontier_size','-1'))!=count:return False
            if 'production_runner_error=' in text:return False
        return True
    probe=NativePipelineProbe(runner,env,beam,world,directory/'probe',fixtures,
        storage,deadline=deadline,verify=verify)
    profile=dict(pipeline['environment']);plans=probe.admit(profile)
    fixture=write_frontiers(contract,rank_frontier_counts(plans),storage,
        directory/'frontiers',deadline=deadline,max_states=beam)
    samples=[replace(row,profile='working-reference') for row in probe.measure(profile,plans)]
    measured=estimate('working-reference',samples)
    result=dict(phase='pipeline_measured',pipeline_verified=True,
        measurement_scope='full_requested_frontier',requested_beam_effective=beam,
        requested_beam_width=beam,environment=profile,selection='working-reference',
        selection_objective='same-profile bounded working-frontier comparison',
        estimate=asdict(measured),measurements=[asdict(x) for x in samples],fixture=fixture,
        full_tested=['working-reference'],cache_hit=False)
    (directory/'selection.json').write_text(json.dumps(result,indent=2))
    return result


def choose_working_capacity(capacity,pipeline,inference,cache_dir,*,measure,max_rounds=3):
    """Largest tested width retaining 80% throughput; never a numeric-width path."""
    current=capacity['effective_beam'];history=[]
    for _ in range(max_rounds):
        evidence=record_evidence({'effective_beam':current},pipeline,inference,cache_dir)
        if evidence['efficiency_verified'] and evidence['near_linear']:
            return current,pipeline,dict(evidence,comparisons=history)
        width=current//2
        if width<inference['parent_batch']*inference['signature']['world_size']:
            raise ValueError('working boundary has no smaller saturated comparison')
        reference=measure(width,pipeline)
        lower=require_full_step(width,reference)
        evidence=assess_efficiency(current,require_full_step(current,pipeline),
            dict(beam=width,seconds_per_parent=lower,profile=reference['selection'],
                 receipt='working-reference-'+str(width)))
        history.append(evidence)
        if evidence['near_linear']:
            return current,pipeline,dict(evidence,comparisons=history.copy())
        current,pipeline=width,reference
    raise ValueError('bounded working-frontier comparison did not find stable scaling')
