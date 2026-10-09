"""Bounded inference-first calibration; downstream tuning uses the frozen winner."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

from .calibration_stats import Measurement, estimate, select
from .errors import NativeBackendError
from .models import verify_prepared_model
from .build import file_sha256
from .calibration_telemetry import CalibrationTelemetry, verified_telemetry


def coarse_batches(cap):
    """Include the real beam/user cap, even when it is not a power of two."""
    if type(cap) is not int or not 1 <= cap <= 65536:
        raise ValueError('inference cap must be in [1,65536]')
    baseline=min(256,cap)
    values=[x for x in (32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536)
            if x<=cap and x!=baseline]
    if cap!=baseline and cap not in values:values.append(cap)
    # Throughput candidates precede expensive tiny capacity-floor probes.
    larger=[x for x in values if x>=baseline and x!=cap]
    smaller=[x for x in values if x<baseline]
    random.Random(20261009).shuffle(larger)
    random.Random(20261009).shuffle(smaller)
    return [baseline]+([cap] if cap!=baseline else [])+larger+smaller


def probe_time_allowance(records, default=12.0):
    """Measured repeats plus warmup/oracle and control overhead."""
    estimates=[10*max(row['seconds'])+3 for row in records if row.get('seconds')]
    return max(default,max(estimates,default=default))


def select_inference(samples, *, baseline):
    """A noisy reference cannot disqualify another complete, stable cohort."""
    rejected={}
    try:
        estimate(baseline,samples)
    except ValueError as error:
        rejected[baseline]=str(error)
        valid=[]
        for profile in sorted({row.profile for row in samples}-{baseline}):
            try:valid.append(estimate(profile,samples))
            except ValueError as candidate_error:rejected[profile]=str(candidate_error)
        if not valid:raise NativeBackendError('no stable complete inference candidate') from error
        baseline=min(valid,key=lambda row:(row.median,row.profile)).profile
    winner,other_rejected=select(samples,baseline=baseline)
    return winner,dict(rejected,**other_rejected)


def refinement_batches(winner,cap,visited):
    """Bounded local search; performance is measured rather than inferred."""
    if type(winner) is not int or not 1 <= winner <= cap:
        raise ValueError('inference winner must fit its cap')
    grain=max(1,min(32,winner//8))
    values=set()
    for numerator,denominator in ((5,8),(3,4),(7,8),(9,8),(5,4),(3,2)):
        batch=((winner*numerator+denominator*grain//2)//(denominator*grain))*grain
        if 1<=batch<=cap and batch not in visited:values.add(batch)
    result=sorted(values)
    random.Random(20261010).shuffle(result)
    return result


def graph_samples(contract, count=128):
    rng=random.Random(20261009)
    state=contract.start
    result=[]
    for _ in range(count):
        for _ in range(16):
            move=contract.generators[rng.randrange(contract.move_count)]
            state=tuple(state[i] for i in move)
        result.append(list(state))
    return result


def calibration_signature(contract,model,runtime,devices,beam_width):
    import torch
    try:
        topology=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True,timeout=10).stdout
    except (OSError,subprocess.TimeoutExpired):topology='unavailable'
    try:
        hardware=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,power.limit',
            '--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True).stdout
    except (OSError,subprocess.SubprocessError):hardware='unavailable'
    import platform
    return {'schema':8,'search_policy':'upper-first-stable-reference-v3','graph':contract.graph_hash,'models':model.artifact_hash,
            'runner':runtime.build_metadata['binary_sha256'],
            'probe':runtime.build_metadata.get('calibration_binary_sha256'),
            'gpu_properties':[str(torch.cuda.get_device_properties(d)) for d in devices],
            'device_indices':list(devices),'world_size':len(devices),'torch':torch.__version__,
            'cuda':torch.version.cuda,'topology_sha256':hashlib.sha256(topology.encode()).hexdigest(),
            'hardware_sha256':hashlib.sha256(hardware.encode()).hexdigest(),
            'gpu_uuids':[str(getattr(torch.cuda.get_device_properties(d),'uuid','unavailable')) for d in devices],
            'host':{'machine':platform.machine(),'processor':platform.processor()},
            'requested_beam_width':beam_width,'precision':'fp16/fp32'}


def tune_inference(contract,model,runtime,options,devices,beam_width,run_dir,environment):
    """Use the actual ensemble executable, all selected GPUs, and graph states.

    Cache hits retain complete source-bound measurement evidence. The native
    planner still admits every downstream configuration against current free
    memory; a calibration record never bypasses that admission.
    """
    from .backend import _stop_process_tree
    helper=runtime.runner.parent/runtime.build_metadata.get('calibration_binary_name','stream1_ensemble_benchmark')
    if not helper.is_file() or file_sha256(helper)!=runtime.build_metadata.get('calibration_binary_sha256'):
        raise NativeBackendError('verified Stream1 calibration executable is required for autotuning')
    verify_prepared_model(model,contract)
    signature=calibration_signature(contract,model,runtime,devices,beam_width)
    signature['max_batch']=options.calibration_max_batch
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    cache=options.cache_dir/'profiles'/f'{key}.json'
    if cache.is_file():
        try:
            data=json.loads(cache.read_text())
            if valid_cached_profile(data,signature,options.calibration_max_batch):
                return dict(data,cache_hit=True)
        except (OSError,ValueError):pass
    directory=Path(run_dir)/'calibration';directory.mkdir(exist_ok=False)
    probe_dir=directory/'inputs';probe_dir.mkdir()
    if model.backend=='ensemble':
        manifest=dict(model.manifest['ensemble'])
        manifest['models']=[dict(entry,weights_dir=str(model.weights_dir/entry['weights_dir'])) for entry in manifest['models']]
    else:
        manifest={'weights_dir':str(model.weights_dir)}
    manifest['calibration_states']=graph_samples(contract)
    manifest['generators']=[list(x) for x in contract.generators]
    (probe_dir/('ensemble.json' if model.backend=='ensemble' else 'calibration.json')).write_text(json.dumps(manifest))
    cap=min(options.calibration_max_batch,max(1,(beam_width+len(devices)-1)//len(devices)))
    candidates=coarse_batches(cap)
    baseline=candidates[0]
    coarse_candidates=list(candidates)
    refined=[]
    parents=min((beam_width+len(devices)-1)//len(devices),max(8192,cap))
    started=time.monotonic()
    deadline=started+options.calibration_seconds
    coarse_deadline=started+.7*options.calibration_seconds
    refinement_queued=False
    samples=[];records=[];rejected={};reserves={};telemetry_records={}
    from .calibration_session import EnsembleProbePool
    pool=EnsembleProbePool(helper,probe_dir,cap,parents,len(devices),environment,directory,deadline) if (
        runtime.build_metadata.get('calibration_protocol')=='json-session-v1') else None
    try:
        for index,batch in enumerate(candidates):
            if time.monotonic()>=deadline:break
            if index>0 and deadline-time.monotonic()<probe_time_allowance(records):break
            if index<len(coarse_candidates) and index>0 and time.monotonic()>=coarse_deadline:
                if not refinement_queued:
                    coarse_winner,_=select_inference(samples,baseline=str(baseline))
                    refined=refinement_batches(int(coarse_winner.profile),cap,set(candidates))
                    candidates.extend(refined)
                    refinement_queued=True
                continue
            # A phase boundary limits starts, never an in-flight healthy rank.
            candidate_deadline=deadline
            processes=[];rows=[]
            telemetry=CalibrationTelemetry()
            telemetry.__enter__()
            try:
                if pool is not None:
                    rows,failures=pool.measure(batch,parents,candidate_deadline)
                    if failures:rejected[str(batch)]='; '.join(failures)
                else:
                    for rank in range(len(devices)):
                        path=directory/f'batch-{batch}-rank-{rank}.log'
                        log=path.open('wb')
                        command=[str(helper),str(probe_dir),str(batch),str(parents),str(rank)]
                        try:
                            process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,start_new_session=True)
                        except Exception:
                            log.close()
                            raise
                        processes.append((rank,process,log,path))
                    for rank,process,log,path in processes:
                        try:code=process.wait(timeout=max(.1,candidate_deadline-time.monotonic()))
                        except subprocess.TimeoutExpired:_stop_process_tree(process);code=-1
                        log.close()
                        parsed=[]
                        for line in path.read_text(errors='replace').splitlines():
                            try:
                                row=json.loads(line)
                                if 'seconds' in row:parsed.append(row)
                            except ValueError:pass
                        if (code or len(parsed)!=1 or not parsed[0].get('correctness_passed')
                                or parsed[0].get('numeric_error') or len(parsed[0]['seconds'])<5):
                            rejected[str(batch)]=f'rank {rank} failed admission or timed out';continue
                        rows.append(parsed[0])
            finally:
                for _,process,log,_ in processes:
                    if process.poll() is None:_stop_process_tree(process)
                    log.close()
                telemetry.__exit__()
                telemetry_records[str(batch)]=telemetry.receipt()
            (directory/f'batch-{batch}-receipt.json').write_text(json.dumps({
                'signature':signature,'batch':batch,'rank_measurements':rows,
                'gpu_telemetry':telemetry_records[str(batch)],
                'failure':rejected.get(str(batch))},indent=2))
            records.extend(rows)
            if len(rows)==len(devices):
                telemetry_verified=verified_telemetry(telemetry_records[str(batch)],signature)
                if not telemetry_verified:
                    rejected[str(batch)]='GPU telemetry missing, incomplete or throttled'
                for repeat in range(min(len(row['seconds']) for row in rows)):
                    samples.append(Measurement(str(batch),parents*len(devices),
                        tuple(row['seconds'][repeat] for row in rows),True,True,
                        throttled=not telemetry_verified))
                reserves[str(batch)]=max(row['torch_reserved_peak_bytes'] for row in rows)+(512<<20)
            elif batch==baseline:
                raise NativeBackendError('baseline inference calibration failed; see '+str(directory))
            if index+1==len(coarse_candidates) and not refinement_queued and time.monotonic()<deadline:
                coarse_winner,_=select_inference(samples,baseline=str(baseline))
                refined=refinement_batches(int(coarse_winner.profile),cap,set(candidates))
                candidates.extend(refined)
                refinement_queued=True
    finally:
        if pool is not None:pool.close()
    winner,stat_rejected=select_inference(samples,baseline=str(baseline))
    verify_prepared_model(model,contract)
    data={'signature':signature,'phase':'inference_verified','parent_batch':int(winner.profile),
          'reserve_bytes':reserves[winner.profile],'estimate':winner.__dict__,
          'records':records,'rejected':rejected,'stat_rejected':stat_rejected,
          'gpu_telemetry':telemetry_records,
          'coarse_candidates':coarse_candidates,'refinement_candidates':refined,
          'calibration_wall_seconds':time.monotonic()-started,
          'probe_process_starts':pool.starts if pool is not None else len(records),
          'probe_protocol':runtime.build_metadata.get('calibration_protocol','oneshot'),
          'pipeline_verified':False,'cache_hit':False,'measured_candidates':sorted({int(s.profile) for s in samples})}
    (directory/'inference-selection.json').write_text(json.dumps(data,indent=2))
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_name(cache.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(data,indent=2));temporary.replace(cache)
    return data


def valid_cached_profile(data, signature, max_batch):
    """A batch value without all-rank measurement evidence is not a profile."""
    try:
        batch=data['parent_batch']
        if (data.get('signature')!=signature or data.get('phase')!='inference_verified'
                or type(batch) is not int or not 0<batch<=max_batch
                or type(data.get('reserve_bytes')) is not int or data['reserve_bytes']<=0):
            return False
        rows=[row for row in data['records'] if row['batch']==batch]
        if signature.get('schema', 0)>=3:
            telemetry=data['gpu_telemetry'][str(batch)]
            if signature.get('schema',0)>=6 and not verified_telemetry(telemetry,signature):
                return False
            if telemetry.get('throttled') is not False:
                return False
            if any(row.get('throttled') is not False
                   for sample in telemetry['samples'] for row in sample):
                return False
        world=signature['world_size']
        if len(rows)!=world or {row['device'] for row in rows}!=set(range(world)):
            return False
        parents={row['parents'] for row in rows}
        if len(parents)!=1 or any(type(n) is not int or n<=0 for n in parents):return False
        for row in rows:
            if row.get('correctness_passed') is not True or row.get('numeric_error')!=0:
                return False
            if len(row['seconds'])<5 or any(not math.isfinite(x) or x<=0 for x in row['seconds']):
                return False
        expected_reserve=max(row['torch_reserved_peak_bytes'] for row in rows)+(512<<20)
        return data['reserve_bytes']==expected_reserve
    except (KeyError,TypeError,ValueError):
        return False
