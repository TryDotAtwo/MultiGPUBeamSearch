"""Keep the requested frontier and admit the fastest measured inference batch."""
from __future__ import annotations
from collections import defaultdict
import time
from pathlib import Path
from .calibration_stats import Measurement, estimate
from .beam_capacity import CapacityRejected
from .errors import NativeBackendError


def measured_candidates(calibration):
    """Discard incomplete cohorts, unsafe readouts and unstable measurements."""
    world=calibration["signature"]["world_size"]
    groups=defaultdict(list)
    for row in calibration["records"]:
        groups[row["batch"]].append(row)
    candidates=[]
    for batch,rows in groups.items():
        if type(batch) is not int or not 0<batch<=calibration['signature'].get('max_batch',65536):
            continue
        if len(rows)!=world or {r["device"] for r in rows}!=set(range(world)):
            continue
        if any(r.get("correctness_passed") is not True or r.get("numeric_error")!=0 for r in rows):
            continue
        telemetry=calibration.get("gpu_telemetry",{}).get(str(batch))
        if calibration["signature"].get("schema",0)>=3 and (
            not telemetry or telemetry.get("throttled") is not False):
            continue
        if len({r["parents"] for r in rows})!=1:
            continue
        if any(type(r['parents']) is not int or r['parents']<=0 or
               type(r.get('torch_reserved_peak_bytes')) is not int or
               r['torch_reserved_peak_bytes']<0 for r in rows):
            continue
        repeats=min(len(r["seconds"]) for r in rows)
        samples=[Measurement(str(batch),rows[0]["parents"]*world,
            tuple(r["seconds"][i] for r in rows),True,True) for i in range(repeats)]
        try:
            value=estimate(str(batch),samples)
        except (ValueError,TypeError,KeyError):
            continue
        reserve=max(r["torch_reserved_peak_bytes"] for r in rows)+(512<<20)
        candidates.append(dict(parent_batch=batch,reserve_bytes=reserve,estimate=value.__dict__))
    candidates.sort(key=lambda x:(x["parent_batch"]!=calibration["parent_batch"],
                                  x["estimate"]["median"],x["reserve_bytes"]))
    return candidates


def select_admitted(calibration, admit):
    """A rejection changes the measured batch, never the requested beam."""
    attempts=[]
    for candidate in measured_candidates(calibration):
        try:
            plans=admit(candidate)
        except CapacityRejected as error:
            attempts.append(dict(candidate,admitted=False,reason=str(error)))
            continue
        attempts.append(dict(candidate,admitted=True))
        result=dict(calibration,**candidate,
            inference_admission=dict(unconstrained_batch=calibration["parent_batch"],
                unconstrained_estimate=calibration["estimate"],attempts=attempts))
        return result,plans
    raise NativeBackendError("no measured inference batch admits the requested frontier")


def admit_inference(contract,model,runtime,options,devices,beam,run_dir,environment,runner,calibration):
    """Fresh native memory snapshots use each batch's own calibrated reserve."""
    from .plan_session import NativePlanSession
    from .models import verify_prepared_model
    verify_prepared_model(model,contract)
    root=Path(run_dir)/"inference-admission"
    root.mkdir()
    deadline=time.monotonic()+min(60.0,options.calibration_pipeline_seconds*.25)
    multiplier=contract.move_count if model.backend=="mlp" and model.manifest["output_dim"]==1 else 1
    def admit(candidate):
        env=dict(environment,BEAM_B_MICRO=str(candidate["parent_batch"]*multiplier),
            BEAM_ENSEMBLE_INFERENCE_MICRO=str(candidate["parent_batch"]),
            BEAM_ENSEMBLE_RESERVE_BYTES=str(candidate["reserve_bytes"]))
        with NativePlanSession(runner,env,len(devices),root/str(candidate["parent_batch"]),
                               deadline=deadline) as session:
            return session.admit(beam,{"BEAM_B_MICRO":env["BEAM_B_MICRO"],
                "BEAM_ENSEMBLE_INFERENCE_MICRO":env["BEAM_ENSEMBLE_INFERENCE_MICRO"]})
    chosen,plans=select_admitted(calibration,admit)
    verify_prepared_model(model,contract)
    chosen["inference_admission"]["requested_beam"]=beam
    chosen["inference_admission"]["plans"]=plans
    calibration.clear();calibration.update(chosen)
    environment.update(BEAM_B_MICRO=str(chosen["parent_batch"]*multiplier),
        BEAM_ENSEMBLE_INFERENCE_MICRO=str(chosen["parent_batch"]),
        BEAM_ENSEMBLE_RESERVE_BYTES=str(chosen["reserve_bytes"]))
    if options.report_calibration and chosen['parent_batch']!=chosen['inference_admission']['unconstrained_batch']:
        original=chosen['inference_admission']
        print(f"[MultiGPUBeamSearch] fastest isolated batch {original['unconstrained_batch']} "
              f"does not admit frontier {beam:,}; fastest measured admitted batch "
              f"{chosen['parent_batch']}, {1/chosen['estimate']['median']:,.0f} parents/s; "
              "frontier preserved",flush=True)
    import json
    (root/"selection.json").write_text(json.dumps(chosen["inference_admission"],indent=2))
    return calibration
