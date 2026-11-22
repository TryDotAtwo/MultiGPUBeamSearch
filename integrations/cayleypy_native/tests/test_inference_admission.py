import pytest
from multigpubeamsearch.inference_admission import select_admitted, measured_candidates
from multigpubeamsearch.beam_capacity import CapacityRejected
from multigpubeamsearch.errors import NativeBackendError


def fixture():
    records=[]
    for batch,seconds,reserve in [(8192,.531,8<<30),(2048,.534,2<<30),(256,.58,400<<20)]:
        for rank in range(2):
            records.append(dict(batch=batch,device=rank,parents=8192,
                seconds=[seconds]*7,torch_reserved_peak_bytes=reserve,
                correctness_passed=True,numeric_error=0))
    return dict(signature={"world_size":2},parent_batch=8192,
        estimate={"median":.531/16384},records=records)


def test_memory_fallback_keeps_frontier_and_uses_own_reserve():
    calibration=fixture();calls=[]
    def admit(candidate):
        calls.append((candidate["parent_batch"],candidate["reserve_bytes"],1_048_576))
        if candidate["parent_batch"]==8192:
            raise CapacityRejected("insufficient pipeline VRAM")
        return [{"GLOBAL_BEAM_WIDTH_EFFECTIVE":1_048_576}]*2
    chosen,plans=select_admitted(calibration,admit)
    assert chosen["parent_batch"]==2048
    assert chosen["reserve_bytes"]==(2<<30)+(512<<20)
    assert calls==[(8192,(8<<30)+(512<<20),1_048_576),
                   (2048,(2<<30)+(512<<20),1_048_576)]
    assert calibration["parent_batch"]==8192
    assert plans[0]["GLOBAL_BEAM_WIDTH_EFFECTIVE"]==1_048_576
    assert chosen["inference_admission"]["unconstrained_batch"]==8192


def test_skip_incomplete_or_throttled_candidate():
    data=fixture()
    data["records"]=[x for x in data["records"] if not (x["batch"]==2048 and x["device"]==1)]
    data["signature"]["schema"]=5
    data["gpu_telemetry"]={str(b):{"throttled":b==8192} for b in (8192,2048,256)}
    assert [x["parent_batch"] for x in measured_candidates(data)]==[256]


def test_all_rejected_is_error_without_smaller_beam():
    data=fixture()
    def reject(candidate):
        raise CapacityRejected("not enough memory")
    with pytest.raises(NativeBackendError,match="requested frontier"):
        select_admitted(data,reject)


def test_runtime_failure_is_not_hidden_as_capacity_rejection():
    def fail(candidate):
        raise RuntimeError("communicator failed")
    with pytest.raises(RuntimeError,match="communicator"):
        select_admitted(fixture(),fail)
