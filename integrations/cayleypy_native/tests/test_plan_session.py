import time
import pytest
from multigpubeamsearch import plan_session as module


def test_planning_cohort_has_shared_unique_identity_and_frozen_inference(tmp_path,monkeypatch):
    sessions=[]
    class Session:
        def __init__(self,command,environment,path,deadline):
            self.env=environment;self.request=None;self.closed=False;sessions.append(self)
        def receive(self):return dict(ready=True,rank=int(self.env['RANK']))
        def send_request(self,request):self.request=request
        def close(self):self.closed=True
    monkeypatch.setattr(module,'InferenceSession',Session)
    with module.NativePlanSession('runner',{'BEAM_ENSEMBLE_INFERENCE_MICRO':'256'},2,
        tmp_path/'a',deadline=time.monotonic()+10) as owner:
        ids={s.env['BEAM_NCCL_RUN_ID'] for s in sessions}
        assert len(ids)==1 and next(iter(ids)).startswith('plan-session-')
        with pytest.raises(ValueError,match='frozen'):
            owner.admit(8192,{'BEAM_ENSEMBLE_INFERENCE_MICRO':'512'})
        assert all(s.request is None for s in sessions)
    assert all(s.closed for s in sessions)
