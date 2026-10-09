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


@pytest.mark.parametrize('allocation_failure',[False,True])
def test_component_collective_waits_for_all_allocations(tmp_path,monkeypatch,allocation_failure):
    events=[]
    class Session:
        def __init__(self,command,environment,path,deadline):
            self.rank=int(environment['RANK']);self.request=None
        def send_request(self,request):
            self.request=request;events.append(('send',self.rank,request))
        def receive(self):
            events.append(('receive',self.rank,self.request))
            if self.request is None:return dict(ready=True,rank=self.rank,component_protocol='persistent-exact-capacity-v1')
            if self.request.get('prepare_component'):return dict(prepared=not(allocation_failure and self.rank==1),rank=self.rank)
            return dict(rank=self.rank,correctness=True)
        def close(self):pass
    monkeypatch.setattr(module,'InferenceSession',Session)
    with module.NativePlanSession('runner',{},2,tmp_path/'components',deadline=time.monotonic()+10) as owner:
        if allocation_failure:
            with pytest.raises(ValueError,match='every rank'):owner.measure_components(8192,{})
            assert not any(e[0]=='send' and e[2].get('run_component') for e in events)
            assert sum(e[0]=='send' and e[2].get('cancel_component',False) for e in events)==2
        else:
            assert len(owner.measure_components(8192,{}))==2
            first_run=next(i for i,e in enumerate(events) if e[0]=='send' and e[2].get('run_component'))
            assert sum(e[0]=='receive' and bool(e[2]) and e[2].get('prepare_component',False) for e in events[:first_run])==2
