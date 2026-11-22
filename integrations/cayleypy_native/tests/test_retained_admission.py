from types import SimpleNamespace

import pytest

from multigpubeamsearch import inference_admission as module
from multigpubeamsearch import plan_session
from multigpubeamsearch.beam_capacity import CapacityRejected


@pytest.mark.parametrize('failure_after_admission', [False, True])
def test_rejected_owner_closed_winner_transferred_only_after_verification(tmp_path, monkeypatch,
        failure_after_admission):
    owners=[]
    class Owner:
        def __init__(self,runner,env,world,path,deadline):
            self.batch=int(env['BEAM_B_MICRO']);self.closed=False;owners.append(self)
        def admit(self,beam,env):
            assert beam == 100000000
            if self.batch==8192:raise CapacityRejected('memory')
            return [{'GLOBAL_BEAM_WIDTH_EFFECTIVE':beam}]
        def close(self):self.closed=True
    monkeypatch.setattr(plan_session,'NativePlanSession',Owner)
    monkeypatch.setattr(module,'measured_candidates',lambda cal:[
        dict(parent_batch=b,reserve_bytes=b*1024,estimate=dict(median=1.)) for b in (8192,2048)])
    import multigpubeamsearch.models as models
    calls=[]
    def verify(*args):
        calls.append(1)
        if failure_after_admission and len(calls)==2:raise ValueError('model changed')
    monkeypatch.setattr(models,'verify_prepared_model',verify)
    args=(SimpleNamespace(move_count=24),SimpleNamespace(backend='ensemble'),None,
          SimpleNamespace(calibration_pipeline_seconds=90,report_calibration=False),
          (0,1),100000000,tmp_path,{},'runner',dict(parent_batch=8192,estimate=dict(median=1.)))
    if failure_after_admission:
        with pytest.raises(ValueError,match='model changed'):
            module.admit_inference(*args,retain_session=True)
        assert all(owner.closed for owner in owners)
    else:
        calibration,owner=module.admit_inference(*args,retain_session=True)
        assert calibration['parent_batch']==2048 and owner is owners[1]
        assert owners[0].closed and not owner.closed
        owner.close()


def test_external_planner_reused_without_constructing_new_cohort(monkeypatch):
    from multigpubeamsearch import pipeline_autotune
    owner=object();calls=[]
    monkeypatch.setattr(pipeline_autotune,'_tune_downstream',
        lambda *args:calls.append(args[-1]) or {'phase':'test-only'})
    assert pipeline_autotune.tune_downstream(*([None]*10),planning_session=owner)['phase']=='test-only'
    assert calls == [owner]
