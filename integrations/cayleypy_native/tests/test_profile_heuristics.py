from multigpubeamsearch.profile_heuristics import narrow, full_challengers
from multigpubeamsearch.profile_heuristics import job_counts


BASE = dict(BEAM_B_MICRO='256', BEAM_ENSEMBLE_INFERENCE_MICRO='256',
            BEAM_STREAM4_BATCH_CANDIDATES='8192')


def test_low_pressure_avoids_tiny_flush_and_keeps_interference_probes():
    rows = [(name, dict(BASE, **changes)) for name, changes in (
        ('service-batch', {'BEAM_STREAM4_BATCH_CANDIDATES':'1024'}),
        ('shards-1', {}), ('lanes-1', {}), ('staging-2', {}),
        ('outer-512', {'BEAM_B_MICRO':'512'}),
        ('bulk', {'BEAM_STREAM4_BATCH_CANDIDATES':'16384'}))]
    selected, trace = narrow(rows, BASE, dict(inference_seconds=10,
        downstream_service_work_seconds=.05, stream4_seconds=.03,
        stream3_seconds=.01, transport_seconds=.01, union_seconds=0))
    assert [name for name, _ in selected] == ['outer-512','staging-2','lanes-1','bulk']
    assert trace['regime']=='inference_dominated'
    assert full_challengers(trace)==1
    assert all(env['BEAM_ENSEMBLE_INFERENCE_MICRO']=='256' for _,env in selected)


def test_sort_pressure_prioritizes_bulk_and_shards():
    rows=[(name,dict(BASE)) for name in ['outer-512','lanes-1','shards-8','staging-2','bulk']]
    selected,trace=narrow(rows,BASE,dict(inference_seconds=1,
        downstream_service_work_seconds=2,stream4_seconds=1.8,
        stream3_seconds=.1,transport_seconds=.1,union_seconds=0))
    assert [x[0] for x in selected][:3]==['bulk','shards-8','lanes-1']
    assert len(selected)==4 and full_challengers(trace)==2


def test_missing_or_nonfinite_evidence_never_claims_low_pressure():
    for estimate in ({},dict(inference_seconds=float('nan'),downstream_service_work_seconds=.1)):
        _,trace=narrow([('lanes-1',dict(BASE))],BASE,estimate)
        assert trace['regime']=='unknown' and full_challengers(trace)==2


def test_jobs_distinguish_outer_dispatch_from_inference_and_sort_work():
    a=job_counts(65536,30,128,128,8,4,16384)
    b=job_counts(65536,30,256,128,4,4,16384)
    tiny=job_counts(65536,30,128,128,8,4,1024)
    assert a['outer_jobs']==512 and b['outer_jobs']==256
    assert a['model_forwards_per_model']==b['model_forwards_per_model']==512
    assert a['stream3_jobs']==b['stream3_jobs']==64
    assert tiny['gross_sort_jobs']==a['gross_sort_jobs']*16
    assert job_counts(257,3,256,128,2,1,1024)['model_forwards_per_model']==3
