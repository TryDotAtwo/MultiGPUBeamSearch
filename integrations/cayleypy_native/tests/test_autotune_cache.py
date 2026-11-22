from copy import deepcopy
from multigpubeamsearch.autotune import valid_cached_profile


def profile():
    signature={'world_size':2}
    rows=[{'batch':1024,'device':i,'parents':8192,'seconds':[1.0]*5,
           'correctness_passed':True,'numeric_error':0,'torch_reserved_peak_bytes':1234}
          for i in range(2)]
    return {'signature':signature,'phase':'inference_verified','parent_batch':1024,
            'reserve_bytes':1234+(512<<20),'records':rows},signature


def test_cache_requires_all_rank_evidence():
    data,signature=profile()
    assert valid_cached_profile(data,signature,8192)
    for broken in ({},dict(data,records=data['records'][:1]),
                   dict(data,reserve_bytes=1234),dict(data,parent_batch=True)):
        assert not valid_cached_profile(broken,signature,8192)


def test_cache_rejects_failed_or_nonfinite_measurements():
    original,signature=profile()
    for field,value in [('device',0),('seconds',[float('nan')]*5),
                        ('numeric_error',1),('correctness_passed',False),('parents',1)]:
        data=deepcopy(original);data['records'][1][field]=value
        assert not valid_cached_profile(data,signature,8192)


def test_new_cache_cannot_drop_or_override_observed_throttling():
    data, signature = profile()
    signature['schema'] = 3
    assert not valid_cached_profile(data, signature, 8192)
    data['gpu_telemetry'] = {'1024': {'throttled': False,
        'samples': [[{'throttled': False}]]}}
    assert valid_cached_profile(data, signature, 8192)
    data['gpu_telemetry']['1024']['samples'][0][0]['throttled'] = True
    assert not valid_cached_profile(data, signature, 8192)
