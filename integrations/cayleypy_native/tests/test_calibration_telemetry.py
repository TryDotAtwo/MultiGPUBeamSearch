import pytest
from multigpubeamsearch.calibration_telemetry import parse_telemetry, CalibrationTelemetry
from multigpubeamsearch.calibration_telemetry import verified_telemetry


def test_thermal_and_external_power_events_are_rejected():
    rows = parse_telemetry('0, GPU-a, 72, Not Active, Active, Not Active\n'
                           '1, GPU-b, 65, Not Active, Not Active, Not Active')
    assert rows[0]['throttled'] and not rows[1]['throttled']
    monitor = CalibrationTelemetry()
    monitor.samples = [rows]
    assert monitor.receipt()['throttled']


@pytest.mark.parametrize('text', ['', '0,GPU-a,72,N/A,Not Active,Not Active',
    '0,GPU-a,72,Not Active,Not Active,Not Active\n0,GPU-a,72,Not Active,Not Active,Not Active'])
def test_unavailable_telemetry_is_never_reported_as_verified(text):
    with pytest.raises(ValueError):
        parse_telemetry(text)


def test_verification_requires_two_complete_selected_gpu_observations():
    from copy import deepcopy
    signature={'world_size':2,'device_indices':[0,1],'gpu_uuids':['GPU-a','GPU-b']}
    sample=[{'index':0,'uuid':'GPU-a','throttled':False},
            {'index':1,'uuid':'GPU-b','throttled':False}]
    receipt={'samples':[sample,deepcopy(sample)],'errors':[],'throttled':False}
    assert verified_telemetry(receipt,signature)
    for samples in ([],[sample],[sample,sample[:1]],
                    [sample,[dict(sample[0],uuid='GPU-other'),sample[1]]]):
        assert not verified_telemetry(dict(receipt,samples=samples),signature)
    assert not verified_telemetry(dict(receipt,errors=['TimeoutExpired']),signature)
    assert not verified_telemetry(dict(receipt,throttled=True),signature)
