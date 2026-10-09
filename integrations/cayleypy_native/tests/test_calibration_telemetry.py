import pytest
from multigpubeamsearch.calibration_telemetry import parse_telemetry, CalibrationTelemetry


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

