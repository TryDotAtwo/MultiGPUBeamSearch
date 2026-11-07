import pytest
from tools.inference_first_calibration import Measurement, estimate, select


def rows(name, timings, **kwargs):
    return [Measurement(name, 1000, (t, t*0.9), True, True, **kwargs) for t in timings]


def test_slowest_rank_and_clear_gain():
    data = rows('baseline', [10]*5) + rows('fast', [7]*5)
    chosen, _ = select(data, baseline='baseline')
    assert chosen.profile == 'fast'
    assert chosen.median == .007


def test_noise_does_not_replace_baseline():
    data = rows('baseline', [10]*5) + rows('noise', [9.5, 10, 10, 10.5, 10])
    assert select(data, baseline='baseline')[0].profile == 'baseline'


def test_throttling_rejected():
    data = rows('baseline', [10]*5) + rows('fast', [1]*5, throttled=True)
    chosen, rejected = select(data, baseline='baseline')
    assert chosen.profile == 'baseline' and 'fast' in rejected


def test_mixed_cohort_rejected():
    data = rows('baseline', [10]*5)
    data.append(Measurement('other', 2000, (10, 10), True, True))
    with pytest.raises(ValueError, match='cohort'):
        select(data, baseline='baseline')


def test_bad_rank_timing_rejected():
    data = rows('baseline', [10]*4)
    data.append(Measurement('baseline', 1000, (float('nan'), 10), True, True))
    with pytest.raises(ValueError, match='finite'):
        estimate('baseline', data)
