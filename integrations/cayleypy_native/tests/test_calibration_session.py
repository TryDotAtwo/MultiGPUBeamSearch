from multigpubeamsearch import calibration_session as module


def test_all_ranks_start_before_readiness_and_requests(monkeypatch, tmp_path):
    events = []

    class Session:
        def __init__(self, command, *args, **kwargs):
            self.rank = int(command[4])
            self.ready = False
            events.append(('start', self.rank))

        def receive(self, deadline):
            if not self.ready:
                assert len([e for e in events if e[0] == 'start']) == 8
                self.ready = True
                events.append(('ready', self.rank))
                return {'ready': True, 'device': self.rank}
            return {'batch': 256, 'parents': 8192, 'device': self.rank,
                    'correctness_passed': True, 'numeric_error': 0, 'seconds': [1.] * 7}

        def send(self, batch, parents):
            assert len([e for e in events if e[0] == 'ready']) == 8
            events.append(('send', self.rank))

        def close(self):
            events.append(('close', self.rank))

    monkeypatch.setattr(module, 'InferenceSession', Session)
    pool = module.EnsembleProbePool('helper', 'inputs', 8192, 8192, 8, {}, tmp_path, 999)
    rows, failures = pool.measure(256, 8192, 999)
    assert len(rows) == 8 and not failures and pool.starts == 8
    pool.close()


def test_readiness_failure_closes_every_started_rank(monkeypatch, tmp_path):
    import pytest
    closed = []

    class Session:
        def __init__(self, command, *args, **kwargs):
            self.rank = int(command[4])

        def receive(self, deadline):
            return {'ready': False, 'device': self.rank}

        def send(self, *args):
            pytest.fail('request sent before readiness')

        def close(self):
            closed.append(self.rank)

    monkeypatch.setattr(module, 'InferenceSession', Session)
    pool = module.EnsembleProbePool('helper', 'inputs', 8192, 8192, 8, {}, tmp_path, 999)
    with pytest.raises(RuntimeError, match='readiness'):
        pool.measure(256, 8192, 999)
    assert sorted(closed) == list(range(8)) and not pool.sessions

