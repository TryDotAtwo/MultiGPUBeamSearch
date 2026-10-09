"""Replay must honor caller's absolute monotonic deadline."""
import time
import pytest
from tools.cayleypy_public.paths import apply_path


def test_expired_replay_deadline_rejects_even_empty_path():
    with pytest.raises(TimeoutError, match='replay deadline'):
        apply_path([1, 0], '', {'swap': [1, 0]}, deadline=time.monotonic() - 1)


def test_replay_deadline_retains_move_orientation():
    assert apply_path([1, 0], 'swap', {'swap': [1, 0]},
                      deadline=time.monotonic() + 10) == (0, 1)


def test_replay_checks_deadline_between_moves(monkeypatch):
    from tools.cayleypy_public import paths
    ticks = iter([0., 0., 0., 2.])
    monkeypatch.setattr(paths, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError, match='replay deadline'):
        apply_path([1, 0], 'swap.swap.swap', {'swap': [1, 0]}, deadline=1.)
