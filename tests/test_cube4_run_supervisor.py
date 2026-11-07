"""Execute supervisor against real bounded child processes, never mock waits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX process-group acceptance')


@pytest.mark.parametrize('raw', ['[]', '{"puzzle_id":7,"puzzle_id":8}'])
def test_malformed_contract_fails_with_terminal_summary_before_child(tmp_path, raw):
    contract = tmp_path / 'contract.json'
    contract.write_text(raw)
    result, summary = launch(tmp_path, 'open("started", "w").write("bad")',
        extra=('--world-size', '2', '--puzzle-contract', str(contract)))
    assert result.returncode != 0
    assert summary['status'] == 'failed'
    assert not (tmp_path / 'run/started').exists()


def test_oversized_submission_cannot_enter_replay(tmp_path):
    from tools.cube4_run_supervisor import verify_rank_result
    directory = tmp_path
    (directory / 'rank-status').mkdir()
    (directory / 'rank-status/rank-0.json').write_text(json.dumps(
        {'rank': 0, 'puzzle_id': 7, 'exit_code': 0, 'status': 'solved'}))
    # A valid empty solution with excessive blank CSV records was accepted.
    (directory / 'submit.csv').write_text('initial_state_id,path\n7,\n' + '\n' * 1048576)
    contract = {'puzzle_id': 7, 'initial_state': [0, 1], 'central_state': [0, 1],
                'generators': {'swap': [1, 0]}}
    with pytest.raises(ValueError, match='submission exceeds'):
        verify_rank_result(directory, 1, contract)


@pytest.mark.parametrize('completed', [None, 7, True])
def test_unsolved_requires_requested_depth_completed(tmp_path, completed):
    from tools.cube4_run_supervisor import verify_rank_result
    (tmp_path / 'rank-status').mkdir()
    report = {'rank': 0, 'puzzle_id': 7, 'exit_code': 0, 'status': 'unsolved'}
    if completed is not None:
        report['completed_depths'] = completed
    (tmp_path / 'rank-status/rank-0.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='requested depth'):
        verify_rank_result(tmp_path, 1, {'puzzle_id': 7, 'requested_depth': 8})


def test_unsolved_at_requested_depth_is_not_a_solution(tmp_path):
    from tools.cube4_run_supervisor import verify_rank_result
    (tmp_path / 'rank-status').mkdir()
    (tmp_path / 'rank-status/rank-0.json').write_text(json.dumps(
        {'rank': 0, 'puzzle_id': 7, 'exit_code': 0, 'status': 'unsolved', 'completed_depths': 8}))
    result = verify_rank_result(tmp_path, 1, {'puzzle_id': 7, 'requested_depth': 8})
    assert result['status'] == 'unsolved'
    assert 'solution_replay_valid' not in result


def launch(tmp_path, payload, elapsed='5', cost='1', rate='1', extra=()):
    directory = tmp_path / 'run'
    result = subprocess.run([sys.executable, str(ROOT / 'tools/cube4_run_supervisor.py'),
        '--run-dir', str(directory), '--max-elapsed-seconds', elapsed,
        '--max-cost-usd', cost, '--hourly-rate-usd', rate, *extra, '--',
        sys.executable, '-c', payload], capture_output=True, text=True, timeout=15)
    assert (directory / 'summary.json').is_file(), result.stderr
    return result, json.loads((directory / 'summary.json').read_text())


@pytest.mark.parametrize('bad_rank,bad_path', [(False, False), (True, False), (False, True)])
def test_rank_status_and_independent_replay_gate(tmp_path, bad_rank, bad_path):
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'puzzle_id': 7, 'initial_state': [1, 0],
        'central_state': [0, 1], 'generators': {'swap': [1, 0]}}))
    reports = [{'rank': rank, 'puzzle_id': 7, 'status': 'solved',
                'exit_code': 9 if bad_rank and rank == 1 else 0} for rank in range(2)]
    path = 'missing' if bad_path else 'swap'
    payload = ('import pathlib,json; pathlib.Path("rank-status").mkdir(); '
        f'reports={reports!r}; '
        '[pathlib.Path("rank-status/rank-"+str(r["rank"])+".json").write_text(json.dumps(r)) for r in reports]; '
        f'pathlib.Path("submit.csv").write_text("initial_state_id,path\\n7,{path}\\n")')
    result, summary = launch(tmp_path, payload,
        extra=('--world-size', '2', '--puzzle-contract', str(contract)))
    if bad_rank or bad_path:
        assert result.returncode != 0
        assert summary['status'] == 'failed'
    else:
        assert result.returncode == 0
        assert summary['status'] == 'solved'
        assert summary['solution_replay_valid'] is True
        assert summary['solution_length'] == 1
        assert summary['production_accepted'] is False


def test_deadline_kills_own_sleeping_child_and_records_timeout(tmp_path):
    result, summary = launch(tmp_path, 'import time; time.sleep(30)', elapsed='0.2')
    assert result.returncode != 0
    assert summary['status'] == 'timed_out'
    assert summary['elapsed_seconds'] < 5


def test_failed_child_is_not_unsolved_success(tmp_path):
    result, summary = launch(tmp_path, 'raise SystemExit(7)')
    assert result.returncode != 0
    assert summary['status'] == 'failed'
    assert summary['child_exit_code'] == 7


def test_exit_zero_is_not_replay_verified_solution(tmp_path):
    result, summary = launch(tmp_path, 'print("puzzle_solved=1")')
    assert result.returncode == 0
    assert summary['production_accepted'] is False
    assert summary['status'] == 'completed_unverified'


def test_cost_cap_shortens_deadline(tmp_path):
    result, summary = launch(tmp_path, 'import time; time.sleep(30)',
                             elapsed='10', cost='0.0001', rate='3.6')
    assert result.returncode != 0
    assert summary['status'] == 'timed_out'
    assert summary['effective_deadline_seconds'] == pytest.approx(0.1)


def test_external_sigterm_preserves_summary_and_stops_child(tmp_path):
    directory = tmp_path / 'signal-run'
    payload = 'import os,time; open("child.pid","w").write(str(os.getpid())); time.sleep(30)'
    command = [sys.executable, str(ROOT / 'tools/cube4_run_supervisor.py'),
        '--run-dir', str(directory), '--max-elapsed-seconds', '20',
        '--max-cost-usd', '1', '--hourly-rate-usd', '1', '--', sys.executable, '-c', payload]
    supervisor = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child = None
    try:
        limit = time.monotonic() + 5
        while not (directory / 'child.pid').exists() and time.monotonic() < limit:
            time.sleep(0.02)
        child = int((directory / 'child.pid').read_text())
        supervisor.send_signal(signal.SIGTERM)
        supervisor.communicate(timeout=5)
        assert (directory / 'summary.json').exists(), 'SIGTERM lost terminal evidence'
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['status'] == 'failed'
        assert summary['interrupted_signal'] == signal.SIGTERM
        assert not Path(f'/proc/{child}').exists(), 'owned child survived supervisor signal'
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()
        # Reap the RED reproducer's owned child only, never unrelated jobs.
        if child is not None:
            try:
                if os.getpgid(child) == child:
                    os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_deadline_kills_term_ignoring_descendant_but_not_unrelated_process(tmp_path):
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        descendant = ('import os,signal,time; '
            'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
            'open("descendant.pid","w").write(str(os.getpid())); time.sleep(30)')
        payload = ('import subprocess,sys,time,pathlib; '
            f'subprocess.Popen([sys.executable,"-c",{descendant!r}]); '
            'time.sleep(30)')
        result, summary = launch(tmp_path, payload, elapsed='1')
        assert summary['status'] == 'timed_out'
        child = int((tmp_path / 'run/descendant.pid').read_text())
        status = Path(f'/proc/{child}/status')
        if status.exists():
            # Orphan zombies are already dead; their adopter owns reaping.
            assert '\nState:\tZ ' in status.read_text()
        assert unrelated.poll() is None, 'supervisor killed an unrelated process'
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_signal_between_spawn_and_assignment_cannot_orphan_child(tmp_path):
    # Wrap real Popen only to deliver SIGTERM at the ownership hand-off.
    # The process and its lifetime are real, not mocked wait/kill behavior.
    directory = tmp_path / 'spawn-run'
    pid_path = tmp_path / 'spawned.pid'
    harness = f'''
import os,signal,subprocess,sys
sys.path.insert(0, {str(ROOT)!r})
from tools import cube4_run_supervisor as supervisor
real_popen = subprocess.Popen
def interrupted_spawn(*args, **kwargs):
    child = real_popen(*args, **kwargs)
    open({str(pid_path)!r}, 'w').write(str(child.pid))
    os.kill(os.getpid(), signal.SIGTERM)
    return child
supervisor.subprocess.Popen = interrupted_spawn
sys.argv = ['supervisor', '--run-dir', {str(directory)!r},
 '--max-elapsed-seconds', '10', '--max-cost-usd', '1', '--hourly-rate-usd', '1',
 '--', sys.executable, '-c', 'import time; time.sleep(30)']
raise SystemExit(supervisor.main())
'''
    child = None
    try:
        result = subprocess.run([sys.executable, '-c', harness], capture_output=True,
                                text=True, timeout=8)
        child = int(pid_path.read_text())
        assert result.returncode != 0
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['interrupted_signal'] == signal.SIGTERM
        status = Path(f'/proc/{child}/status')
        assert not status.exists() or '\nState:\tZ ' in status.read_text(), 'spawned child escaped cleanup'
    finally:
        if child is not None:
            try:
                if os.getpgid(child) == child:
                    os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
