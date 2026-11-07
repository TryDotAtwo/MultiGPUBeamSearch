"""POSIX child-group deadline supervisor; no cloud lifecycle or solve promotion.

Cost is an estimate using the caller's combined hourly rate, not a billing
receipt. Stopping a child does not stop an instance or its storage charges.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import csv
import io
from hashlib import sha256
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cayleypy_public.paths import apply_path, tokenize_path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def json_object(raw):
    value = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError('expected JSON object')
    return value


def load_contract(path: Path) -> tuple[dict, str]:
    with path.open('rb') as stream:
        raw = stream.read(1048577)
    if len(raw) > 1048576:
        raise ValueError('puzzle contract exceeds 1MiB')
    contract = json_object(raw)
    if type(contract.get('puzzle_id')) is not int or not 0 <= contract['puzzle_id'] < 2**64:
        raise ValueError('invalid puzzle identity')
    initial, central = contract['initial_state'], contract['central_state']
    if not isinstance(initial, list) or not isinstance(central, list) or not 1 <= len(initial) <= 120 or len(initial) != len(central):
        raise ValueError('invalid puzzle state lengths')
    if any(type(value) is not int or not 0 <= value <= 255 for value in initial + central):
        raise ValueError('invalid puzzle state values')
    if sorted(initial) != sorted(central):
        raise ValueError('puzzle states have different symbol multisets')
    generators = contract['generators']
    if not isinstance(generators, dict) or not generators:
        raise ValueError('generators must be a nonempty object')
    for name, permutation in generators.items():
        if len(tokenize_path(name)) != 1:
            raise ValueError('generator name must be one move token')
        if (not isinstance(permutation, list) or
            any(type(index) is not int for index in permutation)):
            raise ValueError('generator indices must be exact integers')
    apply_path(initial, '', generators)
    return contract, sha256(raw).hexdigest()


def verify_rank_result(directory: Path, world: int, contract: dict, *, deadline: float | None = None) -> dict:
    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('result verification deadline exceeded')
    check_deadline()
    reports = []
    for rank in range(world):
        with (directory / 'rank-status' / f'rank-{rank}.json').open('rb') as stream:
            raw = stream.read(32769)
        if len(raw) > 32768:
            raise ValueError('rank status exceeds limit')
        report = json_object(raw)
        if (type(report.get('rank')) is not int or report['rank'] != rank or
            type(report.get('exit_code')) is not int or report['exit_code'] != 0 or
            type(report.get('puzzle_id')) is not int or report['puzzle_id'] != contract['puzzle_id']):
            raise ValueError('rank failure or identity mismatch')
        reports.append(report)
    statuses = {report.get('status') for report in reports}
    if statuses == {'unsolved'}:
        requested = contract.get('requested_depth')
        if (type(requested) is not int or not 1 <= requested < 2**32 or
            any(type(report.get('completed_depths')) is not int or
                report['completed_depths'] != requested for report in reports)):
            raise ValueError('unsolved ranks did not complete requested depth')
        if (directory / 'submit.csv').exists():
            raise ValueError('unsolved ranks produced submission')
        check_deadline()
        return {'status': 'unsolved', 'rank_statuses': reports}
    if statuses != {'solved'}:
        raise ValueError('ranks do not agree on terminal result')
    with (directory / 'submit.csv').open('rb') as stream:
        raw = stream.read(1048577)
    if len(raw) > 1048576:
        raise ValueError('submission exceeds 1MiB')
    try:
        with io.StringIO(raw.decode('utf-8'), newline='') as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != ['initial_state_id', 'path']:
                raise ValueError('invalid submission schema')
            rows = list(reader)
    except csv.Error as error:
        raise ValueError('invalid submission CSV: ' + str(error)) from error
    if any(set(row) != {'initial_state_id', 'path'} or
           any(value is None for value in row.values()) for row in rows):
        raise ValueError('invalid submission schema: row field count')
    if len(rows) != 1 or rows[0]['initial_state_id'] != str(contract['puzzle_id']):
        raise ValueError('submission puzzle mismatch')
    path = rows[0]['path']
    if apply_path(contract['initial_state'], path, contract['generators'], deadline=deadline) != tuple(contract['central_state']):
        raise ValueError('independent solution replay failed')
    check_deadline()
    return {'status': 'solved', 'rank_statuses': reports,
            'solution_replay_valid': True, 'solution_length': len(tokenize_path(path)),
            'solution_path': path}


class SupervisorInterrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f'supervisor interrupted by signal {signum}')


def interrupt(signum, _frame):
    raise SupervisorInterrupted(signum)


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('expected positive finite number')
    return number


def atomic_summary(directory: Path, record: dict) -> None:
    temporary = directory / 'summary.json.tmp'
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / 'summary.json')


def stop_group(process: subprocess.Popen) -> None:
    # The child alone creates this session; never enumerate/kill other jobs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # Include descendants even if their immediate parent exited on TERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--max-elapsed-seconds', type=positive, required=True)
    parser.add_argument('--max-cost-usd', type=positive, required=True)
    parser.add_argument('--hourly-rate-usd', type=positive, required=True)
    parser.add_argument('--world-size', type=int)
    parser.add_argument('--puzzle-contract', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('child command required after --')
    if os.name != 'posix':
        parser.error('POSIX process groups required; Windows Job Object support is not implemented')
    if (args.world_size is None) != (args.puzzle_contract is None):
        parser.error('world-size and puzzle-contract must be provided together')
    if args.world_size is not None and not 1 <= args.world_size <= 255:
        parser.error('world-size must be in [1,255]')
    deadline = min(args.max_elapsed_seconds, args.max_cost_usd / args.hourly_rate_usd * 3600)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {'status': 'running', 'command': command,
              'effective_deadline_seconds': deadline,
              'hourly_rate_usd': args.hourly_rate_usd,
              'max_cost_usd': args.max_cost_usd, 'production_accepted': False,
              'shutdown_policy': 'own_child_process_group_only',
              'cost_scope': 'estimated workload time; instance/storage billing continues'}
    process = None
    spawning = False
    pending_signal = None

    def owned_interrupt(signum, frame):
        nonlocal pending_signal
        if spawning:
            pending_signal = signum
        else:
            interrupt(signum, frame)

    previous_handlers = {number: signal.signal(number, owned_interrupt)
                         for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        contract = None
        if args.puzzle_contract is not None:
            contract, identity = load_contract(args.puzzle_contract)
            record['puzzle_contract_sha256'] = identity
        with (args.run_dir / 'child.log').open('x') as log:
            spawning = True
            process = subprocess.Popen(command, cwd=args.run_dir,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            spawning = False
            if pending_signal is not None:
                raise SupervisorInterrupted(pending_signal)
            try:
                code = process.wait(timeout=max(0., started + deadline - time.monotonic()))
                record.update(status='completed_unverified' if code == 0 else 'failed',
                              child_exit_code=code)
            except subprocess.TimeoutExpired:
                record['status'] = 'timed_out'
    except (OSError, ValueError, KeyError, TypeError, KeyboardInterrupt, SupervisorInterrupted) as error:
        record.update(status='failed', error=str(error))
        if isinstance(error, SupervisorInterrupted):
            record['interrupted_signal'] = error.signum
    finally:
        # Repeated cancellation must not interrupt cleanup or its evidence.
        for number in previous_handlers:
            signal.signal(number, signal.SIG_IGN)
        if process is not None:
            stop_group(process)
            record.setdefault('child_exit_code', process.returncode)
        if record['status'] == 'completed_unverified' and contract is not None:
            try:
                record.update(verify_rank_result(args.run_dir, args.world_size, contract,
                                                deadline=started + deadline))
            except TimeoutError as error:
                record.update(status='timed_out', error=str(error))
            except (OSError, ValueError, KeyError, TypeError) as error:
                record.update(status='failed', error=str(error))
        record['elapsed_seconds'] = time.monotonic() - started
        record['estimated_cost_usd'] = record['elapsed_seconds'] * args.hourly_rate_usd / 3600
        atomic_summary(args.run_dir, record)
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    return 0 if record['status'] in ('completed_unverified', 'solved', 'unsolved') else 1


if __name__ == '__main__':
    raise SystemExit(main())
