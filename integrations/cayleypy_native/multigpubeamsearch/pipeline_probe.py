"""Native rank-plan and full-frontier adapters for downstream calibration."""
from __future__ import annotations
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .calibration_stats import Measurement
from .pipeline_profiles import validate_rank_plans


def parse_plan(text):
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)=(\d+)', line)
        if match:
            key, value = match.groups()
            if key in result and result[key] != int(value):
                raise ValueError('native plan changes '+key)
            result[key] = int(value)
    if 'GLOBAL_BEAM_WIDTH_EFFECTIVE' not in result:
        raise ValueError('native plan is missing effective beam')
    return result


def parse_depths(text, expected_parents, repeats):
    rows = []
    for line in text.splitlines():
        if not line.startswith('calibration_depth='):
            continue
        fields = dict(part.split('=', 1) for part in line.split())
        if int(fields['calibration_depth']) != len(rows):
            raise ValueError('missing, duplicate or reordered calibration depth')
        if int(fields['parents']) != expected_parents:
            raise ValueError('frontier workload changed during calibration')
        seconds = float(fields['depth_sec'])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid complete-depth timing')
        rows.append(seconds)
    if len(rows) != repeats:
        raise ValueError('native run did not complete all calibration depths')
    return rows


class NativePipelineProbe:
    """Runs the actual native planner and pipeline, without timing estimates.

    Fixtures must be independently verified legal full-frontier files. Their
    provenance belongs to the caller; this adapter checks exact physical size.
    One process group per rank is always cleaned up on timeout or failure.
    """
    def __init__(self, runner, environment, beam_width, world_size, directory,
                 fixtures, physical_bytes, *, deadline, verify, repeats=6, puzzle_id=0):
        if not 6 <= repeats <= 16:
            raise ValueError('need a warmup depth and at least five measurements')
        self.runner = Path(runner)
        self.environment = dict(environment)
        self.beam = beam_width
        self.world = world_size
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fixtures = tuple(Path(x) for x in fixtures)
        if len(self.fixtures) != world_size:
            raise ValueError('missing rank fixtures')
        self.physical_bytes = physical_bytes
        self.deadline = deadline
        self.repeats = repeats
        if not callable(verify):
            raise TypeError('independent pipeline correctness verifier is required')
        self.verify = verify
        self.counter = 0
        if type(puzzle_id) is not int or puzzle_id < 0:
            raise ValueError('calibration puzzle ID must be nonnegative')
        self.puzzle_id = puzzle_id

    def _run(self, environment, *, planning):
        from .backend import _stop_process_tree
        if time.monotonic() >= self.deadline:
            raise ValueError('pipeline calibration budget expired')
        self.counter += 1
        directory = self.directory / str(self.counter)
        directory.mkdir()
        (directory / 'test_results').mkdir()
        env = dict(self.environment, **environment)
        env.update(BEAM_NCCL_RUN_ID='calibration-'+uuid.uuid4().hex,
                   BEAM_NCCL_ID_FILE=str(directory/'nccl-id.bin'))
        env.pop('BEAM_BENCHMARK_PLAN_ONLY', None)
        env.pop('BEAM_BENCHMARK_FRONTIER_FILE', None)
        if planning:
            env['BEAM_BENCHMARK_PLAN_ONLY'] = '1'
        else:
            env['BEAM_BENCHMARK_FRONTIER_REPEATS'] = str(self.repeats)
            env['BEAM_HISTORY_DIR'] = str(directory/'history')
            env['BEAM_HISTORY_DISK_PATH'] = str(directory/'history')
        processes = []
        try:
            for rank in range(self.world):
                rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(self.world))
                if not planning:
                    rank_env['BEAM_BENCHMARK_FRONTIER_FILE'] = str(self.fixtures[rank])
                log = (directory/f'rank-{rank}.log').open('wb')
                command = [str(self.runner), str(self.puzzle_id), '1' if planning else str(self.repeats),
                           str(self.beam), str(self.world), str(rank)]
                process = subprocess.Popen(command, env=rank_env, cwd=directory, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                processes.append((process, log, directory/f'rank-{rank}.log'))
            texts = []
            for process, log, path in processes:
                try:
                    code = process.wait(timeout=max(.1, self.deadline-time.monotonic()))
                except subprocess.TimeoutExpired as error:
                    raise ValueError('native calibration timed out: '+str(directory)) from error
                log.close()
                text = path.read_text(errors='replace')
                if code or 'production_runner_error=' in text:
                    raise ValueError('native calibration failed: '+str(path))
                texts.append(text)
            return texts
        finally:
            for process, log, _ in processes:
                if process.poll() is None:
                    _stop_process_tree(process)
                log.close()

    def admit(self, environment):
        plans = [parse_plan(text) for text in self._run(environment, planning=True)]
        validate_rank_plans(plans)
        if plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE'] < self.beam:
            raise ValueError('native admission shrinks requested beam')
        return plans

    def measure(self, environment, plans):
        for rank, path in enumerate(self.fixtures):
            if path.stat().st_size != plans[rank]['frontier_state_capacity'] * self.physical_bytes:
                raise ValueError('fixture must fill the exact admitted rank frontier')
        texts = self._run(environment, planning=False)
        if self.verify(texts, plans) is not True:
            raise ValueError('full-pipeline correctness verification failed')
        seconds = [parse_depths(text, plans[rank]['frontier_state_capacity'], self.repeats)
                   for rank, text in enumerate(texts)]
        parents = sum(plan['frontier_state_capacity'] for plan in plans)
        # First complete depth is warmup. No mean-of-means or summed GPU rates.
        return [Measurement('', parents, tuple(row[index] for row in seconds), True, True)
                for index in range(1, self.repeats)]

