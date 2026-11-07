"""CPU shell integration: real launchers must give ranks fresh shared sessions.

Fake executables isolate launch environment only; no GPU/model acceptance.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'requires POSIX bash')
class ManualLauncherSessionTests(unittest.TestCase):
    def check_launcher(self, kind, reject_cgroup=False):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            data = tmp / 'data'
            data.mkdir()
            for name in ('puzzle_info.json', 'test.csv', 'manifest.json'):
                (data / name).write_text('{}')
            build = tmp / 'build'
            build.mkdir()
            runner = build / 'production_runner'
            runner.write_text('#!/bin/bash\nset -eu\n'
                'if [[ ${1:-} == --build-info ]]; then echo \'{"schema_version":1,"state_len":96,"state_storage_len":112,"move_count":24,"candidate_meta_bytes":32,"cuda_architectures":"90a"}\'; exit 0; fi\n'
                'expected="$(dirname "$BEAM_NCCL_ID_FILE")"\n'
                '[[ ${BEAM_NCCL_RUN_ID:-} == "$expected" ]] || exit 19\n'
                'printf "session=%s rank=%s\\n" "$BEAM_NCCL_RUN_ID" "$RANK"\n')
            runner.chmod(0o755)
            shutil.copyfile(runner, build / 'stream_benchmark')
            (build / 'stream_benchmark').chmod(0o755)
            (build / 'CMakeCache.txt').write_text('BEAM_CUDA_ARCHITECTURES:STRING=86\nBEAM_STATE_LOGICAL_BYTES:STRING=96\n')
            env = {**os.environ, 'BEAM_NCCL_RUN_ID': 'inherited-poison',
                   'BEAM_WIDTH': '4096', 'BEAM_B_MICRO': '32',
                   'BEAM_STREAM1_CONCURRENCY': '1', 'BEAM_STREAM3_RING_SLOTS': '2',
                   'BEAM_SHARD_COUNT': '4', 'BEAM_HISTORY_RAM_BYTES': '1024',
                   'BEAM_HISTORY_DISK_BYTES': '1024'}
            prefix = {'b300': 'B300', 'portable': 'PORTABLE', 'h200': 'H200'}[kind]
            env.update({prefix + '_REPO_DIR': str(ROOT), prefix + '_DATA_DIR': str(data),
                        prefix + '_WEIGHT_DIR': str(data), prefix + '_BUILD_DIR': str(build),
                        prefix + '_RUN_ROOT': str(tmp), prefix + '_WORLD_SIZE': '2'})
            if kind in ('portable', 'h200', 'b300'):
                smi = tmp / 'smi'
                smi.write_text('#!/bin/bash\nprintf "8.6\\n8.6\\n"\n')
                smi.chmod(0o755)
                verifier = tmp / 'verifier'
                verifier.write_text('#!/bin/bash\n'
                    'if [[ "$1" == *host_memory_budget.py ]]; then exec "' + sys.executable + '" "$@"; fi\n'
                    'exit 0\n')
                verifier.chmod(0o755)
                env.update(PORTABLE_SMI=str(smi), PORTABLE_PYTHON=str(verifier))
                if kind == 'h200':
                    h200_smi = tmp / 'h200-smi'
                    h200_smi.write_text('#!/bin/bash\nprintf "NVIDIA H200, 143771, 9.0\\nNVIDIA H200, 143771, 9.0\\n"\n')
                    h200_smi.chmod(0o755)
                    env.update(H200_PYTHON=str(verifier), BEAM_WEIGHT_DIR=str(data),
                               H200_SMI=str(h200_smi))
                if kind == 'b300':
                    env['B300_PYTHON'] = str(verifier)
            script = ROOT / 'hpc' / {'b300': 'b300_fp16.sh', 'portable': 'cube4_portable.sh',
                                      'h200': 'h200_dual_large_beam.sh'}[kind]
            run_glob = {'b300': 'b300_run.*', 'portable': 'cube4_run.*', 'h200': 'h200_run.*'}[kind]
            if reject_cgroup:
                meminfo = tmp / 'meminfo'
                meminfo.write_text('MemAvailable: 100000000 kB\n')
                cgroup = tmp / 'cgroup'
                cgroup.mkdir()
                (cgroup / 'memory.max').write_text('2000')
                (cgroup / 'memory.current').write_text('1000')
                env.update(PORTABLE_MEMINFO_PATH=str(meminfo), BEAM_HOST_CGROUP_ROOT=str(cgroup))
                env['H200_MEMINFO_PATH'] = str(meminfo)
                env['B300_MEMINFO_PATH'] = str(meminfo)
                result = subprocess.run(['bash', str(script), 'run'], env=env,
                    cwd=ROOT, capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('insufficient container RAM', result.stderr)
                self.assertFalse(list(tmp.glob(run_glob + '/rank*.log')))
                return
            for _ in range(2):
                result = subprocess.run(['bash', str(script), 'run'], env=env,
                    cwd=ROOT, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runs = list(tmp.glob(run_glob))
            self.assertEqual(len(runs), 2)
            for run in runs:
                for rank in (0, 1):
                    self.assertEqual((run / f'rank{rank}.log').read_text().strip(),
                                     f'session={run} rank={rank}')

    def test_b300_overwrites_inherited_session_and_shares_new_id(self):
        self.check_launcher('b300')

    def test_b300_cgroup_preflight_rejects_before_child_ranks(self):
        self.check_launcher('b300', reject_cgroup=True)

    def test_portable_overwrites_inherited_session_and_shares_new_id(self):
        self.check_launcher('portable')

    def test_portable_cgroup_preflight_rejects_before_child_ranks(self):
        self.check_launcher('portable', reject_cgroup=True)

    def test_h200_cgroup_preflight_rejects_before_child_ranks(self):
        self.check_launcher('h200', reject_cgroup=True)

    def test_h200_positive_memory_gate_preserves_rank_sessions(self):
        self.check_launcher('h200')

    def test_mephi_prepare_replaces_foreign_id_and_isolates_repeated_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            script = ROOT / 'hpc/mephi_8xa100_common.sh'
            command = ['bash', '-c',
                'set -eu; source "$1"; '
                'for attempt in 1 2; do beam_prepare_nccl_file same-tag; '
                'for rank in 0 1; do '
                '(printf "%s|%s|%s\\n" "$rank" "$BEAM_NCCL_RUN_ID" "$BEAM_NCCL_ID_FILE"); '
                'done; done', 'test-wrapper', str(script)]
            result = subprocess.run(command, env={**os.environ, 'JOB_DIR': directory,
                'BEAM_NCCL_RUN_ID': 'inherited-poison'}, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [line.split('|') for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0][1:], rows[1][1:])
            self.assertEqual(rows[2][1:], rows[3][1:])
            self.assertNotEqual(rows[0][1], rows[2][1])
            for rank, session, path in rows:
                self.assertIn(rank, ('0', '1'))
                self.assertNotEqual(session, 'inherited-poison')
                self.assertEqual(Path(path).parent, Path(session))
                self.assertTrue(Path(session).is_dir())

    def check_mephi_memory_gate(self, reject, segment=False):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / 'venv/bin').mkdir(parents=True)
            python = tmp / 'venv/bin/python'
            python.write_text('#!/bin/bash\n'
                'if [[ "$1" == *host_memory_budget.py ]]; then exec "' + sys.executable + '" "$@"; fi\n'
                'touch "$LAUNCH_MARKER"\nexit 0\n')
            python.chmod(0o755)
            meminfo = tmp / 'meminfo'
            meminfo.write_text('MemAvailable: 100000000 kB\n')
            cgroup = tmp / 'cgroup'
            cgroup.mkdir()
            (cgroup / 'memory.max').write_text('2000' if reject else '100000000000')
            (cgroup / 'memory.current').write_text('1000')
            env = {**os.environ, 'LOG_DIR': str(tmp), 'TUNING_DIR': str(tmp),
                'BUILD_DIR': str(tmp), 'REPO_DIR': str(ROOT),
                'NINJA_VENV_DIR': str(tmp / 'venv'), 'BEAM_NCCL_ID_FILE': str(tmp / 'nccl'),
                'TORCHRUN_NNODES': '1', 'TORCHRUN_NPROC_PER_NODE': '2',
                'TORCHRUN_NODE_RANK': '0', 'TORCHRUN_RDZV_ENDPOINT': '127.0.0.1:29500',
                'WORLD_SIZE_EFFECTIVE': '2', 'PUZZLE_ID': '1000', 'DEPTH_LIMIT': '1',
                'BEAM_WIDTH': '4096', 'BEAM_HISTORY_RAM_BYTES': '1024',
                'BEAM_HOST_MEMINFO_PATH': str(meminfo), 'BEAM_HOST_CGROUP_ROOT': str(cgroup),
                'LAUNCH_MARKER': str(tmp / 'launched')}
            env['JOB_DIR'] = str(tmp)
            (tmp / 'plan.tsv').write_text('repair\tdepth\twindow\tstart\ttarget\tlength\tpath\tstate\n')
            launch = ('beam_torchrun_segment_plan fixture "$2" "$JOB_DIR/plan.tsv" '
                      '"$NINJA_VENV_DIR/bin/python" "$JOB_DIR/history"' if segment else
                      'beam_torchrun_production fixture "$2"')
            result = subprocess.run(['bash', '-c',
                'set -eu; source "$1"; ' + launch,
                'test-wrapper', str(ROOT / 'hpc/mephi_8xa100_common.sh'), str(tmp / 'run.log')],
                env=env, capture_output=True, text=True, timeout=20)
            if reject:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('insufficient container RAM', result.stderr)
                self.assertFalse((tmp / 'launched').exists())
                self.assertFalse((tmp / 'nvidia_smi_fixture.log').exists())
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((tmp / 'launched').exists())
                self.assertIn('"available_bytes": 99999999000', result.stdout)

    def test_mephi_memory_gate_rejects_before_torchrun_or_monitor(self):
        self.check_mephi_memory_gate(reject=True)

    def test_mephi_memory_gate_allows_sufficient_budget(self):
        self.check_mephi_memory_gate(reject=False)

    def test_mephi_segment_memory_gate_rejects_before_children(self):
        self.check_mephi_memory_gate(reject=True, segment=True)

    def test_segment_plan_isolates_repeated_repair_id_across_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            plan = tmp / 'plan.tsv'
            plan.write_text('repair\tdepth\twindow\tstart\ttarget\tlength\tpath\tstate\n'
                            '7\t3\t1\t0\t1\t1\tf0\t0\n' +
                            '7\t3\t1\t0\t1\t1\tf0\t0\n')
            runner = tmp / 'runner'
            output = tmp / 'records'
            runner.write_text('#!/bin/bash\nset -eu\n'
                'printf "%s|%s|%s\\n" "$RANK" "$BEAM_NCCL_RUN_ID" "$BEAM_NCCL_ID_FILE" >> "$TEST_RECORDS"\n')
            runner.chmod(0o755)
            script = ROOT / 'hpc/ihes_cube_model/run_solution_repair_plan.sh'
            for rank in (0, 1):
                result = subprocess.run(['bash', str(script), str(plan), str(runner),
                    '4096', str(tmp / 'history'), str(tmp)], env={**os.environ,
                    'RANK': str(rank), 'WORLD_SIZE': '2', 'BEAM_NCCL_RUN_ID': 'outer-session',
                    'BEAM_NCCL_ID_FILE': str(tmp / 'outer-nccl.bin'), 'TEST_RECORDS': str(output)},
                    capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = [row.split('|') for row in output.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0][1:], rows[2][1:])
            self.assertEqual(rows[1][1:], rows[3][1:])
            self.assertNotEqual(rows[0][1], rows[1][1])
            self.assertNotEqual(rows[0][2], rows[1][2])


if __name__ == '__main__':
    unittest.main()
