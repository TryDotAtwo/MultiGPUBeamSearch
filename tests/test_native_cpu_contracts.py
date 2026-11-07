"""Compile and execute current C++ contracts in Linux; no CUDA acceptance."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'Linux g++ required')
class NativeCpuContractTests(unittest.TestCase):
    def check_target(self, target):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / target
            sources = ['config', 'state', 'hash', 'history', 'history_budget',
                       'native_input', 'stream3', 'stream4', 'frontier_cpu']
            command = ['g++', '-std=c++20', '-O1', '-fsanitize=undefined',
                '-fno-sanitize-recover=undefined', '-I', str(ROOT / 'src'), '-I', str(ROOT),
                '-DBEAM_STATE_LOGICAL_BYTES=96', '-DBEAM_STATE_PHYSICAL_BYTES=112',
                '-DBEAM_STATE_ALIGNMENT=16', '-DBEAM_MOVE_COUNT=24',
                str(ROOT / 'tests' / (target + '.cpp'))]
            command += [str(ROOT / 'src' / (name + '.cpp')) for name in sources]
            command += ['-o', str(binary)]
            build = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30,
                                 cwd=directory)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertNotIn('runtime error:', run.stderr)
            print(target + ': ' + run.stdout.strip())

    def test_history_budget_current_cube4(self):
        self.check_target('history_budget_tests')

    def test_history_prefault_bounds(self):
        self.check_target('history_prefault_tests')

    def test_config_limits_current_cube4(self):
        self.check_target('config_limits_tests')

    def test_contracts_current_cube4(self):
        self.check_target('contract_tests')

    def test_unique_frontier_current_cube4(self):
        self.check_target('unique_frontier_tests')

    @unittest.skipUnless(shutil.which('cmake') and shutil.which('nvcc'), 'configured CUDA toolchain required, not GPU')
    def test_registered_cpu_targets_through_cmake_ctest(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / 'build'
            commands = [
                ['cmake', '-S', str(ROOT), '-B', str(build),
                 '-DBEAM_STATE_LOGICAL_BYTES=96', '-DBEAM_MOVE_COUNT=24',
                 '-DCMAKE_CUDA_ARCHITECTURES=86', '-DBEAM_CUDA_ARCHITECTURES=86',
                 '-DCUTLASS_DIR=' + os.environ.get('CUTLASS_DIR', '/opt/cutlass')],
                ['cmake', '--build', str(build), '--parallel', '2', '--target',
                 'contract_tests', 'history_budget_tests', 'config_limits_tests', 'unique_frontier_tests',
                 'stream1_layout_metadata_tests', 'stream1_policy_snapshot_tests', 'stream1_resolved_gemm_tests', 'stream1_loaded_execution_admission_tests', 'stream1_execution_contract_tests', 'production_score_request_tests'],
                ['ctest', '--test-dir', str(build), '--output-on-failure', '-R',
                 '^(contract_tests|history_budget_tests|config_limits_tests|unique_frontier_tests|stream1_layout_metadata_tests|stream1_policy_snapshot_tests|stream1_resolved_gemm_tests|stream1_loaded_execution_admission_tests|stream1_execution_contract_tests|production_score_request_tests)$']]
            for command in commands:
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(result.stdout)
                print(result.stderr)
