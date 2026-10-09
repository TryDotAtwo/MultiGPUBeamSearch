"""Actual native CLI boundary tests; missing fixtures never allocate a GPU."""
import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('BEAM_SCORE_PRODUCER_BINARY'), 'built native producer required')
class NativeScoreCliTests(unittest.TestCase):
    def test_requested_lane_profile_is_parsed_before_missing_fixture(self):
        result = self.run_producer(['--runtime-cuda-graph-scores', '--verify-production-lanes',
                                    '--probe-microbatch', '768', '--probe-lanes', '12'])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('reference_required', result.stderr)

    def test_invalid_lane_profile_rejected_before_fixture(self):
        for arguments in (['--probe-microbatch', '0'], ['--probe-lanes', '0'],
                          ['--probe-microbatch', '-1'], ['--probe-lanes', '65'],
                          ['--probe-microbatch', '65537'], ['--probe-microbatch'],
                          ['--probe-lanes', '4', '--probe-lanes', '4']):
            with self.subTest(arguments=arguments):
                result = self.run_producer(['--runtime-cuda-graph-scores',
                                            '--verify-production-lanes', *arguments])
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertNotIn('missing_reference_fixture', result.stdout)

    def test_invalid_reference_rejected_before_cuda(self):
        # A direct binary invocation must not bypass the Python schema gate.
        # Removing native shape/value validation must make this test fail.
        for invalid in ('not-json', json.dumps({'states': [[0.5] * 96] * 8,
                                                'scores_fp32': [[1] * 24] * 8}),
                        json.dumps({'states': [[0] * 95] * 8,
                                    'scores_fp32': [[1] * 24] * 8})):
            with self.subTest(reference=invalid[:30]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'weights_fp16').mkdir()
                (root / 'weights_fp16/manifest.json').write_text('{}', encoding='utf-8')
                (root / 'reference.json').write_text(invalid, encoding='utf-8')
                env = dict(os.environ, BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR=directory)
                result = subprocess.run([os.environ['BEAM_SCORE_PRODUCER_BINARY'],
                                         '--runtime-cuda-graph-scores'], cwd=directory,
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('invalid_reference', result.stderr)
                self.assertNotIn('CUDA', result.stderr)

    def run_producer(self, arguments):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR=str(Path(directory) / 'absent'))
            result = subprocess.run([os.environ['BEAM_SCORE_PRODUCER_BINARY'], *arguments],
                cwd=directory, env=env, capture_output=True, text=True, timeout=10)
            return result

    def test_runtime_mode_requires_reference_instead_of_skipping(self):
        result = self.run_producer(['--require-reference', '--runtime-final-cls-scores'])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('reference_required', result.stderr)

    def test_unknown_option_fails_before_fixture(self):
        result = self.run_producer(['--unknown-score-mode'])
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertNotIn('missing_reference_fixture', result.stdout)

    def test_graph_score_mode_requires_fixture_without_explicit_require_flag(self):
        # Catch silently skipping graph numerical validation when a deployment
        # has no prepared independent reference. This uses the actual binary.
        result = self.run_producer(['--runtime-cuda-graph-scores'])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('reference_required', result.stderr)

    def test_graph_input_reuse_probe_requires_fixture(self):
        result = self.run_producer(['--runtime-cuda-graph-scores', '--verify-graph-input-reuse'])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('reference_required', result.stderr)

    def test_graph_job_bounds_probe_requires_fixture(self):
        result = self.run_producer(['--runtime-cuda-graph-scores', '--verify-graph-job-bounds'])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('reference_required', result.stderr)

    def test_graph_lane_probe_requires_fixture(self):
        result = self.run_producer(['--runtime-cuda-graph-scores', '--verify-graph-lanes'])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('reference_required', result.stderr)

    def test_production_lane_probe_requires_fixture(self):
        result = self.run_producer(['--runtime-cuda-graph-scores', '--verify-production-lanes'])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('reference_required', result.stderr)

    def test_duplicate_graph_score_option_fails_before_fixture(self):
        result = self.run_producer(['--runtime-cuda-graph-scores', '--runtime-cuda-graph-scores'])
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertNotIn('missing_reference_fixture', result.stdout)
