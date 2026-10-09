"""Live POSIX producer protocol tests; fake producer is not GPU evidence."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]


def prepared_weights(weights):
    """Independent fixed-shape fixture, not production-exporter expectations."""
    model = dict(backend='piece_transformer', state_len=96, move_count=24, output_dim=24,
        seq_len=57, dtype='fp16', activation='relu', pooling='cls', num_classes=6,
        num_pieces=56, max_piece_size=3, num_piece_types=3, d_model=256,
        nhead=8, head_dim=32, num_layers=4, ff_dim=1024, piece_layout='cube4',
        piece_embed_mode='piece_local', input_embedding='fast_slot_projected')
    sizes = dict(cls_token=512, fast_piece_static=28672, fast_slot_projected=9216,
        input_ln_beta=512, input_ln_gamma=512, output_bias=48,
        output_ln_beta=512, output_ln_gamma=512, output_weight_hxk=12288)
    block = dict(attn_out_bias=512, attn_out_weight_hxk=131072, attn_qkv_bias=1536,
        attn_qkv_weight_hxk=393216, ff1_bias=2048, ff1_weight_hxk=524288,
        ff2_bias=512, ff2_weight_hxk=524288, ln1_beta=512, ln1_gamma=512,
        ln2_beta=512, ln2_gamma=512)
    for index in range(4):
        sizes.update({f'block{index}_{key}': value for key, value in block.items()})
    for name, size in sizes.items():
        (weights / (name + '.fp16')).write_bytes(bytes(size))
    positions, mask, cursor = [], [], 0
    for size in [3] * 8 + [2] * 24 + [1] * 24:
        positions.extend(list(range(cursor, cursor + size)) + [0] * (3 - size))
        mask.extend([1] * size + [0] * (3 - size)); cursor += size
    (weights / 'piece_positions.u16').write_bytes(struct.pack('<168H', *positions))
    (weights / 'piece_mask.u8').write_bytes(bytes(mask))
    (weights / 'piece_types.u8').write_bytes(bytes([0] * 8 + [1] * 24 + [2] * 24))
    model['tensor_files'] = {path.name: {'size_bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in weights.iterdir()}
    model['source_weights_sha256'] = 'a' * 64
    (weights / 'manifest.json').write_text(json.dumps(model))


@unittest.skipUnless(os.name == 'posix', 'POSIX supervisor required')
class SupervisorArtifactFailureTests(unittest.TestCase):
    def test_long_replay_obeys_real_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            contract = base / 'contract.json'
            contract.write_text(json.dumps({'puzzle_id': 7, 'initial_state': [0] * 120,
                'central_state': [0] * 120, 'generators': {'s': list(range(120))}}))
            payload = ('import pathlib,json; pathlib.Path("rank-status").mkdir(); '
                'pathlib.Path("rank-status/rank-0.json").write_text(json.dumps('
                '{"rank":0,"exit_code":0,"puzzle_id":7,"status":"solved"})); '
                'pathlib.Path("submit.csv").write_text("initial_state_id,path\\n7," + "s."*60000 + "s\\n")')
            result = subprocess.run([sys.executable, str(ROOT / 'tools/cube4_run_supervisor.py'),
                '--run-dir', str(base / 'run'), '--max-elapsed-seconds', '0.2',
                '--max-cost-usd', '1', '--hourly-rate-usd', '1', '--world-size', '1',
                '--puzzle-contract', str(contract), '--', sys.executable, '-c', payload],
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((base / 'run/summary.json').read_text())
            self.assertEqual(report['status'], 'timed_out')
            self.assertEqual(report['child_exit_code'], 0)
            self.assertIn('replay deadline', report['error'])
            self.assertLess(report['elapsed_seconds'], 3)
            self.assertFalse(report.get('solution_replay_valid', False))

    def test_malformed_csv_keeps_terminal_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            contract = base / 'contract.json'
            contract.write_text(json.dumps({'puzzle_id': 7, 'initial_state': [1, 0],
                'central_state': [0, 1], 'generators': {'swap': [1, 0]}}))
            payload = ('import pathlib,json; pathlib.Path("rank-status").mkdir(); '
                'pathlib.Path("rank-status/rank-0.json").write_text(json.dumps('
                '{"rank":0,"exit_code":0,"puzzle_id":7,"status":"solved"})); '
                'pathlib.Path("submit.csv").write_text("initial_state_id,path\\n7," + "s"*140000 + "\\n")')
            result = subprocess.run([sys.executable, str(ROOT / 'tools/cube4_run_supervisor.py'),
                '--run-dir', str(base / 'run'), '--max-elapsed-seconds', '10',
                '--max-cost-usd', '1', '--hourly-rate-usd', '1', '--world-size', '1',
                '--puzzle-contract', str(contract), '--', sys.executable, '-c', payload],
                capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((base / 'run/summary.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['child_exit_code'], 0)
            self.assertIn('invalid submission CSV', report['error'])
            self.assertFalse(report['production_accepted'])


@unittest.skipUnless(os.name == 'posix', 'POSIX supervisor required')
class NumericPreflightTests(unittest.TestCase):
    def check(self, skip, wrong_manifest=False, bad_state=False, rows=8, runtime=False, wrong_mode=False, wrong_cls=False, invalid_environment=None, graph=False, extra_environment=None, corrupt_before=False, mutate_during=False, mutate_comparison=None, lane_profile=False, lane_artifact='valid', loaded_view='valid', extra_config=None, fused=False, wrong_fused=False, device_fault=None, executor=None):
        graph = graph or lane_profile
        runtime = runtime or graph
        mode = 'runtime_cuda_graph' if graph else 'runtime_final_cls'
        flag = '--runtime-cuda-graph-scores' if graph else '--runtime-final-cls-scores'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fixture = base / 'fixture'
            fixture.mkdir()
            (fixture / 'weights_fp16').mkdir()
            prepared_weights(fixture / 'weights_fp16')
            if corrupt_before:
                (fixture / 'weights_fp16/output_bias.fp16').write_bytes(b'\x00\x3c' + bytes(46))
            (fixture / 'reference.json').write_text(json.dumps({
                'metadata': {'source_weights_sha256': 'a' * 64},
                'states': [[-1 if bad_state else 0] * 96 for _ in range(rows)],
                'scores_fp32': [list(range(24)) for _ in range(rows)]}))
            producer = base / 'producer'
            producer.write_text('#!' + sys.executable + '\n'
                'import json, os, pathlib, sys\n'
                'assert os.environ["BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR"]\n'
                + (f'assert sys.argv[1:] == {repr(["--require-reference", flag] + (["--verify-graph-input-reuse", "--verify-graph-job-bounds"] if graph else []) + (["--verify-production-lanes", "--probe-microbatch", "2", "--probe-lanes", "3"] if lane_profile else []))}\n'
                   'assert os.environ["BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY"] == "1"\n'
                   'assert "BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION" not in os.environ\n'
                   if runtime else 'assert sys.argv[1:] == ["--require-reference"]\n')
                + ('raise SystemExit(77)\n' if skip else
                'pathlib.Path("test_results").mkdir()\n'
                'pathlib.Path("test_results/stream1_full_scores.json").write_text(json.dumps({"scores":[list(range(24)) for _ in range(8)]}))\n'
                + ('pathlib.Path("test_results/stream1_score_execution.json").write_text(json.dumps('
                   + repr({'score_mode': ('runtime_final_cls' if not runtime else 'baseline') if wrong_mode else mode if runtime else 'baseline',
                           'rows': 8, 'microbatch': 8, 'lane': 0, 'compact57': False,
                           'fused_input_layernorm': not fused if wrong_fused else fused,
                           'cls_environment': {
                               'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': '0' if wrong_cls or not runtime else '1',
                               'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION': '0',
                               'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV': '0'}}) + '))\n')))
            if loaded_view != 'missing':
                view = dict(state_len=96, num_classes=6, num_pieces=56, max_piece_size=3,
                    seq_len=57, padded_seq_len=64, sequence_alignment=16, d_model=256,
                    nhead=8, head_dim=32, transformer_layers=4, ff_dim=1024, output_dim=24,
                    dtype=0, activation=1,
                    fp16_layouts=[dict(block=i, qkv_packed_fp16=False, ff1_packed_fp16=False,
                                      ff2_packed_fp16=False) for i in range(4)],
                    block_scales=[dict(block=i, qkv_e4m3_scale=0., ff1_e4m3_scale=0.,
                                      ff2_e4m3_scale=0.) for i in range(4)])
                if loaded_view == 'wrong_shape': view['padded_seq_len'] = 57
                if loaded_view == 'packed': view['fp16_layouts'][2]['ff1_packed_fp16'] = True
                with producer.open('a') as script:
                    script.write('pathlib.Path("test_results/stream1_loaded_view.json").write_text(json.dumps('
                                 + repr(view) + '))\n')
            if lane_profile and lane_artifact != 'missing':
                lane_scores = [list(range(24)) for _ in range(5)]
                if lane_artifact == 'last_score':
                    lane_scores[-1][-1] += 10
                raw = dict(microbatch=4 if lane_artifact == 'wrong_profile' else 2,
                           lane_count=3, parent_bases=[0, 1, 0], active_counts=[2, 1, 2], scores=lane_scores)
                with producer.open('a') as script:
                    script.write('pathlib.Path("test_results/stream1_lane_scores.json").write_text(json.dumps('
                                 + repr(raw) + '))\n')
            if mutate_during:
                with producer.open('a') as script:
                    script.write('p = pathlib.Path(os.environ["BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR"]) / "weights_fp16/output_bias.fp16"\n'
                                 'p.write_bytes(bytes([0,60]) + bytes(46))\n')
            producer.chmod(0o755)
            # Explicit synthetic CUDA mapping; not a hardware observation.
            device = dict(device=0, sm=86, uuid_hex='1' * 32)
            probe = base / 'device-probe'
            probe.write_text('#!/usr/bin/env python3\nimport json\nprint(json.dumps(' + repr(
                dict(schema_version=1, scope='observed_cuda_visible_devices_not_quality',
                     devices=[device], production_quality_accepted=False)) + '))\n')
            probe.chmod(0o755)
            with producer.open('a') as script:
                for filename, value in (
                    ('stream1_reference_device_identity.json', dict(device, schema_version=1, production_quality_accepted=False)),
                    ('stream1_reference_execution_shape.json', dict(schema_version=1,
                        scope='execution_shape_not_kernel_admission', outer_microbatch=8,
                        transformer_microbatch=8, lanes=1, device=0, sm=86, production_quality_accepted=False))):
                    script.write('pathlib.Path("test_results/' + filename + '").write_text(json.dumps(' + repr(value) + '))\n')
            command = [sys.executable, str(ROOT / 'tools/cube4_numeric_preflight.py'),
                '--device-probe', str(probe),
                '--producer', str(producer), '--reference', str(fixture / 'reference.json'),
                '--run-dir', str(base / 'run'), '--max-elapsed-seconds', '10',
                '--max-cost-usd', '1', '--hourly-rate-usd', '1']
            if device_fault:
                with producer.open('a') as script:
                    script.write('p=pathlib.Path("test_results/stream1_reference_device_identity.json")\n')
                    if device_fault == 'missing':
                        script.write('p.unlink()\n')
                    else:
                        key, value = ('uuid_hex', '2' * 32) if device_fault == 'uuid' else ('sm', 90)
                        script.write('d=json.loads(p.read_text()); d[' + repr(key) + ']=' + repr(value) + '; p.write_text(json.dumps(d))\n')
            if runtime:
                command += [flag]
            if lane_profile:
                command += ['--probe-microbatch', '2', '--probe-lanes', '3']
            for name in ('runner', 'manifest', 'config', 'hardware'):
                path = (fixture / 'weights_fp16/manifest.json'
                        if name == 'manifest' and not wrong_manifest else base / name)
                if name == 'config':
                    config = {'execution_environment': ({
                        'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': '1'} if runtime else {})
                        if invalid_environment is None else invalid_environment}
                    config.update(extra_config or {})
                    if executor is not None:
                        config['execution_environment']['BEAM_STREAM1_EXECUTOR'] = executor
                    if fused:
                        config['execution_environment']['BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM'] = '1'
                    path.write_text(json.dumps(config))
                elif name != 'manifest' or wrong_manifest:
                    path.write_bytes(b'fixture identity')
                command += ['--' + name, str(path)]
            caller_env = dict(os.environ, BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION='1')
            caller_env.update(extra_environment or {})
            if mutate_comparison:
                # Deterministic boundary injection: execute the real supervisor
                # and comparator, then mutate one input after comparison returns.
                receipt_names = {'loaded_view': 'stream1_loaded_view.json',
                    'device_identity': 'stream1_reference_device_identity.json',
                    'reference_shape': 'stream1_reference_execution_shape.json'}
                mutation = (
                    '        changed = pathlib.Path(sys.argv[sys.argv.index("--run-dir")+1]) / "test_results/' + receipt_names[mutate_comparison] + '"\n'
                    '        changed.write_bytes(changed.read_bytes() + b" ")\n'
                    if mutate_comparison in receipt_names else
                    f'        pathlib.Path(sys.argv[sys.argv.index("--{mutate_comparison}")+1]).write_bytes(b"modified after comparison")\n')
                injection = ('import sys,pathlib\n'
                    'from tools import cube4_numeric_preflight as p\n'
                    'real = p.subprocess.run\n'
                    'def run(command, **kwargs):\n'
                    '    result = real(command, **kwargs)\n'
                    '    if any(str(part).endswith("cube4_numeric_gate.py") for part in command):\n'
                    + mutation +
                    '    return result\n'
                    'p.subprocess.run = run\n'
                    'raise SystemExit(p.main())\n')
                command = [sys.executable, '-c', injection] + command[2:]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, env=caller_env)
            if extra_config:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('unbound diagnostic configuration', result.stderr)
                self.assertFalse((base / 'run').exists())
            elif device_fault:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((base / 'run/numerical.json').exists())
                self.assertFalse((base / 'run/device_observation.json').exists())
                self.assertEqual(json.loads((base / 'run/summary.json').read_text())['child_exit_code'], 0)
                if device_fault != 'missing':
                    self.assertIn('reference device does not match independent probe', result.stderr)
            elif loaded_view != 'valid':
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((base / 'run/numerical.json').exists())
                self.assertEqual(json.loads((base / 'run/summary.json').read_text())['child_exit_code'], 0)
            elif lane_profile and lane_artifact != 'valid':
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((base / 'run/numerical.json').exists())
                self.assertFalse((base / 'run/lane_numerical.json').exists())
                self.assertEqual(json.loads((base / 'run/summary.json').read_text())['child_exit_code'], 0)
            elif mutate_comparison:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('loaded view artifact changed during comparison' if mutate_comparison == 'loaded_view'
                              else 'reference device binding changed during comparison' if mutate_comparison in ('device_identity', 'reference_shape')
                              else 'inputs changed during comparison', result.stderr)
                self.assertFalse((base / 'run/numerical.json').exists())
                pending = json.loads((base / 'run/numerical.pending.json').read_text())
                self.assertTrue(pending['numerical_accepted'])
            elif corrupt_before or mutate_during:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('content hash mismatch', result.stderr)
                self.assertFalse((base / 'run/numerical.json').exists())
                if corrupt_before:
                    self.assertFalse((base / 'run').exists())
                else:
                    self.assertEqual(json.loads((base / 'run/summary.json').read_text())['child_exit_code'], 0)
            elif extra_environment:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('unbound diagnostic execution environment', result.stderr)
                self.assertFalse((base / 'run').exists())
            elif invalid_environment is not None:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid diagnostic CLS execution environment', result.stderr)
                self.assertFalse((base / 'run').exists())
            elif wrong_mode or wrong_cls or wrong_fused:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('score execution receipt mismatch', result.stderr)
                self.assertFalse((base / 'run/numerical.json').exists())
            elif rows != 8:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('requires exactly 8 reference rows', result.stderr)
                self.assertFalse((base / 'run').exists())
            elif bad_state:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('reference shape mismatch', result.stderr)
                self.assertFalse((base / 'run').exists())
            elif wrong_manifest:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('manifest does not match producer fixture', result.stderr)
                self.assertFalse((base / 'run').exists())
            elif skip:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((base / 'run/numerical.json').exists())
                record = json.loads((base / 'run/summary.json').read_text())
                self.assertEqual(record['child_exit_code'], 77)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((base / 'run/numerical.json').read_text())
                self.assertTrue(report['numerical_accepted'])
                self.assertFalse(report['production_quality_accepted'])
                self.assertEqual(report['elements_compared'], 192)
                prepared = json.loads((base / 'run/prepared_weights.json').read_text())
                expected = json.loads((fixture / 'weights_fp16/manifest.json').read_text())['tensor_files']
                self.assertEqual(len(prepared['tensor_sha256']), 60)
                self.assertEqual(prepared['tensor_sha256'],
                                 {name: record['sha256'] for name, record in expected.items()})
                self.assertFalse(prepared['loaded_tensor_identity_attested'])
                self.assertFalse((base / 'run/numerical.pending.json').exists())
                if lane_profile:
                    lane_report = json.loads((base / 'run/lane_numerical.json').read_text())
                    self.assertEqual(lane_report['status'], 'pass')
                    self.assertEqual(lane_report['elements_compared'], 120)
                    self.assertFalse(lane_report['production_quality_accepted'])

    def test_live_producer_is_compared(self):
        self.check(False)

    def test_unbound_config_fields_cannot_be_silently_hashed_as_execution(self):
        for runtime in (False, True):
            with self.subTest(runtime=runtime):
                self.check(False, runtime=runtime, extra_config={'b_micro': 1024})

    def test_missing_or_unbound_loaded_view_blocks_numeric_publication(self):
        for view in ('missing', 'wrong_shape', 'packed'):
            with self.subTest(view=view):
                self.check(False, loaded_view=view)

    def test_requested_lane_profile_reaches_producer_and_full_comparison(self):
        self.check(False, lane_profile=True)

    def test_fused_input_declared_and_receipt_bound(self):
        self.check(False, lane_profile=True, fused=True)

    def test_wrong_fused_input_receipt_rejected(self):
        self.check(False, lane_profile=True, fused=True, wrong_fused=True)

    def test_baseline_scores_cannot_hide_invalid_lane_scores(self):
        for artifact in ('missing', 'wrong_profile', 'last_score'):
            with self.subTest(artifact=artifact):
                self.check(False, lane_profile=True, lane_artifact=artifact)

    def test_corrupt_tensor_rejected_before_producer(self):
        self.check(False, corrupt_before=True)

    def test_tensor_mutation_after_successful_producer_blocks_numerical_acceptance(self):
        self.check(False, mutate_during=True)

    def test_successful_comparison_cannot_admit_changed_inputs(self):
        for name in ('runner', 'producer', 'reference', 'config', 'hardware'):
            with self.subTest(input=name):
                self.check(False, mutate_comparison=name)

    def test_successful_scores_cannot_admit_replaced_loaded_view(self):
        self.check(False, mutate_comparison='loaded_view')

    def test_successful_comparison_cannot_admit_replaced_device_receipts(self):
        for name in ('device_identity', 'reference_shape'):
            with self.subTest(receipt=name):
                self.check(False, graph=True, mutate_comparison=name)

    def test_runtime_receipt_cannot_satisfy_baseline_request(self):
        self.check(False, wrong_mode=True)

    def test_graph_mode_reaches_bounded_producer(self):
        self.check(False, graph=True)

    def test_wrong_device_cannot_publish_numerical_acceptance(self):
        for fault in ('uuid', 'sm', 'missing'):
            with self.subTest(fault=fault):
                self.check(False, graph=True, device_fault=fault)

    def test_unbound_kernel_selectors_rejected_before_producer(self):
        for flag in ('BEAM_STREAM1_TRANSFORMER_COMPACT57',
                     'BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM',
                     'BEAM_STREAM1_TRANSFORMER_FUTURE_SELECTOR'):
            with self.subTest(flag=flag):
                self.check(False, graph=True, extra_environment={flag: '1'})

    def test_baseline_rejects_unbound_kernel_selector_before_producer(self):
        self.check(False, extra_environment={'BEAM_STREAM1_TRANSFORMER_COMPACT57': '1'})

    def test_eager_receipt_cannot_satisfy_graph_request(self):
        self.check(False, graph=True, wrong_mode=True)

    def test_runtime_score_mode_reaches_bounded_producer(self):
        self.check(False, runtime=True)

    def test_explicit_executor_matches_actual_diagnostic_mode(self):
        self.check(False, graph=True, executor='native_cuda_graph')
        self.check(False, runtime=True, executor='native_eager')

    def test_baseline_receipt_cannot_satisfy_runtime_request(self):
        self.check(False, runtime=True, wrong_mode=True)

    def test_wrong_cls_value_cannot_satisfy_runtime_request(self):
        self.check(False, runtime=True, wrong_cls=True)

    def test_invalid_execution_environment_rejected_before_producer(self):
        for value in ({'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': True},
                      {'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': '1', 'BEAM_STREAM1_EXECUTOR': 'native_cuda_graph'},
                      {'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': '1', 'BEAM_STREAM1_EXECUTOR': 'libtorch_eager'},
                      {'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': '1', 'BEAM_STREAM1_EXECUTOR': None},
                      {'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': 'yes'},
                      {'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': '1', 'LD_PRELOAD': '/untrusted'},
                      {}):
            with self.subTest(environment=value):
                self.check(False, runtime=True, invalid_environment=value)

    def test_skip_exit_cannot_be_promoted(self):
        self.check(True)

    def test_unrelated_manifest_rejected_before_producer(self):
        self.check(False, wrong_manifest=True)

    def test_invalid_reference_rejected_before_gpu_producer(self):
        self.check(False, bad_state=True)

    def test_unsupported_row_count_rejected_before_producer(self):
        self.check(False, rows=7)
