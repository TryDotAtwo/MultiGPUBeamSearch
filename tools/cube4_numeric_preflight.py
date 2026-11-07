"""Run a fresh bounded native score producer, then the diagnostic numeric gate.

POSIX only. A pass is not profile promotion, solve quality, or billing closure.
The reference directory must already contain matching exported model weights.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.full_score_comparison import observe_score_identity, compare_graph_lane_scores
from tools.cube4_numeric_gate import read_json, validate_reference, validate_checkpoint_binding
from tools.cube4_bundle_contract import validate_cube4_weights
from tools.loaded_view_contract import validate_cube4_diagnostic_view, validate_reference_device_binding
from tools.cuda_device_observation import observe as observe_devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner', 'producer', 'manifest', 'reference', 'config', 'hardware', 'run-dir', 'device-probe'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('max-elapsed-seconds', 'max-cost-usd', 'hourly-rate-usd'):
        parser.add_argument('--' + name, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--runtime-final-cls-scores', action='store_true',
                        help='preserve producer FINAL_CLS_ONLY environment; diagnostic eight-row mode')
    modes.add_argument('--runtime-cuda-graph-scores', action='store_true',
                       help='capture/replay native graph; diagnostic eight-row, lane-zero mode')
    parser.add_argument('--probe-microbatch', type=int)
    parser.add_argument('--probe-lanes', type=int)
    args = parser.parse_args()
    runtime_scores = args.runtime_final_cls_scores or args.runtime_cuda_graph_scores
    if os.name != 'posix':
        parser.error('POSIX producer supervision required')
    root = Path(__file__).resolve().parent
    try:
        lane_profile = args.probe_microbatch is not None or args.probe_lanes is not None
        if lane_profile and (not args.runtime_cuda_graph_scores or
                             args.probe_microbatch is None or args.probe_lanes is None or
                             not 1 <= args.probe_microbatch <= 65536 or
                             not 1 <= args.probe_lanes <= 64):
            raise ValueError('lane profile requires graph mode and bounded microbatch/lanes')
        for name in ('runner', 'producer', 'manifest', 'reference', 'config', 'hardware', 'run_dir', 'device_probe'):
            setattr(args, name, getattr(args, name).resolve())
        if args.reference.name != 'reference.json':
            raise ValueError('producer requires reference.json in prepared fixture directory')
        if args.manifest != (args.reference.parent / 'weights_fp16/manifest.json').resolve():
            raise ValueError('manifest does not match producer fixture')
        reference, _ = read_json(args.reference)
        validate_reference(reference)
        manifest, _ = read_json(args.manifest)
        validate_checkpoint_binding(reference, manifest)
        # The current native baseline producer hard-codes eight rows. Reject
        # unsupported fixtures before allocation rather than letting its numeric
        # scanner consume subsequent JSON arrays as missing state values.
        if len(reference['states']) != 8:
            raise ValueError('baseline producer requires exactly 8 reference rows')
        observed = observe_score_identity(args.runner, args.manifest, args.reference,
                                         args.config, args.hardware)
        producer_identity = observe_score_identity(args.producer, args.manifest, args.reference,
                                                  args.config, args.hardware)
        device_observation = observe_devices(args.device_probe)
        command = [sys.executable, str(root / 'cube4_run_supervisor.py'),
                   '--run-dir', str(args.run_dir)]
        for name in ('max_elapsed_seconds', 'max_cost_usd', 'hourly_rate_usd'):
            command += ['--' + name.replace('_', '-'), getattr(args, name)]
        env = dict(os.environ, BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR=str(args.reference.parent))
        supported = {'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY',
                     'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION',
                     'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV'}
        fused_selector = 'BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM'
        compact_selector = 'BEAM_STREAM1_TRANSFORMER_COMPACT57'
        bound_selectors = supported | {fused_selector, compact_selector}
        config, _ = read_json(args.config)
        if set(config) != {'execution_environment'}:
            raise ValueError('unbound diagnostic configuration: only execution_environment is supported')
        executor_selector = 'BEAM_STREAM1_EXECUTOR'
        execution_config = config['execution_environment']
        requested_executor = (execution_config.get(executor_selector)
                              if isinstance(execution_config, dict) else None)
        if isinstance(execution_config, dict) and executor_selector in execution_config:
            expected_executor = 'native_cuda_graph' if args.runtime_cuda_graph_scores else 'native_eager'
            if not runtime_scores or requested_executor != expected_executor:
                raise ValueError('invalid diagnostic CLS execution environment: executor differs from actual mode')
        if executor_selector in env and env[executor_selector] != requested_executor:
            raise ValueError('unbound diagnostic execution environment: ' + executor_selector)
        # CLS/fused/compact selectors plus explicitly matched native executor.
        permitted = bound_selectors | {'BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR'}
        if requested_executor is not None:
            permitted.add(executor_selector)
        unbound = sorted(key for key in env
                         if (key.startswith('BEAM_STREAM1_') and key not in permitted)
                         or key == 'BEAM_HOPPER_NATIVE_FP8')
        if unbound:
            raise ValueError('unbound diagnostic execution environment: ' + ', '.join(unbound))
        for selector in (fused_selector, compact_selector):
            if (selector in env and (not runtime_scores or
                    not isinstance(config['execution_environment'], dict) or
                    selector not in config['execution_environment'])):
                raise ValueError('unbound diagnostic execution environment: ' + selector)
        if runtime_scores:
            execution = config.get('execution_environment')
            # This diagnostic receipt cannot bind other kernel-policy flags.
            # Fail closed instead of silently running inherited selectors and
            # presenting their scores as the narrower declared configuration.
            if (not isinstance(execution, dict) or
                'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY' not in execution or
                not set(execution) <= bound_selectors | {executor_selector} or
                any(type(value) is not str or value not in ('0', '1')
                    for key, value in execution.items() if key != executor_selector)):
                raise ValueError('invalid diagnostic CLS execution environment')
            # Diagnostic CLS/fused/compact and executor bindings. Other profile flags/batch/lane
            # settings remain unbound; this is not a full execution contract.
            for key in bound_selectors:
                env.pop(key, None)
            env.update(execution)
        else:
            if config['execution_environment'] != {}:
                raise ValueError('unbound diagnostic configuration: baseline requires empty execution_environment')
            # Baseline is canonical full-sequence inference, not inherited CLS.
            for key in bound_selectors:
                env.pop(key, None)
        producer_args = ['--', str(args.producer), '--require-reference']
        if runtime_scores:
            producer_args += ['--runtime-cuda-graph-scores' if args.runtime_cuda_graph_scores
                              else '--runtime-final-cls-scores']
        if args.runtime_cuda_graph_scores:
            # Require real graph metadata/input-reuse checks, not only the
            # final eight-row dump. These remain bounded diagnostic fixtures.
            producer_args += ['--verify-graph-input-reuse', '--verify-graph-job-bounds']
        if lane_profile:
            producer_args += ['--verify-production-lanes', '--probe-microbatch', str(args.probe_microbatch),
                             '--probe-lanes', str(args.probe_lanes)]
        prepared = validate_cube4_weights(args.manifest.parent)
        result = subprocess.run(command + producer_args, env=env)
        if result.returncode:
            return result.returncode
        if validate_cube4_weights(args.manifest.parent) != prepared:
            raise ValueError('prepared weights changed during execution')
        loaded_view_path = args.run_dir / 'test_results/stream1_loaded_view.json'
        loaded_view, loaded_view_hash = read_json(loaded_view_path)
        validate_cube4_diagnostic_view(loaded_view, compact57=env.get(compact_selector, '0') == '1')
        device_identity_path = args.run_dir / 'test_results/stream1_reference_device_identity.json'
        shape_path = args.run_dir / 'test_results/stream1_reference_execution_shape.json'
        device_identity, device_identity_hash = read_json(device_identity_path)
        shape, shape_hash = read_json(shape_path)
        validate_reference_device_binding(device_observation['visible_devices'], device_identity, shape)
        if observe_devices(args.device_probe) != device_observation:
            raise ValueError('CUDA device mapping changed during producer execution')
        receipt, _ = read_json(args.run_dir / 'test_results/stream1_score_execution.json')
        expected_receipt = {'score_mode': 'runtime_cuda_graph' if args.runtime_cuda_graph_scores
                            else 'runtime_final_cls' if runtime_scores else 'baseline', 'rows': 8,
                            'microbatch': 8, 'lane': 0,
                            'cls_environment': {key: env.get(key, '0') for key in supported},
                            'fused_input_layernorm': env.get(fused_selector, '0') == '1',
                            'compact57': env.get(compact_selector, '0') == '1'}
        if (receipt != expected_receipt or
            any(type(receipt.get(key)) is not int for key in ('rows', 'microbatch', 'lane')) or
            type(receipt.get('fused_input_layernorm')) is not bool or
            type(receipt.get('compact57')) is not bool):
            raise ValueError('score execution receipt mismatch')
        if (observed != observe_score_identity(args.runner, args.manifest, args.reference,
                                              args.config, args.hardware) or
            producer_identity != observe_score_identity(args.producer, args.manifest, args.reference,
                                                       args.config, args.hardware)):
            raise ValueError('producer or inputs changed during execution')
        lane_report = None
        if lane_profile:
            lane_path = args.run_dir / 'test_results/stream1_lane_scores.json'
            lane_actual, lane_hash = read_json(lane_path)
            lane_report = compare_graph_lane_scores(reference['scores_fp32'], lane_actual,
                microbatch=args.probe_microbatch, lanes=args.probe_lanes)
            if lane_report['status'] != 'pass':
                raise ValueError('requested graph lane numerical comparison failed')
            lane_report.update(actual_scores_sha256=lane_hash,
                               requested_microbatch=args.probe_microbatch,
                               requested_lanes=args.probe_lanes,
                               provenance_attested=False)
        gate = [sys.executable, str(root / 'cube4_numeric_gate.py'), '--actual',
                str(args.run_dir / 'test_results/stream1_full_scores.json')]
        for name in ('runner', 'producer', 'manifest', 'reference', 'config', 'hardware'):
            gate += ['--' + name, str(getattr(args, name))]
        # Do not publish an accepted numerical artifact before the final tensor
        # check. This detects persistent mutation, not transient modify/restore
        # or the identity of GPU-resident tensors; those need execution binding.
        pending = args.run_dir / 'numerical.pending.json'
        with pending.open('x') as report:
            gate_result = subprocess.run(gate, stdout=report).returncode
        if validate_cube4_weights(args.manifest.parent) != prepared:
            raise ValueError('prepared weights changed during comparison')
        if (observed != observe_score_identity(args.runner, args.manifest, args.reference,
                                              args.config, args.hardware) or
            producer_identity != observe_score_identity(args.producer, args.manifest, args.reference,
                                                       args.config, args.hardware)):
            raise ValueError('producer or inputs changed during comparison')
        if lane_profile and read_json(lane_path)[1] != lane_hash:
            raise ValueError('graph lane score artifact changed during comparison')
        if read_json(loaded_view_path)[1] != loaded_view_hash:
            raise ValueError('loaded view artifact changed during comparison')
        if (read_json(device_identity_path)[1] != device_identity_hash or
                read_json(shape_path)[1] != shape_hash or
                observe_devices(args.device_probe) != device_observation):
            raise ValueError('reference device binding changed during comparison')
        with (args.run_dir / 'device_observation.json').open('x') as device_file:
            json.dump(device_observation, device_file, indent=2)
        with (args.run_dir / 'prepared_weights.json').open('x') as receipt_file:
            json.dump({'scope': 'observed prepared tensor files before/after producer and comparison',
                       'model_manifest_sha256': observed['model_manifest_sha256'],
                       'tensor_sha256': prepared['tensor_sha256'],
                       'loaded_tensor_identity_attested': False}, receipt_file, indent=2)
        if lane_report is not None:
            with (args.run_dir / 'lane_numerical.json').open('x') as report:
                json.dump(lane_report, report, indent=2, allow_nan=False)
        pending.rename(args.run_dir / 'numerical.json')
        return gate_result
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('numeric preflight rejected: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
