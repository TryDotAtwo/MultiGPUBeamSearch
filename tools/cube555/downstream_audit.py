"""Coordinate sweep of Stream3/4 with frozen Stream1 and beam semantics."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
from types import SimpleNamespace

from tools.cube555.run import configuration, history_budgets, runtime_plan, _available_ram_bytes
from tools.cayleypy_public.profile import derive_runtime


def downstream_plan(base, overrides):
    allowed = {'stream3_ring_slots', 'stream4_batch_candidates',
               'stream4_trigger_candidates', 'shard_count', 'stream4_active_sort_slots'}
    if set(overrides) - allowed:
        raise ValueError('downstream overrides cannot change Stream1 or semantic capacity')
    runtime = {**base.runtime, **overrides}
    if runtime['stream3_ring_slots'] < runtime['stream1_concurrency']:
        raise ValueError('ring slots must cover inference lanes')
    plan = derive_runtime(dict(runtime=runtime, model_class=base.model_class,
        profile_power=base.profile_power, validation_status='bounded_from_measured'),
        base.requested_beam, 30, 30, base.world_size, allow_single_rank=True)
    if plan.effective_beam != base.effective_beam:
        raise ValueError('diagnostic overrides must preserve effective beam')
    return replace(plan, runtime={**plan.runtime, 'model_micro': base.runtime['model_micro']},
                   cross_puzzle_profile_note='experimental_cube555_downstream')


def saturated_layers(folder, local_beam, required_layers=4, world_size=2):
    ranks = []
    for rank in range(world_size):
        paths = list(Path(folder).rglob(f'rank-{rank}.log'))
        if len(paths) != 1:
            raise ValueError(f'expected exactly one native rank-{rank} log')
        text = paths[0].read_text()
        if 'collection_truncated=1' in text:
            raise ValueError('collection overflow cannot be a passing timing row')
        starts = {int(d): int(n) for d, n in re.findall(
            r'depth_start=(\d+) frontier_size=(\d+)', text)}
        ends = {int(d): float(t) for d, t in re.findall(
            r'depth_done=(\d+) depth_sec=([\d.e+-]+)', text)}
        ranks.append((starts, ends))
    layers = [dict(depth=d, seconds=max(rank[1][d] for rank in ranks),
                   rank_seconds=[rank[1][d] for rank in ranks])
        for d in sorted(set.intersection(*(set(rank[1]) for rank in ranks)))
        if all(rank[0].get(d) == local_beam for rank in ranks)]
    if len(layers) < required_layers:
        raise ValueError(f'insufficient both-rank steady-state evidence: {layers}')
    return layers


def paired_comparison(before, candidate, after):
    """Only resolve a gain larger than surrounding identical-base drift."""
    if any(row['status'] != 'complete' for row in (before, candidate, after)):
        return dict(resolved=False, reason='unstable_or_failed_block')
    base_seconds = (before['median_seconds'] + after['median_seconds']) / 2
    drift = abs(before['median_seconds'] - after['median_seconds']) / base_seconds
    gain = 1 - candidate['median_seconds'] / base_seconds
    return dict(resolved=gain > drift, gain_fraction=gain,
                baseline_drift_fraction=drift, baseline_seconds=base_seconds,
                reason='gain_exceeds_drift' if gain > drift else 'gain_not_larger_than_drift')


def main():
    from tools.cube555.export import export_blend
    from tools.cube555.smoke import make_fixture
    from tools.cube555.telemetry import Telemetry
    from tools.cayleypy_public.data import load_puzzle_contract
    from tools.cayleypy_public.model import ExportedModel
    from tools.run_cayleypy_public import (
        validate_t4_hardware, locate_or_build_runner,
        _run_with_history_budgets, _materialize_run_artifacts,
    )
    import torch
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--competition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-micro', type=int, required=True)
    parser.add_argument('--inference-concurrency', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--neighbor-micro', type=int, required=True)
    parser.add_argument('--runtime-target', choices=['kaggle-2xt4', 'molab-single-gpu'],
                        default='kaggle-2xt4')
    parser.add_argument('--beam', type=int, default=2**20)
    parser.add_argument('--saturated-depths', type=int, choices=(2, 4), default=2,
                        help='equal saturated work in every paired block; default two depths')
    parser.add_argument('--control-parent-groups', type=int, default=32)
    parser.add_argument('--variants', nargs='+', default=['batch-half', 'batch-double',
        'ring-four', 'shards-two', 'shards-eight', 'sort-one'])
    args = parser.parse_args()
    if (args.control_parent_groups % args.inference_concurrency
            or args.control_parent_groups//args.inference_concurrency < 6) or not (
            8 <= args.control_parent_groups <= 128):
        parser.error('--control-parent-groups must be within[8,128] and divide across lanes')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (args.output/'run_summary.json').write_text('{"status":"preparing"}')
    preparation_monitor = Telemetry(args.output)
    preparation_monitor.start()
    info = args.assets/'puzzle_info.json'
    manifest = export_blend(args.assets/'q555_f1_bell2k.pt', args.assets/'q555_2k_BEST.pt',
        args.assets/'piece_layout_555.json', info, args.output/'export')
    model = ExportedModel('cube555-q-blend', 'fp16', manifest['script_sha256'], manifest,
                          'piece_transformer')
    world_size, cuda_arch = 2, 75
    if args.runtime_target == 'molab-single-gpu':
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Molab audit requires one real GPU')
        world_size = 1
        major, minor = torch.cuda.get_device_capability(0)
        cuda_arch = 10 * major + minor
        gpus = [torch.cuda.get_device_name(0)]
    else:
        gpus = validate_t4_hardware()
    cfg = SimpleNamespace(assets=args.assets, competition=args.competition, beam=args.beam,
        depth=140, touch_radius=5, solution_mode='collect',
        collect_until_depth=(6 if args.beam > 4_000_000 else 5) + args.saturated_depths,
        max_collected_solutions=2000)
    base = runtime_plan(cfg.beam, model_micro=args.model_micro,
                        inference_concurrency=args.inference_concurrency, world_size=world_size)
    runner = locate_or_build_runner(args.output, info, backend='piece_transformer',
                                    config=configuration(cfg, 1020, info), cuda_arch=cuda_arch)
    subprocess.run(['cmake', '--build', str(runner.parent), '--target',
                    'cube555_stream1_benchmark', '-j', '2'], check=True)
    binary = runner.parent/'cube555_stream1_benchmark'
    moves = np.asarray(list(json.loads(info.read_text())['generators'].values()))
    rng = np.random.default_rng(555)
    state = np.arange(150, dtype=np.uint8)
    corpus = np.zeros((8192, 160), dtype=np.uint8)
    for i in range(8192):
        state = state[moves[rng.integers(len(moves))]]
        corpus[i, :150] = state
    corpus_path = args.output/'parents.u8'
    corpus.tofile(corpus_path)
    preparation_seconds = time.monotonic()-started
    preparation_performance = preparation_monitor.finish()
    contract = load_puzzle_contract(info, args.competition/'test.csv',
                                   args.competition/'sample_submission.csv', 1020, 1020)
    fixture = args.output/'fixture'
    make_fixture(args.assets, fixture)
    smoke_contract = load_puzzle_contract(info, fixture/'test.csv',
                                          fixture/'sample_submission.csv', 3, 3)
    smoke_cfg = SimpleNamespace(**{**vars(cfg), 'competition': fixture, 'depth': 4,
        'touch_radius': 0, 'solution_mode': 'first', 'collect_until_depth': 0})
    report = dict(gpus=gpus, manifest=manifest, rows=[],
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        preparation_seconds=preparation_seconds, preparation_performance=preparation_performance,
        frozen_stream1=dict(model_micro=args.model_micro, concurrency=args.inference_concurrency),
        metric='minimum median(max(rank0,rank1) saturated step seconds) at fixed beam',
        status='running')
    def save():
        report['wall_seconds'] = time.monotonic()-started
        (args.output/'downstream_report.json').write_text(json.dumps(report, indent=2))
    def isolated(plan, out, label):
        processes = []
        try:
            for gpu in range(world_size):
                log = (out/f'isolated-{label}-gpu-{gpu}.log').open('w')
                proc = subprocess.Popen([str(binary), str(args.output/'export'),
                    str(corpus_path), str(plan.runtime['model_micro']), str(gpu),
                    str(args.control_parent_groups//plan.runtime['stream1_concurrency']),
                    str(plan.runtime['stream1_concurrency'])], stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
                processes.append((gpu, proc, log))
            rows = []
            for gpu, proc, log in processes:
                if proc.wait(timeout=300) != 0:
                    raise RuntimeError(f'bracketed isolated benchmark failed on GPU{gpu}')
                log.close()
                result = json.loads((out/f'isolated-{label}-gpu-{gpu}.log').read_text().splitlines()[-1])
                steady = result['seconds'][len(result['seconds'])//2:]
                result['steady_median_seconds'] = statistics.median(steady)
                result['steady_spread_fraction'] = (max(steady)-min(steady))/result['steady_median_seconds']
                result['parents_per_second'] = (8192*plan.runtime['stream1_concurrency']
                                               /result['steady_median_seconds'])
                rows.append(result)
            return dict(rows=rows, stable=all(row['steady_spread_fraction']<=0.10 for row in rows),
                balanced_parents_per_second=world_size*min(
                row['parents_per_second'] for row in rows))
        finally:
            for _, proc, log in processes:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                log.close()
    def probe(name, plan):
        out = args.output/name
        out.mkdir()
        row = dict(name=name, plan=asdict(plan), status='running')
        report['rows'].append(row)
        save()
        print(f'downstream_start {name} runtime={dict(plan.runtime)}', flush=True)
        (out/'run_summary.json').write_text('{"status":"running"}')
        monitor = Telemetry(out)
        monitor.start()
        start = time.monotonic()
        try:
            ram, disk = history_budgets(_available_ram_bytes(), shutil.disk_usage('/tmp').free)
            smoke = _run_with_history_budgets(ram, disk, configuration(smoke_cfg, 3, info),
                smoke_contract, model, plan, args.output/'export', out/'smoke-logs',
                runner_path=str(runner))
            row['smoke_result'] = _materialize_run_artifacts(smoke, out/'smoke')
            if not smoke.solution_records:
                raise ValueError('four-move synthetic scramble must solve and replay')
            row['isolated_before'] = isolated(plan, out, 'before')
            artifacts = _run_with_history_budgets(ram, disk, configuration(cfg, 1020, info),
                contract, model, plan, args.output/'export', out/'logs', runner_path=str(runner))
            row['result'] = _materialize_run_artifacts(artifacts, out)
            row['layers'] = saturated_layers(out/'logs', plan.local_beam, args.saturated_depths,
                                             world_size=world_size)
            seconds = [layer['seconds'] for layer in row['layers']]
            row['median_seconds'] = statistics.median(seconds)
            row['relative_spread'] = (max(seconds)-min(seconds))/row['median_seconds']
            row['parents_per_second'] = plan.effective_beam/row['median_seconds']
            row['isolated_after'] = isolated(plan, out, 'after')
            before = row['isolated_before']['balanced_parents_per_second']
            after = row['isolated_after']['balanced_parents_per_second']
            row['isolated_drift_fraction'] = abs(after-before)/((after+before)/2)
            row['overhead_reliable'] = (row['isolated_drift_fraction'] <= 0.05
                and row['isolated_before']['stable'] and row['isolated_after']['stable'])
            row['bracketed_overhead_fraction'] = 1-row['parents_per_second']/((before+after)/2)
            row['status'] = ('complete' if row['relative_spread'] <= 0.10
                             and row['overhead_reliable'] else 'unstable')
            row['comparison_exclusions'] = []
            if row['relative_spread'] > 0.10:
                row['comparison_exclusions'].append('saturated_layer_spread_exceeds_10_percent')
            if not row['overhead_reliable']:
                row['comparison_exclusions'].append('isolated_hardware_drift_exceeds_5_percent')
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
        finally:
            row['wall_seconds'] = time.monotonic()-start
            row['performance'] = monitor.finish()
            save()
            print('downstream_result '+json.dumps(row), flush=True)
        return row
    # Paired fixed-base blocks; never select by unpaired chronological minima.
    variants = [
        ('batch-half', dict(stream4_batch_candidates=65536, stream4_trigger_candidates=65536)),
        ('batch-double', dict(stream4_batch_candidates=262144, stream4_trigger_candidates=262144)),
        ('ring-four', dict(stream3_ring_slots=max(4, args.inference_concurrency))),
        ('shards-two', dict(shard_count=2)),
        ('shards-two-batch-double', dict(shard_count=2,
            stream4_batch_candidates=262144, stream4_trigger_candidates=262144)),
        ('shards-eight', dict(shard_count=8)),
        ('sort-one', dict(stream4_active_sort_slots=1)),
    ]
    report['paired_blocks'] = []
    leaders = []
    if set(args.variants) - {name for name, _ in variants}:
        raise ValueError('unknown downstream variant')
    for name, overrides in variants:
        if name not in args.variants:
            continue
        if time.monotonic()-started > 4400:
            raise TimeoutError('bounded downstream audit budget exhausted')
        candidate = downstream_plan(base, overrides)
        before = probe(name+'-base-before', base)
        row = probe(name, candidate)
        after = probe(name+'-base-after', base)
        comparison = dict(name=name, **paired_comparison(before, row, after))
        report['paired_blocks'].append(comparison)
        if comparison['resolved']:
            leaders.append((name, candidate, comparison))
        save()
    confirmed = []
    for name, candidate, original in reversed(leaders):
        if time.monotonic()-started > 4400:
            raise TimeoutError('reverse-order confirmation budget exhausted')
        before = probe(name+'-reverse-base-before', base)
        row = probe(name+'-reverse', candidate)
        after = probe(name+'-reverse-base-after', base)
        comparison = dict(name=name+'-reverse', **paired_comparison(before, row, after))
        report['paired_blocks'].append(comparison)
        if comparison['resolved']:
            confirmed.append((min(original['gain_fraction'], comparison['gain_fraction']), name, candidate))
        save()
    if not confirmed:
        report['downstream_selection'] = 'baseline_retained_no_confirmed_gain'
        best_name, best_plan = 'baseline', base
        save()
    else:
        report['downstream_selection'] = 'confirmed_gain'
        _, best_name, best_plan = max(confirmed, key=lambda item: item[0])
    report['best_fixed_stream1'] = best_name
    neighbor_base = runtime_plan(cfg.beam, model_micro=args.neighbor_micro,
                                inference_concurrency=args.inference_concurrency,
                                world_size=world_size)
    neighbor = downstream_plan(neighbor_base, {k: best_plan.runtime[k] for k in (
        'stream3_ring_slots', 'stream4_batch_candidates', 'stream4_trigger_candidates',
        'shard_count', 'stream4_active_sort_slots')})
    before = probe('neighbor-base-before', best_plan)
    neighbor_row = probe('neighbor-stream1', neighbor)
    after = probe('neighbor-base-after', best_plan)
    comparison = dict(name='neighbor-stream1', **paired_comparison(before, neighbor_row, after))
    report['paired_blocks'].append(comparison)
    if comparison['resolved']:
        reverse_before = probe('neighbor-reverse-base-before', best_plan)
        reverse_row = probe('neighbor-reverse', neighbor)
        reverse_after = probe('neighbor-reverse-base-after', best_plan)
        reverse = dict(name='neighbor-reverse',
                       **paired_comparison(reverse_before, reverse_row, reverse_after))
        report['paired_blocks'].append(reverse)
        report['winner'] = 'neighbor-stream1' if reverse['resolved'] else best_name
    else:
        report['winner'] = best_name
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
