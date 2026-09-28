"""Two-T4 Cube555 launcher using the existing native distributed beam solver."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.data import load_puzzle_contract
from tools.cayleypy_public.model import ExportedModel
from tools.cayleypy_public.profile import RuntimePlan
from tools.cayleypy_public.runner import PublicSearchRunError
from tools.run_cayleypy_public import (
    validate_t4_hardware, locate_or_build_runner, _available_ram_bytes,
    _run_with_history_budgets, _materialize_run_artifacts,
)
from tools.cube555.export import export_blend


def runtime_plan(beam: int) -> RuntimePlan:
    if type(beam) is not int or not 2 <= beam <= 2**24:
        raise ValueError('beam must be in [2, 2**24]; memory preflight may reject it')
    effective = ((beam + 7) // 8) * 8
    local = effective // 2
    micro, slots, batch = 128, 4, 16384
    capacity = max((local * 105 + 399) // 400, micro * 30 * slots + 2 * batch)
    capacity = ((capacity + 1023) // 1024) * 1024
    return RuntimePlan(
        requested_beam=beam, effective_beam=effective, alignment_delta=effective - beam,
        profile_power=beam.bit_length() - 1, model_class='output_move_count',
        local_beam=local, parent_batch=micro, stream3_batch_candidates=micro * 30 * slots,
        shard_capacity_candidates=capacity,
        runtime=dict(b_micro=micro, stream1_concurrency=1, stream3_ring_slots=slots,
                     shard_count=4, shard_capacity_scale_ppm=1050000,
                     stream4_batch_candidates=batch, stream4_trigger_candidates=batch,
                     stream4_active_sort_slots=1, final_materialize_chunk_candidates=16384),
        cross_puzzle_profile_note='Cube555 experimental profile; no Cube4 capacity claim',
    )


def configuration(args, pid, puzzle_info):
    return PublicRunConfig.from_mapping(dict(
        author_name='Ivan Litvak', checkpoint_path=str(args.assets / 'q555_f1_bell2k.pt'),
        puzzle_info_json=str(puzzle_info), test_csv=str(args.competition / 'test.csv'),
        sample_submission_csv=str(args.competition / 'sample_submission.csv'),
        puzzle_id_start=pid, puzzle_id_end=pid, beam_width=args.beam, max_depth=args.depth,
        reflect_mode='off', reflect_source_csv=None, solution_mode='first', collect_until_depth=args.depth,
        max_collected_solutions=1, touch_bfs_radius=args.touch_radius,
        publish_results=False, results_ingest_url='', enable_debug=True,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--competition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pids', type=int, nargs='+', default=[1020, 1034])
    parser.add_argument('--beam', type=int, default=65536)
    parser.add_argument('--depth', type=int, default=200)
    parser.add_argument('--touch-radius', type=int, default=2)
    parser.add_argument('--transformer-weight', type=float, default=0.8)
    args = parser.parse_args()
    args.assets, args.competition, args.output = args.assets.resolve(), args.competition.resolve(), args.output.resolve()
    if args.output.exists():
        raise ValueError('choose a new output directory; previous results are never overwritten')
    args.output.mkdir(parents=True)
    start = time.monotonic()
    summary = dict(status='preflight', pids=args.pids, beam=args.beam, results=[])
    summary_path = args.output / 'run_summary.json'
    def save():
        summary['wall_seconds'] = time.monotonic() - start
        summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    try:
        summary['gpus'] = validate_t4_hardware()
        info = args.assets / 'puzzle_info.json'
        contracts = {}
        for pid in args.pids:
            contracts[pid] = load_puzzle_contract(info, args.competition / 'test.csv',
                args.competition / 'sample_submission.csv', pid, pid)
            if any(sorted(s) != list(range(150)) for s in contracts[pid].initial_states.values()):
                raise ValueError('Cube555 initial states must be permutations of 0..149')
        config = configuration(args, args.pids[0], info)
        export_dir = args.output / 'export'
        manifest = export_blend(args.assets / 'q555_f1_bell2k.pt', args.assets / 'q555_2k_BEST.pt',
            args.assets / 'piece_layout_555.json', info, export_dir, args.transformer_weight)
        model = ExportedModel('cube555-q-blend', 'fp16', manifest['checkpoint_sha256'][0], manifest, 'piece_transformer')
        plan = runtime_plan(args.beam)
        # Bounded total host budgets (both ranks); exact history check stays in runner.
        available = _available_ram_bytes()
        ram = min(8 * 1024**3, available - 4 * 1024**3)
        disk = min(16 * 1024**3, shutil.disk_usage('/tmp').free - 4 * 1024**3)
        if min(ram, disk) < 512 * 1024**2:
            raise ValueError('insufficient host RAM or scratch disk for the two-rank run')
        preflight = dict(plan=asdict(plan), state_len=150, state_storage_len=160,
            state_value_pad=256, history_ram_bytes=ram, history_disk_bytes=disk,
            profile_status='experimental', checkpoint_sha256=manifest['checkpoint_sha256'])
        (args.output / 'preflight.json').write_text(json.dumps(preflight, indent=2), encoding='utf-8')
        print(json.dumps(preflight, indent=2), flush=True)
        summary['status'] = 'building'
        save()
        runner = locate_or_build_runner(args.output, info, backend='piece_transformer', config=config)
        merged = contracts[args.pids[0]].sample_submission.copy()
        for pid in args.pids:
            summary['status'], summary['current_pid'] = 'running', pid
            save()
            out = args.output / f'puzzle-{pid}'
            out.mkdir()
            try:
                artifacts = _run_with_history_budgets(ram, disk, configuration(args, pid, info),
                    contracts[pid], model, plan, export_dir, out / 'logs', runner_path=str(runner))
            except PublicSearchRunError as error:
                _materialize_run_artifacts(error.partial_artifacts, out)
                raise
            result = _materialize_run_artifacts(artifacts, out)
            summary['results'].append(dict(pid=pid, **result))
            selected = artifacts.submission.loc[artifacts.submission.initial_state_id == pid]
            for column in merged.columns:
                if column != 'initial_state_id':
                    merged.loc[merged.initial_state_id == pid, column] = selected[column].values
            merged.to_csv(args.output / 'submission.csv', index=False)
            save()
        summary['status'] = 'complete'
    except Exception as error:
        summary['status'], summary['error'] = 'failed', str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
