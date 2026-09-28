"""Two-T4 Cube555 launcher using the existing native distributed beam solver."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.data import load_puzzle_contract
from tools.cayleypy_public.model import ExportedModel
from tools.cayleypy_public.profile import RuntimePlan, derive_runtime
from tools.kaggle_t4_mlp_profiles import select_profile
from tools.cayleypy_public.runner import maximum_history_depth
from tools.cayleypy_public.runner import PublicSearchRunError
from tools.run_cayleypy_public import (
    validate_t4_hardware, locate_or_build_runner, _available_ram_bytes,
    _run_with_history_budgets, _materialize_run_artifacts, _publish_best_effort, _derive_history_budgets,
)
from tools.cube555.export import export_blend


MAX_BEAM = 4_000_000
MAX_DEPTH = 140
DEFAULT_BEAM = MAX_BEAM
BEAM_PROFILES = (65_536, 131_072, 262_144, 524_288, 1_048_576, 2_097_152, MAX_BEAM)


def history_budgets(available_ram_bytes, tmp_free_bytes):
    # Shared helper already retains 1.5 GB; retain another 4.5 GB for two
    # LibTorch ranks, BFS neighborhoods, Python results and runtime overhead.
    ram, disk = _derive_history_budgets(available_ram_bytes - 4_500_000_000, tmp_free_bytes)
    return min(ram, 24_000_000_000), disk



def runtime_plan(beam: int, profile: str = 'safe', *, b_micro: int = 8192,
                 model_micro: int | None = None) -> RuntimePlan:
    if type(beam) is not int or not 1 <= beam <= MAX_BEAM:
        raise ValueError(f'Cube555 beam must be in [1, {MAX_BEAM}]')
    micro = {'safe': 128, 'balanced': 256, 'throughput': 512}[profile] if model_micro is None else model_micro
    if type(b_micro) is not int or not 1 <= b_micro <= 65536:
        raise ValueError('outer b_micro must be in [1, 65536]')
    if type(micro) is not int or not 1 <= micro <= b_micro:
        raise ValueError('model_micro must be in [1, outer b_micro]')
    registry = json.loads((Path(__file__).resolve().parents[2] / 'configs/kaggle_t4_transformer_profiles.json').read_text())
    seed = select_profile(registry, beam, 30, 30)
    old_batch = seed['runtime']['b_micro']
    seed['runtime']['b_micro'] = b_micro
    # Large Cube4 profiles encode many tiny outer slots. Preserve the candidate
    # transaction budget while changing its unit from 384 to 8192 parents.
    old_slots = seed['runtime']['stream3_ring_slots']
    seed['runtime']['stream3_ring_slots'] = max(2, (old_batch * old_slots + b_micro - 1) // b_micro)
    plan = derive_runtime(seed, beam, 30, 30, 2)
    return replace(plan, runtime={**plan.runtime, 'model_micro': micro},
        cross_puzzle_profile_note='Cube4 Transformer pipeline seed; Cube555 capacity requires native preflight and measurement')


def configuration(args, pid, puzzle_info):
    return PublicRunConfig.from_mapping(dict(
        author_name='Ivan Litvak', checkpoint_path=str(getattr(args, 'checkpoint', None) or args.assets / 'q555_f1_bell2k.pt'),
        puzzle_info_json=str(puzzle_info), test_csv=str(args.competition / 'test.csv'),
        sample_submission_csv=str(args.competition / 'sample_submission.csv'),
        puzzle_id_start=pid, puzzle_id_end=pid, beam_width=args.beam, max_depth=args.depth,
        reflect_mode=getattr(args, 'reflect_mode', 'off'),
        reflect_source_csv=getattr(args, 'reflect_source_csv', None),
        solution_mode=getattr(args, 'solution_mode', 'first'),
        collect_until_depth=args.depth if getattr(args, 'collect_until_depth', None) is None else min(args.collect_until_depth, args.depth),
        max_collected_solutions=getattr(args, 'max_collected_solutions', 100_000), touch_bfs_radius=args.touch_radius,
        publish_results=getattr(args, 'publish', False),
        results_ingest_url=getattr(args, 'ingest_url', ''), enable_debug=True,
        **getattr(args, 'publication', {}),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--competition', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--mlp-checkpoint', type=Path)
    parser.add_argument('--layout', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pids', type=int, nargs='+', default=[1020, 1034])
    parser.add_argument('--beam', type=int, default=DEFAULT_BEAM)
    parser.add_argument('--depth', type=int, default=140)
    parser.add_argument('--touch-radius', type=int, default=5)
    parser.add_argument('--transformer-weight', type=float, default=0.8)
    parser.add_argument('--b-micro', type=int, default=8192)
    parser.add_argument('--model-micro', type=int, default=512)
    parser.add_argument('--reflect-mode', choices=['off', 'after_original', 'only'], default='off')
    parser.add_argument('--reflect-source-csv', type=Path)
    parser.add_argument('--solution-mode', choices=['first', 'collect'], default='collect')
    parser.add_argument('--collect-until-depth', type=int)
    parser.add_argument('--max-collected-solutions', type=int, default=100_000)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--ingest-url', default='https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v1/results')
    parser.add_argument('--publication-json', type=Path)
    args = parser.parse_args()
    if not 1 <= args.depth <= MAX_DEPTH:
        parser.error(f'--depth must be in [1, {MAX_DEPTH}]')
    if args.collect_until_depth is not None and not 0 <= args.collect_until_depth <= args.depth:
        parser.error('--collect-until-depth must be within --depth')
    args.publication = json.loads(args.publication_json.read_text()) if args.publication_json else {}
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
    from tools.cube555.telemetry import Telemetry
    telemetry = Telemetry(args.output)
    telemetry.start()
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
        manifest = export_blend(args.checkpoint or args.assets / 'q555_f1_bell2k.pt',
            args.mlp_checkpoint or args.assets / 'q555_2k_BEST.pt',
            args.layout or args.assets / 'piece_layout_555.json', info, export_dir, args.transformer_weight)
        model = ExportedModel('cube555-q-blend', 'fp16', manifest['script_sha256'], manifest, 'piece_transformer')
        plan = runtime_plan(args.beam, b_micro=args.b_micro, model_micro=args.model_micro)
        # Reuse the existing public profiles' RAM/disk contract and explicit
        # depth cap. Beam is never reduced to fit history.
        ram, disk = history_budgets(_available_ram_bytes(), shutil.disk_usage('/tmp').free)
        requested_depth = args.depth
        budget_depth = maximum_history_depth(plan, 30, args.touch_radius, ram, disk)
        args.depth = min(requested_depth, budget_depth)
        config = configuration(args, args.pids[0], info)
        profile_evidence = f'cube4-seed-p{plan.profile_power}-outer{args.b_micro}-model{plan.runtime["model_micro"]}-cube555-unmeasured'
        preflight = dict(profile_name=f'p{plan.profile_power}', outer_parent_batch=plan.parent_batch,
            model_microbatch=plan.runtime['model_micro'], plan=asdict(plan), state_len=150,
            state_storage_len=160, state_value_pad=256, history_ram_bytes=ram, history_disk_bytes=disk,
            requested_max_depth=requested_depth, budget_max_depth=budget_depth, effective_max_depth=args.depth,
            profile_status=profile_evidence, checkpoint_sha256=manifest['checkpoint_sha256'])
        summary.update(requested_max_depth=requested_depth, effective_max_depth=args.depth)
        if args.depth < requested_depth:
            print(f'History budget: MAX_DEPTH {requested_depth} -> {args.depth}; BEAM_WIDTH stays {args.beam}', flush=True)
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
            publication = _publish_best_effort(configuration(args, pid, info), contracts[pid], model,
                {'profile_registry_schema_version': 1, 'evidence': profile_evidence},
                plan, summary['gpus'], artifacts, out, time.monotonic() - start)
            print('Publication:', json.dumps(publication), flush=True)
            summary['results'].append(dict(pid=pid, publication=publication, **result))
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
        summary['performance'] = telemetry.finish()
        save()


if __name__ == '__main__':
    main()
