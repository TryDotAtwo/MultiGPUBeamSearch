"""Bounded native two-T4 gate for the relative collection window."""
import argparse
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import torch
from tools.cube555.export import export_blend
from tools.cube555.run import configuration, history_budgets, runtime_plan
from tools.cube555.smoke import make_fixture
from tools.cayleypy_public.data import load_puzzle_contract
from tools.cayleypy_public.model import ExportedModel
from tools.run_cayleypy_public import (
    validate_t4_hardware, locate_or_build_runner, _available_ram_bytes,
    _run_with_history_budgets, _materialize_run_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(gpus=validate_t4_hardware(), rows=[])
    fixture = args.output / 'fixture'
    make_fixture(args.assets, fixture)
    info = args.assets / 'puzzle_info.json'
    manifest = export_blend(args.assets/'q555_f1_bell2k.pt',
        args.assets/'q555_2k_BEST.pt', args.assets/'piece_layout_555.json',
        info, args.output/'export')
    model = ExportedModel('cube555-q-blend', 'fp16', manifest['script_sha256'],
                          manifest, 'piece_transformer')
    cfg = SimpleNamespace(assets=args.assets, competition=fixture, beam=8192,
        depth=3, touch_radius=0, solution_mode='collect', max_collected_solutions=2000)
    runner = locate_or_build_runner(args.output, info, backend='piece_transformer',
                                    config=configuration(cfg, 0, info))
    contract = load_puzzle_contract(info, fixture/'test.csv',
                                   fixture/'sample_submission.csv', 0, 0)
    for extra, concurrency in ((0, 1), (1, 2), (2, 4), (10**30, 1)):
        cfg.collect_extra_depths = extra
        plan = runtime_plan(8192, model_micro=128, inference_concurrency=concurrency)
        out = args.output / f'extra-{extra}-c{concurrency}'
        out.mkdir()
        ram, disk = history_budgets(_available_ram_bytes(), shutil.disk_usage('/tmp').free)
        artifacts = _run_with_history_budgets(ram, disk, configuration(cfg, 0, info),
            contract, model, plan, args.output/'export', out/'logs', runner_path=str(runner))
        result = _materialize_run_artifacts(artifacts, out)
        assert artifacts.solution_records, 'one-move fixture must solve'
        ranks = []
        for rank in (0, 1):
            logs = list((out/'logs').rglob(f'rank-{rank}.log'))
            assert len(logs) == 1, logs
            text = logs[0].read_text()
            depths = [int(d) for d in re.findall(r'depth_start=(\d+)', text)]
            assert max(depths) == min(extra, 2), (extra, rank, depths)
            assert 'collection_status=capacity_reached' not in text
            if extra < 3 and rank == 0:
                assert re.search(rf'solve_bucket_stop=1 .*depth_index={extra} first_found_depth_index=0 extra_depths={extra}', text)
            ranks.append(dict(rank=rank, last_depth=max(depths), log=str(logs[0].relative_to(args.output))))
        report['rows'].append(dict(extra_depths=extra, concurrency=concurrency,
                                   status='complete', ranks=ranks, result=result))
        (args.output/'collect_report.json').write_text(json.dumps(report, indent=2))
        print(f'PASS extra={extra} concurrency={concurrency} last_depth={min(extra,2)}', flush=True)
    report['status'] = 'complete'
    (args.output/'collect_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
