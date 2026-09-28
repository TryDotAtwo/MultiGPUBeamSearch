"""Tune model forwards independently of the fixed 8192-parent transaction."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from tools.cube555.models import load_blend


def benchmark(assets, output, outer=8192, micros=(64, 128, 256, 512, 1024)):
    info = json.loads((assets / 'puzzle_info.json').read_text())
    moves = np.asarray(list(info['generators'].values()))
    rng = np.random.default_rng(555)
    state = np.arange(150, dtype=np.uint8)
    samples = []
    for _ in range(outer):
        state = state[moves[rng.integers(len(moves))]]
        samples.append(state.copy())
    cpu = torch.nn.functional.pad(torch.from_numpy(np.stack(samples)), (0, 10))
    rows = []
    for gpu in range(2):
        torch.cuda.set_device(gpu)
        model, _, _ = load_blend(assets/'q555_f1_bell2k.pt', assets/'q555_2k_BEST.pt', assets/'piece_layout_555.json')
        model = torch.jit.script(model.half()).to('cuda').eval()
        x = cpu.to('cuda')
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            reference = model(x[:128]).float().cpu()
            for micro in micros:
                row = dict(gpu=gpu, outer_parent_batch=outer, model_microbatch=micro)
                try:
                    for _ in range(2):
                        model(x[:micro])
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    times=[]
                    for _ in range(3):
                        start=time.perf_counter()
                        for begin in range(0, outer, micro):
                            y=model(x[begin:begin+micro])
                        torch.cuda.synchronize()
                        times.append(time.perf_counter()-start)
                    actual=model(x[:max(128,micro)])[:128].float().cpu()
                    if not torch.isfinite(actual).all():
                        raise ValueError('nonfinite model output')
                    error=float((actual-reference).abs().max())
                    if error > 0.3:
                        raise ValueError(f'batch-dependent Q drift exceeds 0.3: {error}')
                    row.update(status='complete', seconds=times, median_seconds=statistics.median(times),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        max_abs_vs_micro128=error, top1_agreement_vs_micro128=float((actual.argmin(1)==reference.argmin(1)).float().mean()))
                except torch.OutOfMemoryError:
                    row.update(status='oom')
                    gc.collect();torch.cuda.empty_cache()
                rows.append(row)
                print(json.dumps(row), flush=True)
                output.write_text(json.dumps(dict(rows=rows),indent=2))
        del model,x,y
        gc.collect();torch.cuda.empty_cache()
    candidates=[]
    for micro in micros:
        matches=[r for r in rows if r['model_microbatch']==micro and r['status']=='complete']
        if len(matches)==2:
            candidates.append((max(r['median_seconds'] for r in matches),micro))
    if not candidates:
        raise RuntimeError('no microbatch passed on both physical GPUs')
    selected=min(candidates)[1]
    report=dict(outer_parent_batch=outer,selected_model_micro=selected,rows=rows,
        note='Isolated model measurement; native depth loop and total device memory must be verified separately.')
    output.write_text(json.dumps(report,indent=2))
    return selected


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); benchmark(a.assets,a.output)
