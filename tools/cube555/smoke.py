"""Deterministic legal scrambles and CUDA eager/script parity for the two-T4 gate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tools.cube555.models import load_blend


def make_fixture(assets: Path, output: Path):
    info = json.loads((assets / 'puzzle_info.json').read_text())
    names = list(info['generators'])
    moves = np.asarray(list(info['generators'].values()))
    rows, baseline = [], []
    # Different axes; IDs 0..3 here belong only to this synthetic fixture.
    for pid, path in enumerate(([0], [0, 10], [0, 10, 20], [0, 12, 24, 6])):
        state = np.arange(150, dtype=np.uint8)
        for move in path:
            state = state[moves[move]]
        inverse = [names[m ^ 1] for m in reversed(path)]
        replay = state.copy()
        for name in inverse:
            replay = replay[moves[names.index(name)]]
        assert np.array_equal(replay, np.arange(150))
        rows.append(dict(initial_state_id=pid, initial_state=','.join(map(str, state))))
        baseline.append(dict(initial_state_id=pid, solution='.'.join(inverse)))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'test.csv', index=False)
    pd.DataFrame(baseline).to_csv(output / 'sample_submission.csv', index=False)


def validate_cuda(assets: Path, output: Path):
    info = json.loads((assets / 'puzzle_info.json').read_text())
    moves = np.asarray(list(info['generators'].values()))
    rng = np.random.default_rng(555)
    state = np.arange(150, dtype=np.uint8)
    states = []
    for _ in range(128):
        state = state[moves[rng.integers(30)]]
        states.append(state.copy())
    cpu_states = torch.from_numpy(np.stack(states))
    report = {}
    # Each physical GPU runs its own comparison; no fake ranks or shared cuda:0.
    for index in range(2):
        model, _, _ = load_blend(assets / 'q555_f1_bell2k.pt', assets / 'q555_2k_BEST.pt', assets / 'piece_layout_555.json')
        device = torch.device('cuda', index)
        candidate = model.to(device).eval()
        x = cpu_states.to(device)
        with torch.inference_mode():
            reference = candidate.float()(x).float()
            candidate.half()
            script = torch.jit.script(candidate)
            actual = candidate(x)
            if not torch.isfinite(actual).all():
                raise AssertionError('nonfinite FP16 blend logits')
            # FP16 is a different arithmetic path from the TPU BF16 run.
            torch.testing.assert_close(actual, reference, atol=0.3, rtol=0.01)
            max_script_error = 0.0
            for count in (1, 7, 128):
                padded = torch.nn.functional.pad(x[:count], (0, 10), value=255)
                with torch.jit.optimized_execution(False):
                    out = script(padded)
                eager = candidate(x[:count])
                print(f"GPU {index} batch {count} script/eager max_abs={float((out-eager).abs().max())}", flush=True)
                torch.testing.assert_close(out, eager, atol=1e-4, rtol=1e-5)
                max_script_error = max(max_script_error, float((out - eager).abs().max()))
            report[str(index)] = dict(gpu=torch.cuda.get_device_name(index),
                max_abs_fp16_fp32=float((actual-reference).abs().max()),
                top1_agreement=float((actual.argmin(1) == reference.argmin(1)).float().mean()),
                script_max_abs=max_script_error)
        del script
        model = candidate.float().cpu()
        torch.cuda.empty_cache()
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
