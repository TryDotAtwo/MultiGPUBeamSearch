"""Reference parity against the independently supplied Artgor JAX implementation."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from tools.cube555.models import load_blend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    root = args.assets
    spec = importlib.util.spec_from_file_location('cube555_jax_reference', root / 'jax_model.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    import jax.numpy as jnp
    model, _, _ = load_blend(root / 'q555_f1_bell2k.pt', root / 'q555_2k_BEST.pt', root / 'piece_layout_555.json')
    info = json.loads((root / 'puzzle_info.json').read_text())
    moves = np.asarray(list(info['generators'].values()))
    state = np.arange(150, dtype=np.uint8)
    states = [state.copy()]
    rng = np.random.default_rng(555)
    for depth in range(1, 41):
        state = state[moves[rng.integers(30)]]
        if depth in (1, 2, 5, 10, 20, 40):
            states.append(state.copy())
    x = np.stack(states)
    t = ref.load_params_from_pt(root / 'q555_f1_bell2k.pt')
    m = ref.load_params_from_pt(root / 'q555_2k_BEST.pt')
    references = [t, m, ref.make_blend([t, m], [0.8, 0.2])]
    output = {}
    with torch.inference_mode():
        for name, candidate, params in zip(['transformer', 'resmlp', 'blend'], [model.transformer, model.mlp, model], references):
            expected = np.asarray(ref.apply(params, jnp.asarray(x), dtype=jnp.float32))
            actual = candidate(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            output[name] = dict(max_abs=float(np.abs(actual - expected).max()),
                best_action_agreement=float(np.mean(actual.argmin(1) == expected.argmin(1))))
        scripted = torch.jit.script(model)
        for batch in (1, 3, len(x)):
            padded = torch.nn.functional.pad(torch.from_numpy(x[:batch]), (0, 10), value=255)
            torch.testing.assert_close(scripted(padded), model(torch.from_numpy(x[:batch])), rtol=1e-5, atol=1e-5)
    output.update(states=len(x), seed=555, includes_sticker_values_128_149=True,
                  scripted_variable_batch_and_padding='passed', status='passed', scope='CPU FP32 parity; not a GPU solve')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
