"""Export the two checkpoint Q blend for the existing LibTorch Stream1 hook."""
import hashlib
import json
from pathlib import Path
import torch
from tools.cube555.models import load_blend


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_blend(transformer, mlp, layout_path, puzzle_info, output, weight=0.8):
    output = Path(output)
    if output.exists():
        raise ValueError(f'export directory already exists: {output}')
    model, config, layout = load_blend(Path(transformer), Path(mlp), Path(layout_path), weight)
    info = json.loads(Path(puzzle_info).read_text(encoding='utf-8'))
    if info['central_state'] != list(range(150)) or list(info['generators']) != layout['_move_names']:
        raise ValueError('Cube555 picture state and checkpoint move order must match exactly')
    if any(sorted(g) != list(range(150)) for g in info['generators'].values()):
        raise ValueError('every generator must be a permutation of 150 facelets')
    # Check script vs eager at different batch sizes, including padded native input.
    probe = torch.arange(150, dtype=torch.uint8).unsqueeze(0)
    scripted = torch.jit.script(model)
    with torch.inference_mode():
        for batch in (1, 3):
            states = torch.nn.functional.pad(probe.repeat(batch, 1), (0, 10))
            torch.testing.assert_close(scripted(states), model(states), rtol=1e-5, atol=1e-5)
    output.mkdir(parents=True)
    # Keep a non-frozen script: the C++ loader relocates parameters and integer buffers.
    scripted.half().save(str(output / 'cube555_blend.ts'))
    manifest = dict(
        backend='piece_transformer', model_family='cube555_q_blend',
        runtime='libtorch_eager', dtype='fp16', state_len=150, num_classes=150,
        move_count=30, output_dim=30, num_pieces=98, max_piece_size=1,
        seq_len=99, d_model=config['d_model'], nhead=config['nhead'],
        head_dim=config['d_model'] // config['nhead'], num_layers=config['num_layers'],
        transformer_layers=config['num_layers'], ff_dim=config['ff_dim'],
        activation='silu', pooling='cls', piece_layout='cube555', piece_embed_mode='orbit_head',
        source_generators=str(Path(puzzle_info).resolve()),
        blend_weights=[weight, 1.0 - weight],
        checkpoint_sha256=[sha256(transformer), sha256(mlp)],
        layout_sha256=sha256(layout_path), puzzle_info_sha256=sha256(puzzle_info),
        script_sha256=sha256(output / 'cube555_blend.ts'),
        move_names=layout['_move_names'], score_contract='parent_q_fp32_blend_then_clamp_round',
    )
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest
