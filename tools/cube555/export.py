"""Export two graph-bound feature models for explicit NativeEnsemble readout.

The ensemble performs FP32 readout and blending, then one shared clamp/round.
This is deliberately distinct from rounding each FP16 head before blending.
"""
import json
from pathlib import Path
import shutil

import torch
from torch import nn

from tools.cube555.models import load_model, PieceTransformerQ555, ResMLPQ


class Features(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        torch._assert(states.dtype == torch.uint8, 'Cube555 features require uint8 sticker IDs')
        torch._assert(states.dim() == 2, 'Cube555 features require a batch matrix')
        torch._assert(states.size(1) == 150 or states.size(1) == 160, 'Cube555 input width must be 150 or padded 160')
        return self.model.features(states)


def export_native_models(transformer, mlp, layout_path, puzzle_info, output, *, contract):
    """Return (Transformer NativeModel, ResMLP NativeModel), with no implicit blend.

    contract is GraphContract.from_graph(graph, graph.definition.central_state).
    Call NativeEnsemble(members, (weight, 1.0-weight)) explicitly after export.
    Requires native Cube555 feature-family executor support before a GPU launch.
    """
    from multigpubeamsearch.cube555_artifacts import file_hash, read_json, validate_definition, validate_bundle
    from multigpubeamsearch.models import NativeModel
    output = Path(output)
    if output.exists():
        raise ValueError('export directory already exists: '+str(output))
    layout, info = read_json(layout_path), read_json(puzzle_info)
    validate_definition(layout, info, contract)
    models = []
    for checkpoint, kind in ((transformer, PieceTransformerQ555), (mlp, ResMLPQ)):
        model, config = load_model(Path(checkpoint), layout)
        if not isinstance(model, kind):
            raise ValueError('expected Transformer then ResMLP checkpoints')
        if config['d_model'] % 8 or not 8 <= config['d_model'] <= 65536:
            raise ValueError('Cube555 feature width must be a multiple of eight')
        if any(not torch.isfinite(parameter).all() or (parameter.abs() > 65504).any()
               for parameter in model.parameters()):
            raise ValueError('Cube555 parameters must be finite and representable as FP16')
        models.append((checkpoint, model.half().eval(), config))
    # No output is created until both source models and graph/layout are accepted.
    output.mkdir(parents=True)
    members = []
    for family, (checkpoint, model, config) in zip(('cube555_transformer', 'cube555_mlp'), models):
        directory = output/family
        directory.mkdir()
        feature_model = Features(model).eval()
        script = torch.jit.script(feature_model)
        probes = torch.stack([torch.arange(150), torch.arange(149, -1, -1)]).to(torch.uint8)
        with torch.inference_mode():
            for states in (probes, torch.nn.functional.pad(probes, (0, 10), value=255)):
                torch.testing.assert_close(script(states), model.features(states), rtol=0, atol=0)
        script.save(str(directory/'features.ts'))
        model.q_head.weight.detach().T.contiguous().numpy().astype('<f2').tofile(directory/'output_weight.fp16')
        model.q_head.bias.detach().float().numpy().astype('<f4').tofile(directory/'output_bias.fp32')
        shutil.copyfile(layout_path, directory/'layout.json')
        shutil.copyfile(puzzle_info, directory/'puzzle_info.json')
        names = ('features.ts', 'output_weight.fp16', 'output_bias.fp32', 'layout.json', 'puzzle_info.json')
        manifest = dict(schema_version=1, family=family, dtype='fp16', state_len=150,
            num_classes=150, output_dim=30, feature_dim=config['d_model'], layernorm_eps=1e-5,
            hidden1=config['d_model'], hidden2=config['d_model'], residual_count=1,
            normalization='none', planning_descriptor_only=True,
            graph_hash=contract.graph_hash, move_names=list(contract.generator_names),
            score_semantics='distance_q', readout_semantics='fp16_features_fp32_accumulation',
            feature_input_dtype='uint8', feature_output_dtype='fp16', output_weight_layout='hidden_by_output',
            checkpoint_sha256=file_hash(checkpoint), layout_sha256=file_hash(layout_path),
            puzzle_info_sha256=file_hash(puzzle_info), files={name: file_hash(directory/name) for name in names})
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        prepared = validate_bundle(directory, contract, family)
        members.append(NativeModel(directory, contract.graph_hash, backend=family,
                                   expected_artifact_hash=prepared.artifact_hash))
    return tuple(members)
