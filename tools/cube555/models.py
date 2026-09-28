"""Exact checkpoint schemas from Artgor's CC0 Cube555 artifacts.

The native runtime consumes parent Q logits, in the layout's move order.
Use uint8 states: sticker IDs 128..149 are valid. No scalar-head offset.
"""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.width = width
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * width))
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.SiLU(), nn.Linear(ff, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class PieceTransformerQ555(nn.Module):
    def __init__(self, config: dict, layout: dict):
        super().__init__()
        if (layout['state_size'], layout['num_pieces'], layout['n_actions']) != (150, 98, 30):
            raise ValueError('Cube555 requires the 150-facelet / 98-piece / 30-action layout')
        if config.get('activation') != 'silu' or config.get('output_dim') != 30:
            raise ValueError('unsupported Cube555 transformer configuration')
        d = config['d_model']
        heads = torch.tensor(layout['head_facelet'], dtype=torch.long)
        self.register_buffer('heads', heads, persistent=False)
        self.register_buffer('sticker_local', torch.tensor(layout['sticker_local'], dtype=torch.long), persistent=False)
        offsets = torch.tensor(layout['sticker_orbit'], dtype=torch.long)[heads] * layout['sticker_vocab']
        self.register_buffer('orbit_offset', offsets, persistent=False)
        self.register_buffer('piece_types', torch.tensor(layout['piece_types'], dtype=torch.long), persistent=False)
        self.sticker_embedding = nn.Embedding(len(layout['orbit_sizes']) * layout['sticker_vocab'], d)
        self.piece_position_embedding = nn.Embedding(98, d)
        self.piece_type_embedding = nn.Embedding(layout['num_piece_types'], d)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d))
        self.input_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([Block(d, config['nhead'], config['ff_dim']) for _ in range(config['num_layers'])])
        self.output_norm = nn.LayerNorm(d)
        self.q_head = nn.Linear(d, 30)
        self.v_head = nn.Linear(d, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = states[:, :150].long()
        stickers = x.index_select(1, self.heads)
        tokens = self.sticker_local[stickers] + self.orbit_offset.unsqueeze(0)
        h = self.sticker_embedding(tokens) + self.piece_position_embedding.weight.unsqueeze(0)
        h = h + self.piece_type_embedding(self.piece_types).unsqueeze(0)
        h = self.input_norm(torch.cat([self.cls_token.expand(x.size(0), -1, -1), h], dim=1))
        for block in self.blocks:
            h = block(h)
        return self.q_head(self.output_norm(h[:, 0]))


class Residual(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(d, d), nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(self.lin1(x)))
        return F.relu(x + self.ln2(self.lin2(h)))


class ResMLPQ(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if (config['state_size'], config['num_classes'], config['output_dim'], config['encoding']) != (150, 150, 30, 'embedding'):
            raise ValueError('unsupported Cube555 ResMLPQ configuration')
        d, e = config['d_model'], config['embed_dim']
        self.embedding = nn.Embedding(150, e)
        self.input_stack = nn.Sequential(nn.Linear(150 * e, d), nn.LayerNorm(d), nn.ReLU())
        self.res_blocks = nn.ModuleList([Residual(d) for _ in range(config['num_res_blocks'])])
        self.q_head, self.v_head = nn.Linear(d, 30), nn.Linear(d, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        h = self.input_stack(self.embedding(states[:, :150].long()).flatten(1))
        for block in self.res_blocks:
            h = block(h)
        return self.q_head(h)


class Blend(nn.Module):
    def __init__(self, transformer: PieceTransformerQ555, mlp: ResMLPQ, transformer_weight: float = 0.8):
        super().__init__()
        if not 0 <= transformer_weight <= 1:
            raise ValueError('transformer weight must be finite and in [0, 1]')
        self.transformer, self.mlp = transformer, mlp
        self.transformer_weight = transformer_weight

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # Mix in fp32 before the one shared clamp/round in the native score ring.
        return self.transformer(states).float() * self.transformer_weight + self.mlp(states).float() * (1.0 - self.transformer_weight)


def load_model(path: Path, layout: dict):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['model_config']
    cls = config['model_class']
    if cls == 'PieceTransformerQ555':
        model = PieceTransformerQ555(config, layout)
    elif cls == 'ResMLPQ':
        model = ResMLPQ(config)
    else:
        raise ValueError(f'unsupported Cube555 model class {cls!r}')
    state = checkpoint.get('model', checkpoint.get('state_dict'))
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in state.items()}, strict=True)
    return model.eval(), config


def load_blend(transformer: Path, mlp: Path, layout_path: Path, weight: float = 0.8):
    layout = json.loads(layout_path.read_text(encoding='utf-8'))
    t, config = load_model(transformer, layout)
    m, _ = load_model(mlp, layout)
    if not isinstance(t, PieceTransformerQ555) or not isinstance(m, ResMLPQ):
        raise ValueError('expected Transformer then ResMLP checkpoints')
    return Blend(t, m, weight).eval(), config, layout
