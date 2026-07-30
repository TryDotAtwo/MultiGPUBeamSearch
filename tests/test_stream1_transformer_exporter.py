import unittest

import torch
import torch.nn.functional as F

from tools.export_stream1 import detect_format_from_state_dict
from tools.export_stream1_transformer import (
    build_fast_input_tables,
    infer_architecture,
    normalize_metadata,
    strip_state_prefixes,
)


def make_piece_transformer_state(num_layers=4, output_dim=24, ff_dim=1024):
    torch.manual_seed(123)
    sd = {
        "cls_token": torch.randn(1, 1, 256),
        "local_value_embedding.weight": torch.randn(360, 256),
        "piece_projection.weight": torch.randn(256, 768),
        "piece_projection.bias": torch.randn(256),
        "piece_position_embedding.weight": torch.randn(50, 256),
        "piece_type_embedding.weight": torch.randn(2, 256),
        "input_norm.weight": torch.randn(256),
        "input_norm.bias": torch.randn(256),
        "output_norm.weight": torch.randn(256),
        "output_norm.bias": torch.randn(256),
        "output_layer.weight": torch.randn(output_dim, 256),
        "output_layer.bias": torch.randn(output_dim),
    }
    for block in range(num_layers):
        prefix = f"blocks.{block}"
        sd[f"{prefix}.norm1.weight"] = torch.randn(256)
        sd[f"{prefix}.norm1.bias"] = torch.randn(256)
        sd[f"{prefix}.attn.in_proj_weight"] = torch.randn(768, 256)
        sd[f"{prefix}.attn.in_proj_bias"] = torch.randn(768)
        sd[f"{prefix}.attn.out_proj.weight"] = torch.randn(256, 256)
        sd[f"{prefix}.attn.out_proj.bias"] = torch.randn(256)
        sd[f"{prefix}.norm2.weight"] = torch.randn(256)
        sd[f"{prefix}.norm2.bias"] = torch.randn(256)
        sd[f"{prefix}.ff.0.weight"] = torch.randn(ff_dim, 256)
        sd[f"{prefix}.ff.0.bias"] = torch.randn(ff_dim)
        sd[f"{prefix}.ff.3.weight"] = torch.randn(256, ff_dim)
        sd[f"{prefix}.ff.3.bias"] = torch.randn(256)
    return sd


def make_metadata(transformer_layers=4, transformer_ff_dim=1024, n_gens=24):
    return {
        "model_arch": "piece_transformer",
        "state_size": 120,
        "n_gens": n_gens,
        "transformer_d_model": 256,
        "transformer_heads": 8,
        "transformer_layers": transformer_layers,
        "transformer_ff_dim": transformer_ff_dim,
        "transformer_activation": "silu",
        "transformer_pooling": "cls",
        "piece_layout": "p900",
        "piece_embed_mode": "full_s120",
        "num_pieces": 50,
        "max_piece_size": 3,
        "model_id": 1782210824,
    }



def make_cube4_state():
    sd = make_piece_transformer_state()
    sd["local_value_embedding.weight"] = torch.randn(18, 256)
    sd["piece_position_embedding.weight"] = torch.randn(56, 256)
    sd["piece_type_embedding.weight"] = torch.randn(3, 256)
    return sd


def make_cube4_metadata():
    metadata = make_metadata()
    metadata.update({"state_size": 96, "transformer_activation": "relu",
                     "piece_layout": "cube4", "piece_embed_mode": "piece_local",
                     "num_pieces": 56})
    return metadata



class Stream1TransformerExporterTests(unittest.TestCase):
    def test_normalizes_bundle_cube4_metadata(self):
        raw = {
            "model": {"provider": "piece_transformer", "layout": "cube4", "kwargs": {
                "transformer_d_model": 256, "transformer_heads": 8,
                "transformer_layers": 4, "transformer_ff_dim": 1024,
                "transformer_activation": "relu", "transformer_pooling": "cls"}},
            "state_size": 96, "num_actions": 24,
        }
        metadata = normalize_metadata(raw)
        self.assertEqual(metadata["model_arch"], "piece_transformer")
        self.assertEqual(metadata["piece_layout"], "cube4")
        self.assertEqual(metadata["piece_embed_mode"], "piece_local")
        self.assertEqual(metadata["num_pieces"], 56)
        self.assertEqual(metadata["max_piece_size"], 3)


    def test_detects_piece_transformer_from_marker_keys_after_prefix_strip(self):
        sd = {f"module._orig_mod.{key}": value for key, value in make_piece_transformer_state().items()}
        self.assertEqual(detect_format_from_state_dict(sd), "piece-transformer")
        stripped = strip_state_prefixes(sd)
        self.assertIn("blocks.0.attn.in_proj_weight", stripped)
        self.assertNotIn("module._orig_mod.output_layer.weight", stripped)

    def test_infers_required_p900_piece_transformer_architecture(self):
        arch = infer_architecture(
            make_piece_transformer_state(),
            make_metadata(),
            move_names=[f"m{i}" for i in range(24)],
            num_classes=120,
        )
        self.assertEqual(arch["backend"], "piece_transformer")
        self.assertEqual(arch["state_len"], 120)
        self.assertEqual(arch["num_classes"], 120)
        self.assertEqual(arch["move_count"], 24)
        self.assertEqual(arch["output_dim"], 24)
        self.assertEqual(arch["seq_len"], 51)
        self.assertEqual(arch["d_model"], 256)
        self.assertEqual(arch["nhead"], 8)
        self.assertEqual(arch["head_dim"], 32)
        self.assertEqual(arch["num_layers"], 4)
        self.assertEqual(arch["ff_dim"], 1024)
        self.assertEqual(arch["activation"], "silu")
        self.assertEqual(arch["pooling"], "cls")

    def test_infers_cube4_piece_transformer_architecture(self):
        arch = infer_architecture(
            make_cube4_state(), make_cube4_metadata(),
            move_names=[f"m{i}" for i in range(24)], num_classes=6)
        self.assertEqual(arch["state_len"], 96)
        self.assertEqual(arch["num_classes"], 6)
        self.assertEqual(arch["num_pieces"], 56)
        self.assertEqual(arch["num_piece_types"], 3)
        self.assertEqual(arch["seq_len"], 57)
        self.assertEqual(arch["activation"], "relu")
        self.assertEqual(arch["piece_layout"], "cube4")
        self.assertEqual(arch["piece_embed_mode"], "piece_local")





    def test_rejects_consistent_non_24_move_piece_transformer(self):
        with self.assertRaisesRegex(ValueError, "requires move_count=24"):
            infer_architecture(
                make_piece_transformer_state(output_dim=23),
                make_metadata(n_gens=23),
                move_names=[f"m{i}" for i in range(23)],
                num_classes=120,
            )

    def test_rejects_consistent_non_4_layer_piece_transformer(self):
        with self.assertRaisesRegex(ValueError, "requires transformer_layers=4"):
            infer_architecture(
                make_piece_transformer_state(num_layers=3),
                make_metadata(transformer_layers=3),
                move_names=[f"m{i}" for i in range(24)],
                num_classes=120,
            )

    def test_rejects_consistent_non_1024_ff_piece_transformer(self):
        with self.assertRaisesRegex(ValueError, "requires transformer_ff_dim=1024"):
            infer_architecture(
                make_piece_transformer_state(ff_dim=2048),
                make_metadata(transformer_ff_dim=2048),
                move_names=[f"m{i}" for i in range(24)],
                num_classes=120,
            )

    def test_fast_input_tables_match_kaggle_fast_cache_formula(self):
        sd = make_piece_transformer_state()
        piece_positions = torch.arange(0, 150, dtype=torch.int64).view(50, 3) % 120
        piece_mask = torch.ones((50, 3), dtype=torch.bool)
        piece_types = torch.arange(50, dtype=torch.int64) % 2

        tables = build_fast_input_tables(sd, piece_positions, piece_mask, piece_types)
        self.assertEqual(tuple(tables["fast_slot_projected"].shape), (3, 120, 256))
        self.assertEqual(tuple(tables["fast_piece_static"].shape), (50, 256))

        expected_slots = []
        for slot in range(3):
            projection_slice = sd["piece_projection.weight"][:, slot * 256 : (slot + 1) * 256]
            embeddings = sd["local_value_embedding.weight"][slot * 120 : (slot + 1) * 120]
            expected_slots.append(F.linear(embeddings, projection_slice))
        expected_static = (
            sd["piece_position_embedding.weight"].index_select(0, torch.arange(50))
            + sd["piece_type_embedding.weight"].index_select(0, piece_types)
            + sd["piece_projection.bias"].view(1, -1)
        )
        self.assertTrue(torch.equal(tables["piece_positions"], piece_positions.to(torch.int16)))
        self.assertTrue(torch.equal(tables["piece_mask"], piece_mask.to(torch.uint8)))
        self.assertTrue(torch.equal(tables["piece_types"], piece_types.to(torch.uint8)))
        self.assertTrue(torch.allclose(tables["fast_slot_projected"], torch.stack(expected_slots)))
        self.assertTrue(torch.allclose(tables["fast_piece_static"], expected_static))


if __name__ == "__main__":
    unittest.main()
