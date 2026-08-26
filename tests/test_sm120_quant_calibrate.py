from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.sm120_quant_calibrate import (
    activation_block_statistics,
    fake_sm120_activation_quant,
    fake_sm120_int8_activation_quant,
    fake_sm120_int8_weight_quant,
    fake_sm120_weight_quant,
    fake_sm120_nvfp4_activation_quant,
    fake_sm120_nvfp4_weight_quant,
    reconstruct_frontiers,
    stratified_reservoir,
)


def test_fake_nvfp4_uses_k16_blocks_and_ue4m3_scales() -> None:
    values = torch.tensor([
        [0.0, 0.5, -1.0, 1.5, -2.0, 3.0, -4.0, 6.0] * 4,
        [0.0] * 32,
    ], dtype=torch.float32)
    quantized, scales = fake_sm120_nvfp4_activation_quant(values)
    assert quantized.shape == values.shape
    assert scales.shape == (2, 2)
    assert torch.isfinite(scales).all() and torch.all(scales > 0)
    assert torch.count_nonzero(quantized[1]) == 0
    with pytest.raises(ValueError, match="divisible by 16"):
        fake_sm120_nvfp4_activation_quant(torch.ones(2, 17))

    weight = torch.linspace(-3.0, 3.0, 32 * 3).reshape(32, 3)
    quantized_weight, weight_scales = fake_sm120_nvfp4_weight_quant(weight)
    assert quantized_weight.shape == weight.shape
    assert weight_scales.shape == (2, 3)
    assert torch.isfinite(quantized_weight).all()


def _write_history(path: Path, entries: list[tuple[int, int]]) -> None:
    path.write_bytes(b"".join(struct.pack("<QII", parent, move, 0) for parent, move in entries))


def test_reconstruct_frontiers_and_stratified_reservoir(tmp_path: Path) -> None:
    initial = np.array([0, 1, 2, 3], dtype=np.uint8)
    generators = np.array([[1, 0, 2, 3], [0, 1, 3, 2]], dtype=np.int64)
    _write_history(tmp_path / "depth_0.candidate_meta.bin", [(0, 0), (0, 1)])
    _write_history(tmp_path / "depth_1.candidate_meta.bin", [(0, 1), (1, 0)])
    frontiers = reconstruct_frontiers(tmp_path, initial, generators)
    np.testing.assert_array_equal(frontiers[0], np.array([[1, 0, 2, 3], [0, 1, 3, 2]], dtype=np.uint8))
    np.testing.assert_array_equal(frontiers[1], np.array([[1, 0, 3, 2], [1, 0, 3, 2]], dtype=np.uint8))
    states, depths, indices = stratified_reservoir(frontiers, depths=(0, 1), per_depth=1)
    assert states.shape == (2, 4)
    np.testing.assert_array_equal(depths, np.array([0, 1]))
    np.testing.assert_array_equal(indices, np.array([0, 0]))


def test_fake_sm120_quantization_preserves_zero_and_bounds_error() -> None:
    generator = torch.Generator().manual_seed(9)
    activation = torch.randn(17, 256, generator=generator)
    activation[0].zero_()
    quant_activation, activation_scales = fake_sm120_activation_quant(activation)
    assert torch.count_nonzero(quant_activation[0]) == 0
    assert activation_scales.shape == (17, 2)
    assert torch.mean((quant_activation - activation).square()) < 0.002

    weight = torch.randn(256, 768, generator=generator) * 0.05
    quant_weight, weight_scales = fake_sm120_weight_quant(weight)
    assert weight_scales.shape == (2, 6)
    assert torch.mean((quant_weight - weight).square()) < 1e-5


def test_fake_block_int8_quantization_preserves_zero_and_uses_expected_scales() -> None:
    generator = torch.Generator().manual_seed(17)
    activation = torch.randn(9, 256, generator=generator)
    activation[0].zero_()
    quant_activation, activation_scales = fake_sm120_int8_activation_quant(activation)
    assert torch.count_nonzero(quant_activation[0]) == 0
    assert activation_scales.shape == (9, 2)
    assert torch.all(activation_scales > 0)
    assert torch.mean((quant_activation - activation).square()) < 0.0002

    weight = torch.randn(256, 768, generator=generator) * 0.05
    quant_weight, weight_scales = fake_sm120_int8_weight_quant(weight)
    assert weight_scales.shape == (2, 6)
    assert torch.all(weight_scales > 0)
    assert torch.mean((quant_weight - weight).square()) < 1e-6


def test_activation_statistics_include_scale_exponents_and_rates() -> None:
    tensor = torch.tensor([[0.0] * 127 + [448.0], [1e-8] * 128], dtype=torch.float32)
    stats = activation_block_statistics(tensor)
    assert stats["rows"] == 2
    assert stats["blocks_per_row"] == 1
    assert stats["zero_block_fraction"] == 0.0
    assert stats["scale_exponent_min"] <= stats["scale_exponent_max"]
    assert 0.0 <= stats["quantized_zero_fraction"] <= 1.0
    assert 0.0 <= stats["saturation_fraction"] <= 1.0
