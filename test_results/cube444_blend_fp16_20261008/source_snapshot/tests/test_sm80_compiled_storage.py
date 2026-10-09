"""Remote-only actual independent compiled Params table, never synthetic sizes."""
import json
import os
import subprocess

from tools.gemm_parameter_storage import validate_gemm_parameter_storage


def test_compiled_sm80_argument_storage_covers_full_and_split_routes():
    value = json.loads(subprocess.check_output(
        [os.environ['GEMM_STORAGE_PROBE'], '--sm80'], text=True, timeout=30))
    assert value['scope'] == 'compiled_SM80_classic_gemm_parameters_not_admission'
    assert len(validate_gemm_parameter_storage(value)) == 5
    descriptors = {(row['family'], tuple(row['tile']), row['stages']) for row in value['entries']}
    assert descriptors == {
        ('gemm_residual', (128, 64, 32), 3),
        ('gemm_pipelined', (128, 64, 32), 2),
        ('gemm_bias', (128, 64, 32), 3),
        ('gemm_relu', (128, 64, 32), 3),
        ('gemm_bias', (128, 128, 32), 3),
    }
