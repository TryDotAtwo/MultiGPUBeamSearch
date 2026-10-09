"""The intended fused-input profile must reach the actual runner receipt."""
import json,os
from pathlib import Path
import pytest

def test_fused_input_profile_reaches_actual_runner():
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    if not selected or os.environ.get('BEAM_TEST_EXPECT_FUSED_INPUT_LAYERNORM')!='1':
        pytest.skip('explicit authorized real fused-input capture required')
    execution=json.loads((Path(selected)/'execution.json').read_bytes())
    assert execution['requested_launch_policies']['BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM']=='1', \
        'intended input fusion never reached the real runner'
