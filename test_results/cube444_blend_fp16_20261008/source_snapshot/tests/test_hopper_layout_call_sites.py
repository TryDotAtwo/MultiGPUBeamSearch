"""Structural caller contract; not runtime/numerical GPU evidence."""
from pathlib import Path
import re


def test_every_qkv_ff1_helper_call_forwards_loaded_block_layout():
    source = (Path(__file__).resolve().parents[1]/'cuda/stream1_transformer.cu').read_text()
    for name,flag in [('stream1_transformer_linear_bias_cuda','qkv_hopper_fp16'),
                      ('stream1_transformer_ff1_linear_bias_activation_cuda','ff1_hopper_fp16')]:
        calls=[]
        for match in re.finditer(r'\b'+name+r'\s*\(',source):
            start=match.end()
            depth=1
            end=start
            while depth:
                assert end < len(source), 'unterminated helper call/signature'
                if source[end]=='(': depth+=1
                elif source[end]==')': depth-=1
                end+=1
            suffix=source[end:].lstrip()
            if suffix.startswith('{'): # helper definition, not a caller
                continue
            assert suffix.startswith(';'), 'unexpected helper use requires audit'
            calls.append(source[start:end-1])
        assert len(calls)==4, 'changed caller inventory requires explicit audit'
        for arguments in calls:
            assert re.search(r',\s*block\.'+flag+r'\s*$',arguments), arguments
