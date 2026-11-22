"""Compare every FP16 and BF16 bit pattern with the installed PyTorch."""
import json
from pathlib import Path
import sys

import torch
from multigpubeamsearch import _beam_runtime


def main():
    assert sys.byteorder == 'little'
    raw = b''.join(i.to_bytes(2, 'little') for i in range(65536))
    counts = {}
    for name, dtype in [('fp16', torch.float16), ('bf16', torch.bfloat16)]:
        expected = torch.isfinite(torch.frombuffer(bytearray(raw), dtype=dtype)).tolist()
        observed = [_beam_runtime.mlp_finite_16le(raw[2*i:2*i+2], name) for i in range(65536)]
        assert observed == expected, name
        counts[name] = {'tested': len(observed), 'finite': sum(observed),
                        'nonfinite': len(observed) - sum(observed)}
        assert _beam_runtime.mlp_finite_16le(b'', name)
        bad = b'\x00\x7c' if name == 'fp16' else b'\x80\x7f'
        for offset in (0, 16, 30):
            chunk = bytearray(32)
            chunk[offset:offset+2] = bad
            assert not _beam_runtime.mlp_finite_16le(bytes(chunk), name)
        try:
            _beam_runtime.mlp_finite_16le(b'\x00', name)
        except RuntimeError:
            pass
        else:
            raise AssertionError('odd byte length was accepted')
    for data, dtype in [(b'\x00\x00', 'f32'), (bytearray(2), 'fp16')]:
        try:
            _beam_runtime.mlp_finite_16le(data, dtype)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError('invalid dtype or mutable input was accepted')
    result = {'scope': 'all 16-bit encodings compared individually with PyTorch on CPU',
              'torch': torch.__version__, 'counts': counts}
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
