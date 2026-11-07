"""Check runner-reported build compatibility and record a binary hash, not quality."""
import argparse
import csv
import io
import hashlib
import json
from pathlib import Path
import re
import sys

try:
    from tools.cube4_profile import unique
except ModuleNotFoundError:
    from cube4_profile import unique


def validate_build_info(info, expected_arch='90a'):
    expected = dict(schema_version=1, state_len=96, state_storage_len=112,
                    move_count=24, candidate_meta_bytes=32)
    if not isinstance(info, dict) or set(info) != set(expected) | {'cuda_architectures'}:
        raise ValueError('runner build-info schema mismatch')
    for key, value in expected.items():
        if type(info[key]) is not int or info[key] != value:
            raise ValueError('incompatible runner: ' + key)
    arches = info['cuda_architectures']
    if not isinstance(arches, str) or not re.fullmatch(r'[0-9]+a?(?:-(?:real|virtual))?(?:,[0-9]+a?(?:-(?:real|virtual))?)*', arches):
        raise ValueError('invalid/unknown runner CUDA architectures')
    if not re.fullmatch(r'[0-9]+a?', expected_arch):
        raise ValueError('invalid expected architecture')
    if not {expected_arch, expected_arch + '-real'} & set(arches.split(',')):
        raise ValueError('runner lacks required native architecture: ' + expected_arch)
    return info


def validate_h200_hardware(raw: str, world: int) -> dict:
    """Validate nvidia-smi name,memory.total,compute_cap CSV (MiB, no units)."""
    if type(world) is not int or not 1 <= world <= 8:
        raise ValueError('invalid GPU count')
    if not isinstance(raw, str) or len(raw) > 32768:
        raise ValueError('invalid hardware report')
    rows = list(csv.reader(io.StringIO(raw), skipinitialspace=True))
    if len(rows) != world:
        raise ValueError('GPU count mismatch')
    devices = []
    variants = set()
    for row in rows:
        if len(row) != 3:
            raise ValueError('hardware CSV schema mismatch')
        name, memory, capability = (value.strip() for value in row)
        if name not in ('NVIDIA H200', 'NVIDIA H200 NVL'):
            raise ValueError('expected H200 hardware')
        if not re.fullmatch(r'[0-9]+', memory) or not 120 * 1024 <= int(memory) <= 160 * 1024:
            raise ValueError('incompatible H200 VRAM')
        if capability != '9.0':
            raise ValueError('incompatible H200 compute capability')
        variants.add(name.removeprefix('NVIDIA '))
        devices.append({'name': name, 'vram_mib': int(memory), 'compute_capability': capability})
    if len(variants) != 1:
        raise ValueError('mixed H200 variants require separate profile')
    return {'gpu_count': world, 'variant': variants.pop(), 'devices': devices,
            'quality_accepted': False,
            'scope': 'reported GPU compatibility, not topology or inference acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runner', type=Path)
    parser.add_argument('build_info', type=Path)
    parser.add_argument('--expected-arch', default='90a')
    parser.add_argument('--hardware-report', type=Path)
    parser.add_argument('--world-size', type=int)
    args = parser.parse_args()
    try:
        info = json.loads(args.build_info.read_text(), object_pairs_hook=unique)
        validate_build_info(info, args.expected_arch)
        if (args.hardware_report is None) != (args.world_size is None):
            raise ValueError('hardware-report and world-size must be paired')
        hardware = None
        if args.hardware_report is not None:
            hardware = validate_h200_hardware(args.hardware_report.read_text(), args.world_size)
        digest = hashlib.sha256()
        with args.runner.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        print(json.dumps({'build_info': info, 'runner_sha256': digest.hexdigest(),
                          'hardware': hardware,
                          'quality_accepted': False,
                          'scope': 'runner-reported compile compatibility; not hardware execution'}, indent=2))
        return 0
    except (ValueError, OSError) as error:
        print('build rejected: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
