"""Exercise the Rust worker's native ABI inspection in a child process."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--state-len', type=int, required=True)
    parser.add_argument('--storage-len', type=int, required=True)
    parser.add_argument('--move-count', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ, BEAM_NATIVE_LIBRARY=str(args.library.resolve()))
    result = subprocess.run([str(args.runner.resolve()), '--native-abi-info'],
                            env=env, capture_output=True, text=True, timeout=30, check=True)
    info = json.loads(result.stdout)
    expected = dict(abi_version=1, state_len=args.state_len, storage_len=args.storage_len,
                    state_alignment=16, move_count=args.move_count, candidate_bytes=32,
                    candidate_alignment=32, hash_bytes=16, hash_alignment=16)
    assert info == expected, (info, expected)
    missing = args.out.parent / 'nonexistent-native-library.so'
    env['BEAM_NATIVE_LIBRARY'] = str(missing)
    failure = subprocess.run([str(args.runner.resolve()), '--native-abi-info'],
                             env=env, capture_output=True, text=True, timeout=30, check=False)
    assert failure.returncode != 0 and str(missing) in failure.stderr
    args.out.write_text(json.dumps(dict(layout=info, missing_library_exit=failure.returncode,
                                        missing_library_stderr=failure.stderr), indent=2) + '\n')
    print('NATIVE_ABI_PROBE_OK', json.dumps(info, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
