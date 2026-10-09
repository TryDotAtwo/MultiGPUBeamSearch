"""Observe runner-selected NCCL bytes; not collective or numerical admission."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

try:
    from tools.cube4_profile import unique
except ModuleNotFoundError:
    from cube4_profile import unique


def digest(path):
    result = sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def query(runner):
    result = subprocess.run([str(runner), '--runtime-library-info'], check=True,
                            capture_output=True, text=True, timeout=15)
    if len(result.stdout) > 32768:
        raise ValueError('runtime library receipt too large')
    info = json.loads(result.stdout, object_pairs_hook=unique)
    if (not isinstance(info, dict) or set(info) != {'schema_version', 'nccl_version',
            'nccl_library_path', 'communicator_initialized'} or
            type(info['schema_version']) is not int or info['schema_version'] != 1 or
            type(info['nccl_version']) is not int or info['nccl_version'] <= 0 or
            info['communicator_initialized'] is not False or
            not isinstance(info['nccl_library_path'], str)):
        raise ValueError('invalid runtime library receipt')
    path = Path(info['nccl_library_path'])
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('runtime NCCL path is not canonical absolute path')
    return info


def observe(runner):
    runner = runner.resolve(strict=True)
    runner_hash = digest(runner)
    info = query(runner)
    library = Path(info['nccl_library_path'])
    library_hash = digest(library)
    if (query(runner) != info or digest(library) != library_hash or
            digest(runner) != runner_hash):
        raise ValueError('runner or selected NCCL changed during observation')
    return dict(schema_version=1, runner_sha256=runner_hash,
        loaded_library=info, nccl_library_sha256=library_hash,
        scope='two loader observations and file hashes; not future rank/library attestation',
        production_quality_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runner', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(observe(args.runner), indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, 'NCCL observation rejected: ' + str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
