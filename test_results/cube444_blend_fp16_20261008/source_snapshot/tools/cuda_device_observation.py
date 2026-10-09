"""Observe CUDA-visible device mapping twice; not future execution attestation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cube4_profile import unique
from tools.loaded_view_contract import validate_visible_device_map
from tools.nccl_library_observation import digest

def query(probe):
    # Spool instead of capture_output: untrusted child output cannot grow RAM.
    import tempfile
    if os.name != 'posix':
        raise ValueError('bounded CUDA probe execution requires POSIX')
    import resource
    def cap_output_file():
        resource.setrlimit(resource.RLIMIT_FSIZE, (262144, 262144))
    with tempfile.TemporaryFile() as output:
        subprocess.run([str(probe)], stdout=output, stderr=subprocess.DEVNULL,
                       check=True, timeout=15, preexec_fn=cap_output_file)
        if output.tell() > 262144:
            raise ValueError('CUDA device map exceeds output bound')
        output.seek(0)
        result = json.loads(output.read(), object_pairs_hook=unique)
    validate_visible_device_map(result)
    return result

def observe(probe):
    probe = Path(probe).resolve(strict=True)
    fingerprint = digest(probe)
    mapping = query(probe)
    if query(probe) != mapping or digest(probe) != fingerprint:
        raise ValueError('CUDA probe or visible mapping changed during observation')
    return dict(schema_version=1, probe_sha256=fingerprint, visible_devices=mapping,
        production_quality_accepted=False,
        scope='two CUDA map observations and probe hash; not future device attestation')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe',type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(observe(args.probe),indent=2))
        return 0
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        parser.exit(2,'CUDA device observation rejected: '+str(error)+'\n')
if __name__=='__main__': raise SystemExit(main())
