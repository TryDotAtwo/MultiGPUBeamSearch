"""Observe filesystem free space before rank startup; not a reservation/quota guarantee."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys


def unsigned(value):
    if (not re.fullmatch(r'0|[1-9][0-9]*', value) or len(value) > 20 or
        (len(value) == 20 and value > '18446744073709551615')):
        raise argparse.ArgumentTypeError('expected canonical uint64 bytes')
    return int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--history-disk-bytes', type=unsigned, required=True)
    parser.add_argument('--headroom-bytes', type=unsigned, required=True)
    args = parser.parse_args()
    try:
        path = args.path.resolve(strict=True)
        if not path.is_dir():
            raise ValueError('filesystem probe requires existing directory')
        required = args.history_disk_bytes + args.headroom_bytes
        if required > 18446744073709551615:
            raise ValueError('disk requirement exceeds uint64')
        available = shutil.disk_usage(path).free
        if available < required:
            raise ValueError(f'insufficient filesystem space: available={available} required={required}')
        print(json.dumps({'path': str(path), 'available_bytes': available,
            'required_bytes': required, 'history_disk_bytes': args.history_disk_bytes,
            'headroom_bytes': args.headroom_bytes, 'physical_reservation_confirmed': False}))
        return 0
    except (OSError, ValueError) as error:
        print('disk preflight rejected: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
