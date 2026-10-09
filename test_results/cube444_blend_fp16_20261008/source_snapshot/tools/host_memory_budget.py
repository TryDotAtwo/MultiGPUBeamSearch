"""Conservative RAM availability across visible cgroup v1/v2 ancestors."""
from pathlib import Path
import os


def explicit_ram_cap() -> int | None:
    text = os.environ.get('BEAM_HOST_RAM_CAP_BYTES')
    if text is None:
        return None
    if not text.isascii() or not text.isdecimal():
        raise ValueError('BEAM_HOST_RAM_CAP_BYTES must be positive uint64 bytes')
    cap = int(text)
    if not 0 < cap <= (1 << 64) - 1:
        raise ValueError('BEAM_HOST_RAM_CAP_BYTES must be positive uint64 bytes')
    return cap


def clamp_explicit_ram_cap(available: int) -> int:
    cap = explicit_ram_cap()
    return available if cap is None else min(available, cap)


def clamp_memory_to_cgroup(host_available: int, root: Path) -> int:
    if host_available < 0:
        raise ValueError('host available memory must be nonnegative')
    available = host_available
    locations = [(root, True), (root / 'memory', False)]
    membership = Path('/proc/self/cgroup')
    if membership.is_file():
        try:
            rows = membership.read_text().splitlines()
        except OSError as error:
            raise RuntimeError('cannot read process cgroup membership') from error
        for row in rows:
            fields = row.split(':', 2)
            if len(fields) != 3:
                continue
            _, controller_names, member_path = fields
            version2 = controller_names == ''
            if not version2 and 'memory' not in controller_names.split(','):
                continue
            parts = member_path.split('/')
            if not member_path.startswith('/') or '..' in parts or '.' in parts:
                raise RuntimeError('invalid process memory cgroup path')
            base = root if version2 else root / 'memory'
            current = base.joinpath(*(part for part in parts if part))
            while current != base:
                locations.append((current, version2))
                current = current.parent
    controllers = ((location / ('memory.max' if v2 else 'memory.limit_in_bytes'),
                    location / ('memory.current' if v2 else 'memory.usage_in_bytes'), v2)
                   for location, v2 in locations)
    for limit_path, usage_path, version2 in controllers:
        if not limit_path.exists() and not usage_path.exists():
            continue
        try:
            limit_text = limit_path.read_text().strip()
            if version2 and limit_text == 'max':
                continue
            usage_text = usage_path.read_text().strip()
            if not limit_text.isascii() or not limit_text.isdecimal() or not usage_text.isascii() or not usage_text.isdecimal():
                raise ValueError('nondecimal cgroup memory value')
            limit, usage = int(limit_text), int(usage_text)
            # v1 represents an unlimited limit by a page-aligned signed-long max.
            if not version2 and limit >= (1 << 60):
                continue
            available = min(available, max(0, limit - usage))
        except (OSError, ValueError) as error:
            raise RuntimeError(f'cannot determine detected cgroup memory limit: {limit_path}') from error
    return available


def main() -> int:
    import argparse
    import json
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--meminfo', type=Path, default=Path('/proc/meminfo'))
    parser.add_argument('--cgroup-root', type=Path, default=Path('/sys/fs/cgroup'))
    parser.add_argument('--history-ram-bytes', type=int, required=True)
    parser.add_argument('--headroom-bytes', type=int, required=True)
    args = parser.parse_args()
    try:
        if not (0 < args.history_ram_bytes <= (1 << 64) - 1 and
                0 < args.headroom_bytes <= (1 << 64) - 1):
            raise ValueError('history RAM and headroom must be positive uint64 byte budgets')
        host = None
        for row in args.meminfo.read_text().splitlines():
            fields = row.split()
            if len(fields) == 3 and fields[0] == 'MemAvailable:' and fields[1].isascii() and fields[1].isdecimal() and fields[2] == 'kB':
                host = int(fields[1]) * 1024
                break
        if host is None:
            raise ValueError('MemAvailable missing or malformed')
        available = clamp_memory_to_cgroup(host, args.cgroup_root)
        cap = explicit_ram_cap()
        if cap is not None:
            available = min(available, cap)
        if args.history_ram_bytes > available or args.headroom_bytes > available - args.history_ram_bytes:
            raise ValueError('insufficient container RAM for history plus explicit headroom')
        print(json.dumps({'host_available_bytes': host, 'available_bytes': available,
            'explicit_ram_cap_bytes': cap,
            'history_ram_bytes': args.history_ram_bytes, 'headroom_bytes': args.headroom_bytes,
            'remaining_after_history_bytes': available - args.history_ram_bytes,
            'production_accepted': False,
            'scope': 'availability snapshot and declared headroom, not physical RAM reservation'}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
