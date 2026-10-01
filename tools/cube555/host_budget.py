"""Available history RAM bounded by the process's visible memory controllers."""
from pathlib import Path, PurePosixPath
import re


def available_ram_bytes(host_available, proc_root=Path('/proc/self')):
    memberships = {}
    try:
        membership_lines = (proc_root / 'cgroup').read_text().splitlines()
        mount_lines = (proc_root / 'mountinfo').read_text().splitlines()
    except OSError as error:
        raise RuntimeError('Cannot resolve process memory controller') from error
    for line in membership_lines:
        _, controllers, member = line.split(':', 2)
        if not controllers:
            memberships['cgroup2'] = member
        elif 'memory' in controllers.split(','):
            memberships['cgroup'] = member
    for line in mount_lines:
        before, after = line.split(' - ', 1)
        fields, fs = before.split(), after.split()
        kind = fs[0]
        if kind not in memberships or (kind == 'cgroup' and
                'memory' not in fs[-1].split(',')):
            continue
        decode = lambda value: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)
        mount = Path(decode(fields[4]))
        try:
            relative = PurePosixPath(memberships[kind]).relative_to(
                PurePosixPath(decode(fields[3])))
        except ValueError:
            continue
        if '..' in relative.parts:
            continue
        leaf = mount.joinpath(*relative.parts)
        usage_name, limit_name = (('memory.current', 'memory.max') if kind == 'cgroup2'
                                  else ('memory.usage_in_bytes', 'memory.limit_in_bytes'))
        if not (leaf / usage_name).is_file() or not (leaf / limit_name).is_file():
            continue
        remaining = int(host_available)
        current = leaf
        while True:
            limit_file = current / limit_name
            if limit_file.is_file():
                limit = limit_file.read_text().strip()
                if limit != 'max':
                    # A parent's usage includes siblings; subtract its own usage.
                    used = int((current / usage_name).read_text().strip())
                    remaining = min(remaining, max(0, int(limit) - used))
            if current == mount:
                return remaining
            current = current.parent
    raise RuntimeError('Cannot resolve process memory controller; host MemAvailable alone '
                       'cannot safely size Cube555 history RAM')
