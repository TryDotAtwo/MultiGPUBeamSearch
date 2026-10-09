"""Record launch selectors including unset defaults; not resolved-kernel admission."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re

SELECTOR = re.compile(r'\b(?:BEAM_STREAM1_[A-Z0-9_]+|BEAM_HOPPER_NATIVE_FP8)\b')


def observe_environment(source_root, environment):
    sources, names = {}, set()
    for directory in ('cuda', 'include', 'src', 'tools'):
        base = source_root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix in ('.cu', '.cpp', '.hpp', '.h'):
                raw = path.read_bytes()
                selectors = set(SELECTOR.findall(raw.decode('utf-8')))
                if selectors:
                    sources[path.relative_to(source_root).as_posix()] = sha256(raw).hexdigest()
                    names.update(selectors)
    if not sources:
        raise ValueError('no native selector sources found')
    inherited = {key: value for key, value in environment.items() if SELECTOR.fullmatch(key)}
    unknown = sorted(set(inherited) - names)
    if unknown:
        raise ValueError('unknown native Stream1 selector: ' + ', '.join(unknown))
    return dict(schema_version=1,
        scope='source selector inventory and actual environment; not resolved kernel or numeric admission',
        production_quality_accepted=False,
        selectors={key: environment.get(key) for key in sorted(names)},
        selector_sources_sha256=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(observe_environment(args.source_root, os.environ), indent=2))
        return 0
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, 'execution environment rejected: ' + str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
