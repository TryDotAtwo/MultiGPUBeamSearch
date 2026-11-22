"""Check that the one-read loader keeps supported ordered JSON formats."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('--state-len', type=int, required=True)
    parser.add_argument('--moves', type=int, required=True)
    parser.add_argument('--tmpdir', type=Path, required=True)
    args = parser.parse_args()
    names = [f'm{i}' for i in range(args.moves)]
    moves = [[(i + move) % args.state_len for i in range(args.state_len)]
             for move in range(args.moves)]
    ordered = dict(zip(names, moves))
    cases = [dict(generators=ordered), dict(generators=ordered, names=names),
             dict(generators=ordered, move_names=names)]
    cases += [{key: moves, field: names} for key in ('actions', 'moves', 'generators')
              for field in ('names', 'move_names')]
    with tempfile.TemporaryDirectory(dir=args.tmpdir) as directory:
        fixture = Path(directory) / 'moves.json'
        for case in cases:
            fixture.write_text(json.dumps(case) + '\n')
            result = subprocess.run([str(args.binary.resolve()), '--snapshot', str(fixture)],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, (case.keys(), result.stderr)
            assert result.stdout.strip() == 'native_generator_snapshot PASS'
        bad = dict(generators=ordered, names=list(reversed(names)))
        fixture.write_text(json.dumps(bad) + '\n')
        result = subprocess.run([str(args.binary.resolve()), '--snapshot', str(fixture)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and 'move order mismatch' in result.stderr
    print(f'NATIVE_GENERATOR_FORMATS_OK valid={len(cases)} rejected=1')


if __name__ == '__main__':
    main()
