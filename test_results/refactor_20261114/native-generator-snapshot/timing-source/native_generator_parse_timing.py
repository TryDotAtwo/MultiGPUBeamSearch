"""Five alternating uninstrumented 100-load timing pairs per fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    for name in ("baseline", "candidate", "fixture", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    runners = {name: getattr(args, name).resolve() for name in ("baseline", "candidate")}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = {name: digest(path) for name, path in runners.items()}
    fixture_hash = digest(args.fixture)
    pairs = []
    for pair in range(5):
        samples = {}
        for name in (("baseline", "candidate") if pair % 2 == 0
                     else ("candidate", "baseline")):
            samples[name] = json.loads(subprocess.check_output(
                [str(runners[name]), str(args.fixture.resolve())], text=True))
        assert samples['baseline']['checksum'] == samples['candidate']['checksum']
        assert samples['baseline']['repeats'] == samples['candidate']['repeats'] == 100
        samples['change'] = samples['candidate']['seconds'] / samples['baseline']['seconds'] - 1
        pairs.append(samples)
    assert before == {name: digest(path) for name, path in runners.items()}
    assert fixture_hash == digest(args.fixture)
    result = dict(fixture_sha256=fixture_hash, binary_sha256=before, pairs=pairs,
        median_paired_change=statistics.median(pair['change'] for pair in pairs),
        median_seconds={name: statistics.median(pair[name]['seconds'] for pair in pairs)
                        for name in runners}, scope='five alternating 100-load pairs, uninstrumented')
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['median_seconds'], sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
