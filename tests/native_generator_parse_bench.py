"""Compare old and new generator loaders using identical ordered input.

Both executables are built from native_generator_parse_benchmark.cpp. The
reported heap counters cover ordinary C++ new/delete calls, not process RSS,
allocator overhead, over-aligned allocations, or GPU memory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    for key in ("baseline", "candidate", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--state-len", type=int, required=True)
    parser.add_argument("--moves", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    generators = {f"m{move}": [(i + move) % args.state_len
                                 for i in range(args.state_len)]
                  for move in range(args.moves)}
    fixture = args.out / "moves.json"
    fixture.write_text(json.dumps(dict(generators=generators)) + "\n")
    runners = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    rows = []
    for pair in range(5):
        observations = {}
        for name in (("baseline", "candidate") if pair % 2 == 0
                     else ("candidate", "baseline")):
            data = subprocess.check_output([str(runners[name]), str(fixture)], text=True)
            observations[name] = json.loads(data)
            assert observations[name]["repeats"] == 20
        assert observations["baseline"]["checksum"] == observations["candidate"]["checksum"]
        rows.append(observations)
    fields = ("seconds", "ordinary_allocations", "ordinary_requested_bytes",
              "ordinary_peak_live_bytes")
    medians = {name: {field: statistics.median(row[name][field] for row in rows)
                      for field in fields} for name in runners}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(state_len=args.state_len, moves=args.moves,
                  fixture_sha256=digest(fixture),
                  binaries={name: digest(path) for name, path in runners.items()},
                  pairs=rows, medians=medians,
                  scope="ordinary C++ new/delete only; five alternating 20-load CPU pairs")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["medians"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
