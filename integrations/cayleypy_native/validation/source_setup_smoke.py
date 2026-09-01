"""Exercise the installed wheel's real public downloads and offline cache reuse."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    from cayleypy_native import NativeOptions, setup_sources
    from cayleypy_native.sources import NATIVE_SOURCE, CUTLASS_SOURCE

    cache = args.cache_dir.resolve()
    if (cache / "sources").exists():
        raise ValueError("use a fresh cache directory for a download acceptance run")
    options = setup_sources(options=NativeOptions(cache_dir=cache, build_jobs=3))
    offline = setup_sources(options=replace(options, source_dir=None, cutlass_dir=None), offline=True)
    assert offline == options and options.build_jobs == 3
    assert not (cache / "runs").exists() and not (cache / "builds").exists()
    report = {"status": "passed", "setup": "real HTTPS downloads plus checked offline reuse", "sources": {}}
    for spec, path in ((NATIVE_SOURCE, options.source_dir), (CUTLASS_SOURCE, options.cutlass_dir)):
        archive = path.parent / "source.tar.gz"
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        assert actual == spec.sha256
        report["sources"][spec.name] = {"revision": spec.revision, "sha256": actual,
                                         "archive_bytes": archive.stat().st_size}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
