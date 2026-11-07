"""Resolve a colocated native checkout and verified dependencies before search."""
from dataclasses import replace
from pathlib import Path

from .errors import NativeUnavailable


def prepare_sources(options):
    if options.runner_path is not None:
        return options
    if options.source_dir is None:
        checkout = Path(__file__).resolve().parents[3]
        if (checkout / "tools" / "stream1_mlp_libtorch_backend.hpp").is_file():
            options = replace(options, source_dir=checkout)
        else:
            bundled = Path(__file__).resolve().parent / "_native_source"
            if not (bundled / "tools/stream1_mlp_libtorch_backend.hpp").is_file():
                raise NativeUnavailable("native sources are missing from this installation; reinstall the complete native wheel")
            import hashlib
            import json
            from .errors import NativeBackendError
            manifest = json.loads((bundled / "source_manifest.json").read_text(encoding="utf-8"))
            for name, expected in manifest.items():
                candidate = bundled / name
                if (not candidate.resolve().is_relative_to(bundled.resolve()) or candidate.is_symlink()
                        or not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected):
                    raise NativeBackendError("bundled native source integrity check failed")
            options = replace(options, source_dir=bundled)
    if options.cutlass_dir is None:
        from .sources import setup_sources
        options = setup_sources(options=options)
    return options
