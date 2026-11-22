#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python3.11}"
env_dir="$here/.venv"
if [[ ! -f "$here/SHA256SUMS" ]]; then
    echo "Use the extracted runtime bundle produced by prepare_bundle.sh." >&2
    exit 1
fi
(
    flock 9
    if [[ ! -f "$env_dir/.installed" ]]; then
        version="$("$python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "$version" != "3.11" ]]; then
            echo "This bundle needs Python 3.11. Set PYTHON to its executable." >&2
            exit 1
        fi
        (cd -- "$here" && sha256sum --quiet -c SHA256SUMS)
        "$python" -m venv "$env_dir"
        "$env_dir/bin/python" -m pip install --no-index --only-binary=:all: \
            --find-links "$here/wheels" \
            'torch==2.8.0+cu128' 'cayleypy==0.2.0' 'multigpubeamsearch==0.1.0.dev1'
        "$env_dir/bin/python" -m pip check
        touch "$env_dir/.installed"
    fi
) 9>"$here/.install.lock"

export PATH="$env_dir/bin:$PATH"
exec "$env_dir/bin/python" -u "$here/solve_cube555.py" "$@"
