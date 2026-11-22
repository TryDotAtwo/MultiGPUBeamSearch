#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python3.11}"
solver_rev=534f63516552081e48f5c575c6471a888c9db5b9
cayleypy_rev=3cd11ecefc98e54d6fc0ccdec967d0bcc8ca4a93
solver_url=https://github.com/TryDotAtwo/MultiGPUBeamSearch.git
torch_index=https://download.pytorch.org/whl/cu128
archive="${1:-cube555-standalone-linux-x86_64-py311.tar}"
archive="$(realpath -m -- "$archive")"
mkdir -p -- "$(dirname -- "$archive")"
if [[ -e "$archive" || -e "$archive.sha256" ]]; then
    echo "Output already exists: $archive or $archive.sha256" >&2
    exit 1
fi

python_target="$("$python" -c 'import platform, sys; print(platform.system(), platform.machine(), platform.python_implementation(), sys.version_info[:2])')"
if [[ "$python_target" != 'Linux x86_64 CPython (3, 11)' ]]; then
    echo "Need Linux x86_64 and CPython 3.11, got $python_target." >&2
    exit 1
fi
staging="$(mktemp -d)"
partial_archive=""
trap 'if [[ -n "$partial_archive" ]]; then rm -f -- "$partial_archive"; fi; rm -rf -- "$staging"' EXIT
source_dir="$staging/source"
bundle="$staging/cube555-standalone"
pinned="$staging/pinned-wheels"
mkdir -p "$bundle/wheels" "$pinned"
cp -- "$here/solve_cube555.py" "$here/run.sh" "$here/README.md" "$bundle/"
chmod +x "$bundle/run.sh"

git init -q "$source_dir"
git -C "$source_dir" fetch -q --depth 1 "$solver_url" "$solver_rev"
git -C "$source_dir" checkout -q --detach "$solver_rev"
"$python" -m venv "$staging/build-env"
build_python="$staging/build-env/bin/python"
wheels="$bundle/wheels"

"$build_python" -m pip wheel --no-cache-dir --no-deps --wheel-dir "$pinned" \
    "cayleypy @ https://github.com/cayleypy/cayleypy/archive/$cayleypy_rev.tar.gz" \
    "$source_dir/integrations/cayleypy_native"
"$build_python" -m pip wheel --no-cache-dir --only-binary=:all: \
    --wheel-dir "$wheels" \
    --index-url https://pypi.org/simple --extra-index-url "$torch_index" \
    'torch==2.8.0+cu128' "$pinned"/*.whl
"$build_python" -m pip install --no-index --only-binary=:all: --find-links "$wheels" \
    'torch==2.8.0+cu128' 'cayleypy==0.2.0' 'multigpubeamsearch==0.1.0.dev1'
"$build_python" -m pip check

PYTHONPATH="$source_dir" "$build_python" "$here/prepare_assets.py" \
    "$bundle/assets" "$source_dir" "$staging/cache"
"$build_python" "$bundle/solve_cube555.py" --check

(
    cd -- "$bundle"
    find assets wheels solve_cube555.py run.sh README.md -type f -print0 \
        | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
)
partial_archive="$(mktemp "$archive.XXXXXX")"
tar -C "$staging" -cf "$partial_archive" cube555-standalone
mv -T -- "$partial_archive" "$archive"
(
    cd -- "$(dirname -- "$archive")"
    sha256sum "$(basename -- "$archive")" > "$(basename -- "$archive").sha256"
)
printf 'Bundle ready: %s\n' "$archive"
