#!/bin/bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/repo}"
RUN_DIR="${RUN_DIR:-${BASE_DIR}/ihes_cube_model}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model.pth}"
MODEL_RELEASE_REPO="${MODEL_RELEASE_REPO:-TryDotAtwo/MultiGPUBeamSearch}"
MODEL_RELEASE_TAG="${MODEL_RELEASE_TAG:-ihes-p888-model}"
MODEL_RELEASE_ASSET="${MODEL_RELEASE_ASSET:-p888-t000_1778521793_e32692.pth}"
DATA_RELEASE_REPO="${DATA_RELEASE_REPO:-${MODEL_RELEASE_REPO}}"
DATA_RELEASE_TAG="${DATA_RELEASE_TAG:-${MODEL_RELEASE_TAG}}"
DATA_RELEASE_ASSET="${DATA_RELEASE_ASSET:-cayleypy-ihes-cube.zip}"
WEIGHT_DIR="${WEIGHT_DIR:-${RUN_DIR}/stream1_weights_ihes_bf16}"
NINJA_VENV_DIR="${NINJA_VENV_DIR:-${VIRTUAL_ENV:-/mnt/pool/3/vokirova/ninja-venv}}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${WEIGHT_DIR}"

if [ -f "${NINJA_VENV_DIR}/bin/activate" ]; then
  source "${NINJA_VENV_DIR}/bin/activate"
fi
if [ ! -x "${NINJA_VENV_DIR}/bin/python" ]; then
  echo "missing_python=${NINJA_VENV_DIR}/bin/python"
  exit 2
fi
if [ ! -d "${REPO_DIR}/.git" ]; then
  echo "missing_repo=${REPO_DIR}"
  exit 2
fi

echo "repo_dir=${REPO_DIR}"
echo "run_dir=${RUN_DIR}"
echo "model_path=${MODEL_PATH}"
echo "data_dir=${DATA_DIR}"
echo "weight_dir=${WEIGHT_DIR}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "download_model_from_github=1"
  if command -v gh >/dev/null 2>&1; then
    gh release download "${MODEL_RELEASE_TAG}" \
      --repo "${MODEL_RELEASE_REPO}" \
      --pattern "${MODEL_RELEASE_ASSET}" \
      --dir "${RUN_DIR}" \
      --clobber
  else
    "${NINJA_VENV_DIR}/bin/python" - "${MODEL_RELEASE_REPO}" "${MODEL_RELEASE_TAG}" "${MODEL_RELEASE_ASSET}" "${RUN_DIR}" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

repo, tag, asset_name, run_dir = sys.argv[1:]
api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
with urllib.request.urlopen(api, timeout=60) as response:
    release = json.load(response)
for asset in release["assets"]:
    if asset["name"] == asset_name:
        url = asset["browser_download_url"]
        break
else:
    raise RuntimeError(f"asset not found: {asset_name}")
target = Path(run_dir) / asset_name
with urllib.request.urlopen(url, timeout=300) as response:
    target.write_bytes(response.read())
print(f"downloaded_model_asset={target}")
PY
  fi
  if [ ! -f "${RUN_DIR}/${MODEL_RELEASE_ASSET}" ]; then
    echo "missing_downloaded_model=${RUN_DIR}/${MODEL_RELEASE_ASSET}"
    exit 2
  fi
  mv -f "${RUN_DIR}/${MODEL_RELEASE_ASSET}" "${MODEL_PATH}"
else
  echo "download_model_from_github=0"
fi

if [ ! -f "${DATA_DIR}/puzzle_info.json" ] || [ ! -f "${DATA_DIR}/test.csv" ]; then
  echo "download_ihes_cube_data_from_github=1"
  if command -v gh >/dev/null 2>&1; then
    gh release download "${DATA_RELEASE_TAG}" \
      --repo "${DATA_RELEASE_REPO}" \
      --pattern "${DATA_RELEASE_ASSET}" \
      --dir "${DATA_DIR}" \
      --clobber
  else
    "${NINJA_VENV_DIR}/bin/python" - "${DATA_RELEASE_REPO}" "${DATA_RELEASE_TAG}" "${DATA_RELEASE_ASSET}" "${DATA_DIR}" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

repo, tag, asset_name, data_dir = sys.argv[1:]
api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
with urllib.request.urlopen(api, timeout=60) as response:
    release = json.load(response)
for asset in release["assets"]:
    if asset["name"] == asset_name:
        url = asset["browser_download_url"]
        break
else:
    raise RuntimeError(f"asset not found: {asset_name}")
target = Path(data_dir) / asset_name
with urllib.request.urlopen(url, timeout=300) as response:
    target.write_bytes(response.read())
print(f"downloaded_data_asset={target}")
PY
  fi
  if [ ! -f "${DATA_DIR}/${DATA_RELEASE_ASSET}" ]; then
    echo "missing_downloaded_data=${DATA_DIR}/${DATA_RELEASE_ASSET}"
    exit 2
  fi
  unzip -o "${DATA_DIR}/${DATA_RELEASE_ASSET}" -d "${DATA_DIR}"
else
  echo "download_ihes_cube_data_from_github=0"
fi

if [ ! -f "${WEIGHT_DIR}/manifest.json" ] || [ "${MODEL_PATH}" -nt "${WEIGHT_DIR}/manifest.json" ]; then
  echo "export_stream1_weights=1"
  "${NINJA_VENV_DIR}/bin/python" "${REPO_DIR}/tools/export_stream1_mlp.py" \
    --weights "${MODEL_PATH}" \
    --out "${WEIGHT_DIR}" \
    --format batchnorm-folded \
    --dtype bf16 \
    --num-classes 72
else
  echo "export_stream1_weights=0"
fi

"${NINJA_VENV_DIR}/bin/python" - "${DATA_DIR}/puzzle_info.json" "${WEIGHT_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
manifest = json.loads(Path(sys.argv[2]).read_text())
move_count = len(info["generators"])
print(f"prepared_state_len={manifest['state_len']}")
print(f"prepared_num_classes={manifest['num_classes']}")
print(f"prepared_output_dim={manifest['output_dim']}")
print(f"prepared_move_count={move_count}")
if manifest["state_len"] != 72 or manifest["num_classes"] != 72:
    raise SystemExit("IHES manifest dimensions mismatch")
if manifest["output_dim"] != 1:
    raise SystemExit("IHES model must be one-output")
if move_count != 18:
    raise SystemExit("IHES move count mismatch")
PY

echo "prepare_done=1"
