#!/usr/bin/env bash
set -euo pipefail

source_dir="$1"
run_root="$2"
result_dir="$3"
source_sha="$4"
instance_id="$5"
manifest_sha="$6"
checkpoint_sha="$7"
acceptance_run_id="$8"
handoff_run_root="$9"

source "$source_dir/tools/vast_cnot_panel_cli_args.sh"

python_bin="$(command -v python3)"
command -v cargo >/dev/null
command -v cmake >/dev/null
command -v ninja >/dev/null
command -v git >/dev/null
cuda_path="${CUDA_PATH:-/usr/local/cuda}"
[[ -x "${cuda_path}/bin/nvcc" ]]
[[ "$handoff_run_root" == /private/* && "$handoff_run_root" != *"/../"* && "$handoff_run_root" != */.. ]]
python3 - "$handoff_run_root" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
if not root.is_absolute() or root.parts[:2] != ("/", "private") or len(root.parts) < 3:
    raise SystemExit("private handoff root is outside the approved mount")
if root.resolve(strict=True) != root or root.is_symlink():
    raise SystemExit("private handoff root is not canonical")
reports = root / "reports"
receipt = reports / "learning-curve.private.json"
if reports.is_symlink() or not reports.is_dir() or receipt.is_symlink() or not receipt.is_file():
    raise SystemExit("private handoff receipt is unavailable")
PY
upstream=/private/TheQuantumProject
manifest=/private/gl4/search-qhat.json
expected_upstream=c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb
expected_checkpoint="$checkpoint_sha"
[[ "$(git -C "$upstream" rev-parse HEAD)" == "$expected_upstream" ]]
[[ -s "$manifest" ]]
actual_manifest_sha="$(sha256sum "$manifest" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$manifest_sha" ]] || { echo "private search manifest changed after preflight" >&2; exit 1; }
[[ "$(find /private/gl4/checkpoint -maxdepth 1 -type f -name 'step-*.qcp' -exec sha256sum {} + | awk -v expected="$expected_checkpoint" '$1==expected {n++} END {print n+0}')" == 1 ]]
"$python_bin" - <<'PY'
import re, subprocess, torch
if torch.__version__.split("+")[0] != "2.9.0" or torch.version.cuda is None:
    raise SystemExit("the installed CUDA PyTorch 2.9.0 stack is required")
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("two CUDA devices are required on the verified instance")
names = [torch.cuda.get_device_name(i) for i in range(2)]
if any("RTX 3060" not in name.upper() or "TI" in name.upper() for name in names):
    raise SystemExit(f"expected two desktop RTX 3060 GPUs, found {names!r}")
cuda = re.search(r"release ([0-9]+\.[0-9]+)", subprocess.check_output(["nvcc", "--version"], text=True))
if not cuda or cuda.group(1) != torch.version.cuda or torch.cuda.nccl.version() is None:
    raise SystemExit("nvcc, PyTorch CUDA and NCCL versions are incompatible")
print(f"CUDA_DEVICE_COUNT={torch.cuda.device_count()} GPU_NAMES={names!r} TORCH={torch.__version__} CUDA={torch.version.cuda} NCCL={torch.cuda.nccl.version()}")
PY
mapfile -t physical_gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ "${#physical_gpu_uuids[@]}" == 2 && "${physical_gpu_uuids[0]}" != "${physical_gpu_uuids[1]}" ]] || { echo "two unique physical GPU UUIDs are required" >&2; exit 1; }
export CARGO_TARGET_DIR="${run_root}/cargo-target"
export PYO3_PYTHON="$python_bin"
export CUDA_PATH="$cuda_path"
export CUDACXX="${CUDA_PATH}/bin/nvcc"
export CMAKE_GENERATOR=Ninja
export CNOT103_CUDA_ARCHITECTURES=86
export LIBTORCH_USE_PYTORCH=1
wheel_dir="${run_root}/wheel-dist"
venv="${run_root}/venv"
mkdir -p "$wheel_dir" "${run_root}/wheel-smoke" "$result_dir"
handoff_preflight="${result_dir}/handoff-preflight.json"
panel_result_dir="${result_dir}/panel"
[[ ! -e "$handoff_preflight" && ! -L "$handoff_preflight" && ! -e "$panel_result_dir" && ! -L "$panel_result_dir" ]]
mkdir -p "$panel_result_dir"
"$python_bin" "$source_dir/tools/validate_cnot_handoff.py" \
  --run-root "$handoff_run_root" --source-root "$source_dir" --output "$handoff_preflight"
handoff_preflight_sha="$(sha256sum "$handoff_preflight" | awk '{print $1}')"
mapfile -t panel_pins < <("$python_bin" - "$handoff_preflight" "$handoff_run_root" <<'PY'
import json
import pathlib
import sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
root = pathlib.Path(sys.argv[2]).resolve(strict=True)
relative = pathlib.PurePosixPath(report["search_manifest_relative_path"])
if relative.is_absolute() or ".." in relative.parts:
    raise SystemExit("preflight manifest path is not run-relative")
manifest = root.joinpath(*relative.parts).resolve(strict=True)
if not manifest.is_relative_to(root):
    raise SystemExit("preflight manifest escaped the private run root")
print(report["checkpoint_sha256"])
print(report["search_manifest_sha256"])
print(manifest)
PY
)
[[ "${#panel_pins[@]}" == 3 ]]
panel_checkpoint_sha="${panel_pins[0]}"
panel_manifest_sha="${panel_pins[1]}"
panel_manifest="${panel_pins[2]}"
cd "$source_dir/integrations/cayleypy_native"
cutlass_cache="/private/cache/cayleypy-native"
cutlass_dir="$("$python_bin" - "$source_dir" "$cutlass_cache" <<'PY'
import json, sys
from pathlib import Path

import cayleypy
import torch
from multigpubeamsearch.build import discover_nccl
from multigpubeamsearch.options import NativeOptions
from multigpubeamsearch.sources import CUTLASS_SOURCE, setup_sources

if not hasattr(cayleypy, "CayleyGraph"):
    raise SystemExit("the acceptance Python environment lacks the CayleyPy graph API")
nccl = discover_nccl()
options = setup_sources(
    options=NativeOptions(source_dir=Path(sys.argv[1]), cache_dir=Path(sys.argv[2])),
    offline=True,
)
cutlass = options.cutlass_dir
if cutlass != Path(sys.argv[2]) / "sources" / f"cutlass-{CUTLASS_SOURCE.revision}" / "tree":
    raise SystemExit(f"the CUTLASS source is not the expected pre-staged cache: {cutlass}")
print("REMOTE_BUILD_INPUT_PREFLIGHT=" + json.dumps({
    "cayleypy": getattr(cayleypy, "__version__", "installed"),
    "torch": str(torch.__version__),
    "nccl_header_sha256": nccl["header_sha256"],
    "nccl_library_sha256": nccl["library_sha256"],
    "cutlass_revision": CUTLASS_SOURCE.revision,
    "cutlass_archive_sha256": CUTLASS_SOURCE.sha256,
    "cutlass_source": str(cutlass),
    "source_cache_mode": "offline",
}, sort_keys=True), file=sys.stderr)
print(cutlass)
PY
)"
[[ -s "${cutlass_dir}/include/cutlass/cutlass.h" ]]
"$python_bin" -m pip wheel --verbose --no-deps --wheel-dir "$wheel_dir" .
"$python_bin" -m venv --system-site-packages "$venv"
wheel="$(find "$wheel_dir" -maxdepth 1 -type f -name 'multigpubeamsearch-*.whl' -print -quit)"
test -n "$wheel"
wheel_sha256="$(sha256sum "$wheel" | awk '{print $1}')"
python3 - "$result_dir/receipt.json" "$wheel_sha256" <<'PY'
import json, pathlib, sys
receipt_path = pathlib.Path(sys.argv[1])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
receipt["wheel_sha256"] = sys.argv[2]
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
wheel_python="${venv}/bin/python"
"$wheel_python" -m pip install --force-reinstall --no-deps "$wheel"
"$wheel_python" - "$venv" <<'PY'
import sys
from pathlib import Path
import multigpubeamsearch
venv = Path(sys.argv[1]).resolve()
module_path = Path(multigpubeamsearch.__file__).resolve()
if not module_path.is_relative_to(venv):
    raise SystemExit(f"wheel import escaped the acceptance venv: {module_path}")
print("INSTALLED_PACKAGE_PATH=" + str(module_path))
PY
"$wheel_python" -m pip check
cd "${run_root}/wheel-smoke"
unset PYTHONPATH
"$wheel_python" - "$source_dir" "$cutlass_cache" "$cutlass_dir" <<'PY'
import sys
from pathlib import Path

from multigpubeamsearch.options import NativeOptions
from multigpubeamsearch.sources import setup_sources

options = setup_sources(
    options=NativeOptions(source_dir=Path(sys.argv[1]), cache_dir=Path(sys.argv[2])),
    offline=True,
)
if options.cutlass_dir != Path(sys.argv[3]):
    raise SystemExit("installed wheel resolved a different CUTLASS cache than pre-build admission")
print("INSTALLED_WHEEL_CUTLASS_OFFLINE_RECHECK=" + str(options.cutlass_dir))
PY
GITHUB_WORKSPACE="$source_dir" "$wheel_python" "$source_dir/integrations/cayleypy_native/tests/installed_wheel_smoke.py"

scorer_build="${run_root}/private-cnot-scorer"
"$python_bin" "$source_dir/integrations/cnot_scorer/build.py" --source "$upstream" --build-dir "$scorer_build"
scorer_library="${scorer_build}/target/release/libcnot103_native.so"
test -s "$scorer_library"
scorer_sha256="$(sha256sum "$scorer_library" | awk '{print $1}')"
python3 - "$result_dir/receipt.json" "$scorer_sha256" <<'PY'
import json, pathlib, sys
receipt_path = pathlib.Path(sys.argv[1])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
receipt["scorer_library_sha256"] = sys.argv[2]
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
domain_build="${run_root}/cnot-owned-domain-build"
cmake -S "$source_dir" -B "$domain_build" -DCMAKE_BUILD_TYPE=Release -DBEAM_CNOT_OWNED_DOMAIN_ONLY=ON -DBEAM_CUDA_ARCHITECTURES=86
cmake --build "$domain_build" --target beam_cnot_owned_domain --parallel 2
domain_library="${domain_build}/libbeam_cnot_owned_domain.so"
test -s "$domain_library"
export BEAM_CNOT_OWNED_LIBRARY="$domain_library"
export MGBS_CNOT_OWNED_LIBRARY="$domain_library"
export MGBS_TQP_LIBRARY="$scorer_library"
export MGBS_TQP_MANIFEST="$manifest"
export MGBS_CNOT_EXPECTED_TQP_LIBRARY_SHA256="$scorer_sha256"
export MGBS_CNOT_EXPECTED_SEARCH_MANIFEST_SHA256="$manifest_sha"
export CNOT_RUST_EVIDENCE="${result_dir}/ranks"
export TORCH_EXTENSIONS_DIR="${run_root}/torch-extensions"
export RUST_BACKTRACE=1
mkdir -p "$CNOT_RUST_EVIDENCE"
MGBS_TQP_LIBRARY="$scorer_library" MGBS_TQP_MANIFEST="$manifest" "$wheel_python" -m unittest discover -s "$source_dir/integrations/cayleypy_native/tests" -p test_cnot_scorer.py -v
"$wheel_python" "$source_dir/tools/verify_permutation_cuda.py" --source-dir "$source_dir" --cache-dir "${run_root}/permutation-cache" --cutlass-dir "$cutlass_dir" --output "${result_dir}/permutation-1-and-2-gpu.json" --source-sha "$source_sha" --wheel-sha256 "$wheel_sha256" --acceptance-run-id "$acceptance_run_id"

expected_checkpoint_sha="$expected_checkpoint"
# Paired topology gate: same GL4 root/head/checkpoint/mode/B/K/precision.
run_trained_case() {
  local case_id="$1" visible_devices="$2" nproc="$3" gpu_uuid_list="$4"
  shift 4
  local case_log="${result_dir}/${case_id}.log"
  if CUDA_VISIBLE_DEVICES="$visible_devices" \
    MGBS_CNOT_STRICT_BACKEND=1 \
    MGBS_CNOT_EXPECTED_CHECKPOINT_SHA256="$expected_checkpoint_sha" \
    MGBS_CNOT_EXPECTED_SEARCH_MANIFEST_SHA256="$manifest_sha" \
    MGBS_CNOT_ACCEPTANCE_RUN_ID="$acceptance_run_id" \
    MGBS_CNOT_ACCEPTANCE_CASE="$case_id" \
    MGBS_CNOT_ACCEPTANCE_SOURCE_SHA="$source_sha" \
    MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256="$wheel_sha256" \
    MGBS_CNOT_ACCEPTANCE_GPU_UUIDS="$gpu_uuid_list" \
    "$wheel_python" -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
    "$source_dir/tools/verify_cnot.py" "$@" >"$case_log" 2>&1; then
    cat "$case_log"
  else
    local case_exit=$?
    cat "$case_log"
    return "$case_exit"
  fi
}
run_panel_case() {
  local case_id="$1" visible_devices="$2" nproc="$3" gpu_uuid_list="$4"
  local case_log="${panel_result_dir}/${case_id}.log"
  local -a panel_args=()
  mapfile -d '' -t panel_args < <(
    mgbs_vast_cnot_panel_cli_args "${handoff_run_root}" \
      "${panel_result_dir}/${case_id}.json" "${scorer_library}" \
      "${handoff_preflight}" "${handoff_preflight_sha}"
  )
  if CUDA_VISIBLE_DEVICES="${visible_devices}" \
    MGBS_TQP_MANIFEST="${panel_manifest}" \
    MGBS_CNOT_EXPECTED_CHECKPOINT_SHA256="${panel_checkpoint_sha}" \
    MGBS_CNOT_EXPECTED_SEARCH_MANIFEST_SHA256="${panel_manifest_sha}" \
    MGBS_CNOT_ACCEPTANCE_RUN_ID="${acceptance_run_id}" \
    MGBS_CNOT_ACCEPTANCE_CASE="${case_id}" \
    MGBS_CNOT_ACCEPTANCE_SOURCE_SHA="${source_sha}" \
    MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256="${wheel_sha256}" \
    MGBS_CNOT_ACCEPTANCE_GPU_UUIDS="${gpu_uuid_list}" \
    "${wheel_python}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}" \
    "${source_dir}/tools/verify_cnot.py" "${panel_args[@]}" \
    >"${case_log}" 2>&1; then
    cat "$case_log"
  else
    local case_exit=$?
    cat "$case_log"
    return "$case_exit"
  fi
}
run_trained_case cnot-trained-qhat-1-gpu "${physical_gpu_uuids[0]}" 1 "${physical_gpu_uuids[0]}" \
  --case trained --backend nccl --mode qhat --engine rust --require-rust-scorer \
  --library "$scorer_library" --manifest "$manifest" \
  --output "${result_dir}/cnot-trained-qhat-1-gpu.json"
run_trained_case cnot-trained-deep-paired-1-gpu "${physical_gpu_uuids[0]}" 1 "${physical_gpu_uuids[0]}" \
  --case trained-deep --backend nccl --mode proposal --width 2 --engine rust --require-rust-scorer \
  --library "$scorer_library" --manifest "$manifest" \
  --output "${result_dir}/cnot-trained-deep-paired-1-gpu.json"
run_trained_case cnot-trained-deep-paired-2-gpu "${physical_gpu_uuids[0]},${physical_gpu_uuids[1]}" 2 "${physical_gpu_uuids[0]},${physical_gpu_uuids[1]}" \
  --case trained-deep --backend nccl --mode proposal --width 2 --engine rust --require-rust-scorer \
  --library "$scorer_library" --manifest "$manifest" \
  --output "${result_dir}/cnot-trained-deep-paired-2-gpu.json"
"$wheel_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$source_dir/integrations/cayleypy_native/tests/cnot_stream5_control_nccl.py"
"$wheel_python" -m torch.distributed.run --standalone --nproc_per_node=2 "$source_dir/integrations/cayleypy_native/tests/cnot_device_unseen_distributed.py"
run_panel_case cnot-panel-test-qhat-1-gpu "${physical_gpu_uuids[0]}" 1 "${physical_gpu_uuids[0]}"
run_panel_case cnot-panel-test-qhat-2-gpu "${physical_gpu_uuids[0]},${physical_gpu_uuids[1]}" 2 "${physical_gpu_uuids[0]},${physical_gpu_uuids[1]}"
mkdir -p "${result_dir}/backend-events" "${panel_result_dir}/backend-events"
for case_id in cnot-trained-qhat-1-gpu cnot-trained-deep-paired-1-gpu cnot-trained-deep-paired-2-gpu; do
  "$python_bin" "$source_dir/tools/cnot_acceptance_events.py" validate \
    --trace "${result_dir}/${case_id}.log" --receipts "$CNOT_RUST_EVIDENCE" \
    --report "${result_dir}/${case_id}.json" --expected-source-sha "$source_sha" \
    --expected-wheel-sha256 "$wheel_sha256" --expected-checkpoint-sha256 "$expected_checkpoint" \
    --expected-manifest-sha256 "$manifest_sha" --expected-scorer-library-sha256 "$scorer_sha256" \
    > "${result_dir}/backend-events/${case_id}.json"
done
for case_id in cnot-panel-test-qhat-1-gpu cnot-panel-test-qhat-2-gpu; do
  "$python_bin" "$source_dir/tools/cnot_acceptance_events.py" validate \
    --trace "${panel_result_dir}/${case_id}.log" --receipts "$CNOT_RUST_EVIDENCE" \
    --report "${panel_result_dir}/${case_id}.json" \
    --expected-source-sha "$source_sha" --expected-wheel-sha256 "$wheel_sha256" \
    --expected-checkpoint-sha256 "$panel_checkpoint_sha" \
    --expected-manifest-sha256 "$panel_manifest_sha" \
    --expected-scorer-library-sha256 "$scorer_sha256" \
    > "${panel_result_dir}/backend-events/${case_id}.json"
done
"$python_bin" "$source_dir/tools/verify_cnot_gpu_pair.py" \
  --result-dir "$result_dir" --source-sha "$source_sha" --instance-id "$instance_id" \
  --manifest-sha256 "$manifest_sha" --checkpoint-sha256 "$expected_checkpoint" \
  --scorer-library-sha256 "$scorer_sha256" --wheel-sha256 "$wheel_sha256" \
  --run-id "$acceptance_run_id" \
  --panel-result-dir "$panel_result_dir" --handoff-preflight "$handoff_preflight"
