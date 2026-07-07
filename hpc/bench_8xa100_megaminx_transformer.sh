#!/bin/bash
#SBATCH --job-name=megaminx-s1-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${SCRIPT_DIR}/mephi_8xa100_common.sh"

beam_setup_paths

PUZZLE_ID="${PUZZLE_ID:-991}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1000000}"
STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-262144}"
STREAM4_TRIGGER_MULT="${STREAM4_TRIGGER_MULT:-4}"
STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-$((STREAM4_BATCH_CANDIDATES * STREAM4_TRIGGER_MULT))}"
BEAM_STREAM4_ACTIVE_SORT_SLOTS="${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}"
BEAM_SHARD_BUFFER_COUNT="${BEAM_SHARD_BUFFER_COUNT:-2}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-98304}"
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-2000000}"
BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM="${BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM:-1200000}"
BEAM_STREAM1_TRANSFORMER_MICRO="${BEAM_STREAM1_TRANSFORMER_MICRO:-512}"
BEAM_STREAM1_TRANSFORMER_BLOCK51="${BEAM_STREAM1_TRANSFORMER_BLOCK51:-1}"
BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY="${BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY:-1}"

SMOKE_BEAM_WIDTH="${SMOKE_BEAM_WIDTH:-64000000}"
SMOKE_DEPTH_LIMIT="${SMOKE_DEPTH_LIMIT:-12}"
SMOKE_SHARD_COUNT="${SMOKE_SHARD_COUNT:-4}"
SMOKE_B_MICRO="${SMOKE_B_MICRO:-8192}"
SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-8}"
SMOKE_SHARD_CAPACITY_SCALE_PPM="${SMOKE_SHARD_CAPACITY_SCALE_PPM:-1250000}"

TARGET_BEAM_WIDTH="${TARGET_BEAM_WIDTH:-900000000}"
TARGET_DEPTH_LIMIT="${TARGET_DEPTH_LIMIT:-8}"
TARGET_BACKEND_SWEEP="${TARGET_BACKEND_SWEEP:-native_cuda_graph libtorch_eager}"
TARGET_B_MICRO_SWEEP="${TARGET_B_MICRO_SWEEP:-8192 12288}"
TARGET_CONCURRENCY_SWEEP="${TARGET_CONCURRENCY_SWEEP:-8}"
TARGET_RING_SLOTS_SWEEP="${TARGET_RING_SLOTS_SWEEP:-8}"
TARGET_SHARD_COUNT_SWEEP="${TARGET_SHARD_COUNT_SWEEP:-32}"
TARGET_FINAL_CHUNK_SWEEP="${TARGET_FINAL_CHUNK_SWEEP:-98304}"
TARGET_SHARD_CAPACITY_SCALE_PPM="${TARGET_SHARD_CAPACITY_SCALE_PPM:-1000000}"

ISOLATED_B_MICRO_SWEEP="${ISOLATED_B_MICRO_SWEEP:-${ISOLATED_BATCH_LIST:-${ISOLATED_NATIVE_B_MICRO_SWEEP:-512 1024 2048 4096 8192 12288 16384}}}"
ISOLATED_CONCURRENCY_SWEEP="${ISOLATED_CONCURRENCY_SWEEP:-${ISOLATED_NATIVE_CONCURRENCY_SWEEP:-1 2 4 8}}"
ISOLATED_BATCH_LIST="${ISOLATED_BATCH_LIST:-${ISOLATED_B_MICRO_SWEEP}}"
ISOLATED_BATCHES="${ISOLATED_BATCHES:-${ISOLATED_BATCH_LIST// /,}}"
ISOLATED_WARMUP="${ISOLATED_WARMUP:-8}"
ISOLATED_ITERS="${ISOLATED_ITERS:-25}"
ISOLATED_PASSES="${ISOLATED_PASSES:-2}"
RUN_ISOLATED_STREAM1="${RUN_ISOLATED_STREAM1:-1}"
RUN_FULL_SMOKE="${RUN_FULL_SMOKE:-0}"
SMOKE_BACKEND_SWEEP="${SMOKE_BACKEND_SWEEP:-native_cuda_graph libtorch_eager}"
RUN_TARGET_SWEEP="${RUN_TARGET_SWEEP:-0}"
RUN_SELECTED_900M_AFTER_STREAM1="${RUN_SELECTED_900M_AFTER_STREAM1:-0}"
RUN_PIPELINE_SMOKE="${RUN_PIPELINE_SMOKE:-0}"
PIPELINE_SMOKE_MODES="${PIPELINE_SMOKE_MODES:-stream12 stream123}"
PIPELINE_GRAPH_WINDOW_SWEEP="${PIPELINE_GRAPH_WINDOW_SWEEP:-16 32 64}"
PIPELINE_B_MICRO_SWEEP="${PIPELINE_B_MICRO_SWEEP:-${TARGET_B_MICRO_SWEEP%% *}}"
PIPELINE_CONCURRENCY_SWEEP="${PIPELINE_CONCURRENCY_SWEEP:-${TARGET_CONCURRENCY_SWEEP%% *}}"
PIPELINE_SMOKE_RINGS="${PIPELINE_SMOKE_RINGS:-32}"
PIPELINE_SHARD_COUNT="${PIPELINE_SHARD_COUNT:-32}"
DEPTH_AVG_MIN="${DEPTH_AVG_MIN:-3}"

MODEL_DIR="${MODEL_DIR:-${JOB_DIR}/models/megaminx_vlad_transformer}"
REPO_WEIGHT_DIR="${REPO_DIR}/weights/megaminx_vlad_transformer_fp16"
if [ -z "${BEAM_WEIGHT_DIR:-}" ]; then
  if [ -f "${REPO_WEIGHT_DIR}/manifest.json" ]; then
    BEAM_WEIGHT_DIR="${REPO_WEIGHT_DIR}"
  else
    BEAM_WEIGHT_DIR="${JOB_DIR}/stream1_transformer_weights_fp16"
  fi
fi

find_transformer_checkpoint() {
  if [ -n "${BEAM_TRANSFORMER_PTH:-}" ]; then
    printf '%s\n' "${BEAM_TRANSFORMER_PTH}"
    return 0
  fi
  if [ -d "${MODEL_DIR}" ]; then
    find "${MODEL_DIR}" -maxdepth 5 -type f \( -name '*.pth' -o -name '*.pt' \) | sort | head -n 1
  fi
}

prepare_transformer_weights() {
  if [ -f "${BEAM_WEIGHT_DIR}/manifest.json" ]; then
    echo "using_weight_dir=${BEAM_WEIGHT_DIR}"
    return 0
  fi
  local checkpoint
  checkpoint="$(find_transformer_checkpoint || true)"
  if [ -z "${checkpoint}" ] || [ ! -f "${checkpoint}" ]; then
    echo "missing_stream1_weights=${BEAM_WEIGHT_DIR}/manifest.json"
    echo "missing_transformer_checkpoint=${MODEL_DIR}/*.pth"
    return 2
  fi
  echo "export_transformer_checkpoint=${checkpoint}"
  rm -rf -- "${BEAM_WEIGHT_DIR}.tmp"
  mkdir -p "${BEAM_WEIGHT_DIR}.tmp"
  "${NINJA_VENV_DIR}/bin/python" "${REPO_DIR}/tools/export_stream1.py" \
    --weights "${checkpoint}" \
    --out "${BEAM_WEIGHT_DIR}.tmp" \
    --dtype fp16 \
    --format piece-transformer
  rm -rf -- "${BEAM_WEIGHT_DIR}"
  mv "${BEAM_WEIGHT_DIR}.tmp" "${BEAM_WEIGHT_DIR}"
  echo "exported_weight_dir=${BEAM_WEIGHT_DIR}"
}

cleanup_done=0
cleanup() {
  local rc=$?
  if [ "${cleanup_done}" -ne 0 ]; then
    exit "${rc}"
  fi
  cleanup_done=1
  echo "cleanup_start rc=${rc} at $(date -Is)"
  beam_safe_clean_child "${BUILD_DIR}" "build"
  beam_safe_clean_child "${HISTORY_DIR}" "history"
  echo "cleanup_done rc=${rc} at $(date -Is)"
  exit "${rc}"
}
trap cleanup EXIT TERM INT

beam_preflight

export BEAM_ENABLE_LIBTORCH_STREAM1=ON
if [ -z "${CMAKE_PREFIX_PATH:-}" ]; then
  CMAKE_PREFIX_PATH="$("${NINJA_VENV_DIR}/bin/python" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"
  export CMAKE_PREFIX_PATH
fi
TORCH_LIB_DIR="$("${NINJA_VENV_DIR}/bin/python" - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"

prepare_transformer_weights

beam_configure_build stream_benchmark
beam_configure_build stream_pipeline_benchmark
beam_configure_build stream1_transformer_libtorch_benchmark
beam_configure_build production_runner
beam_configure_build production_runner_libtorch_stream1
beam_export_common_runtime
export BEAM_WEIGHT_DIR
export BEAM_PREDICT_STATS_VERBOSE="${BEAM_PREDICT_STATS_VERBOSE:-1}"

SUMMARY="${TUNING_DIR}/megaminx_transformer_bench_${SLURM_JOB_ID:-manual}.tsv"
BEST_ENV="${LOG_DIR}/best_megaminx_transformer.env"
BEST_STREAM1_ENV="${LOG_DIR}/best_megaminx_transformer_stream1.env"
ISOLATED_SUMMARY="${TUNING_DIR}/megaminx_transformer_stream1_isolated_${SLURM_JOB_ID:-manual}.tsv"
PIPELINE_SUMMARY="${TUNING_DIR}/megaminx_transformer_pipeline_smoke_${SLURM_JOB_ID:-manual}.tsv"
mkdir -p "${TUNING_DIR}"
echo -e "stage\tbackend\tmode\tstatus\tbest_candidates_per_sec\tbest_batch\tb_micro\tconcurrency\tbatch_csv\tlog" > "${ISOLATED_SUMMARY}"
echo -e "mode\twindow\tb_micro\tconcurrency\tring_slots\tstream3_batch\tgraph_window_jobs\tphysical_jobs\tcandidates_per_sec\tdepth_like_ms\tstatus\tlog" > "${PIPELINE_SUMMARY}"
echo -e "stage\tbackend\tstatus\tavg_depth_sec\tlast_depth\tmax_gpu_mem_mib\tmax_gpu_util_pct\tbeam_width\tdepth_limit\tshard_count\tb_micro\tconcurrency\tring_slots\tstream3_batch\tstream4_batch\tstream4_trigger\tfinal_chunk\tfinal_exchange_scale_ppm\tshard_capacity_scale_ppm\tshard_capacity\tlogical_shard\trunner\tlog" > "${SUMMARY}"

parse_best_cps_batch() {
  local log="$1"
  awk '
    /candidates_per_sec=|candidates_per_s=/ {
      cps=""; batch="";
      for (i=1; i<=NF; ++i) {
        if ($i ~ /^candidates_per_sec=/ || $i ~ /^candidates_per_s=/) { split($i,a,"="); cps=a[2]; }
        if ($i ~ /^batch=/) { split($i,a,"="); batch=a[2]; }
        if ($i ~ /^b_micro=/) { split($i,a,"="); batch=a[2]; }
      }
      if (cps != "" && cps+0 > best) { best=cps+0; best_batch=batch; }
    }
    END {
      if (best > 0) printf "%.6f\t%s", best, (best_batch != "" ? best_batch : "NA");
      else printf "NA\tNA";
    }
  ' "${log}"
}

parse_avg_depth_sec() {
  local log="$1"
  awk -v min_depth="${DEPTH_AVG_MIN}" '
    /depth_done=/ && /depth_sec=/ {
      depth=""; sec="";
      for (i=1; i<=NF; ++i) {
        if ($i ~ /^depth_done=/) { split($i,a,"="); depth=a[2]; }
        if ($i ~ /^depth_sec=/) { split($i,a,"="); sec=a[2]; }
      }
      if (depth != "" && sec != "" && depth+0 >= min_depth) { sum += sec; n += 1; }
      if (depth != "" && depth+0 > last) { last = depth+0; }
    }
    END { if (n > 0) printf "%.6f\t%d", sum / n, last; else printf "NA\t%d", last; }
  ' "${log}"
}

parse_gpu_maxima() {
  local gpu_log="$1"
  if [ ! -f "${gpu_log}" ]; then
    printf "NA\tNA"
    return 0
  fi
  awk -F',' '
    NF >= 8 {
      mem=$5; util=$7;
      gsub(/^[ ]+|[ ]+$/, "", mem);
      gsub(/^[ ]+|[ ]+$/, "", util);
      if (mem+0 > max_mem) { max_mem=mem+0; }
      if (util+0 > max_util) { max_util=util+0; }
    }
    END {
      if (max_mem > 0 || max_util > 0) printf "%d\t%d", max_mem, max_util;
      else printf "NA\tNA";
    }
  ' "${gpu_log}"
}

run_isolated_backend() {
  local backend="$1"
  local mode="$2"
  local b_micro="${3:-NA}"
  local concurrency="${4:-NA}"
  local batch_csv="${5:-}"
  if [ -z "${batch_csv}" ]; then
    if [ "${b_micro}" != "NA" ]; then
      batch_csv="${b_micro}"
    else
      batch_csv="${ISOLATED_BATCHES}"
    fi
  fi
  local tag="isolated_${backend}_${mode}_b${b_micro}_c${concurrency}"
  local log="${TUNING_DIR}/${tag}.log"
  local status="OK"
  local rc=0
  local cmd=("${NINJA_VENV_DIR}/bin/python" "${REPO_DIR}/tools/stream1_transformer_backends.py" --backend "${backend}" --mode "${mode}" --weight-dir "${BEAM_WEIGHT_DIR}" --build-dir "${BUILD_DIR}" --batches "${batch_csv}" --warmup "${ISOLATED_WARMUP}" --iters "${ISOLATED_ITERS}" --passes "${ISOLATED_PASSES}" --device cuda:0)
  if [ "${concurrency}" != "NA" ]; then
    cmd+=(--concurrency "${concurrency}")
  fi
  if [ "${backend}" = "libtorch" ]; then
    cmd+=(--csv "${TUNING_DIR}/${tag}.csv")
  else
    cmd+=(--report "${TUNING_DIR}/${tag}.md")
  fi
  if [ "${backend}" = "native_cutlass" ]; then
    cmd+=(--b-micro "${b_micro}" --synthetic-states)
  fi
  echo "isolated_start backend=${backend} mode=${mode} b_micro=${b_micro} concurrency=${concurrency} batches=${batch_csv} log=${log}"
  set +e
  "${cmd[@]}" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -ne 0 ]; then
    status="FAIL_${rc}"
  fi
  local cps_batch
  cps_batch="$(parse_best_cps_batch "${log}")"
  echo -e "isolated\t${backend}\t${mode}\t${status}\t${cps_batch}\t${b_micro}\t${concurrency}\t${batch_csv}\t${log}" >> "${ISOLATED_SUMMARY}"
}
parse_pipeline_smoke_result() {
  local log="$1"
  awk '
    /stream_pipeline_benchmark/ {
      for (i=1; i<=NF; ++i) {
        split($i, a, "=");
        if (a[1] == "ring_slots") ring_slots=a[2];
        if (a[1] == "stream3_batch") stream3_batch=a[2];
        if (a[1] == "graph_window_jobs") graph_window_jobs=a[2];
        if (a[1] == "physical_jobs") physical_jobs=a[2];
        if (a[1] == "candidates_per_sec") cps=a[2];
        if (a[1] == "depth_like_ms") ms=a[2];
      }
    }
    END {
      if (cps == "") cps="NA";
      if (ms == "") ms="NA";
      if (ring_slots == "") ring_slots="NA";
      if (stream3_batch == "") stream3_batch="NA";
      if (graph_window_jobs == "") graph_window_jobs="NA";
      if (physical_jobs == "") physical_jobs="NA";
      printf "%s\t%s\t%s\t%s\t%s\t%s", ring_slots, stream3_batch, graph_window_jobs, physical_jobs, cps, ms;
    }
  ' "${log}"
}

run_pipeline_smoke_config() {
  local mode="$1"
  local window="$2"
  local b_micro="$3"
  local concurrency="$4"
  local tag="pipeline_${mode}_w${window}_b${b_micro}_c${concurrency}"
  local log="${TUNING_DIR}/${tag}.log"
  local status="OK"
  local rc=0
  local stream3_batch="${PIPELINE_STREAM3_BATCH_CANDIDATES:-$((b_micro * 24 * BEAM_STREAM3_RING_SLOTS))}"

  echo "pipeline_smoke_start mode=${mode} window=${window} b_micro=${b_micro} transformer_micro=${BEAM_STREAM1_TRANSFORMER_MICRO} concurrency=${concurrency} stream3_batch=${stream3_batch} log=${log}"
  set +e
  BEAM_PIPELINE_BENCH_MODE="${mode}" \
  BEAM_RING_GRAPH_EXECS_PER_LANE="${window}" \
  BEAM_B_MICRO="${b_micro}" \
  BEAM_STREAM1_TRANSFORMER_MICRO="${BEAM_STREAM1_TRANSFORMER_MICRO}" \
  BEAM_STREAM1_TRANSFORMER_BLOCK51="${BEAM_STREAM1_TRANSFORMER_BLOCK51}" \
  BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY="${BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY}" \
  BEAM_STREAM1_CONCURRENCY="${concurrency}" \
  BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS}" \
  BEAM_STREAM3_BATCH_CANDIDATES="${stream3_batch}" \
  BEAM_PIPELINE_SMOKE_RINGS="${PIPELINE_SMOKE_RINGS}" \
  BEAM_SHARD_COUNT="${PIPELINE_SHARD_COUNT}" \
  BEAM_WEIGHT_DIR="${BEAM_WEIGHT_DIR}" \
  "${BUILD_DIR}/stream_pipeline_benchmark" "${PUZZLE_ID}" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -ne 0 ]; then
    status="FAIL_${rc}"
  fi
  local parsed
  parsed="$(parse_pipeline_smoke_result "${log}")"
  echo -e "${mode}\t${window}\t${b_micro}\t${concurrency}\t${parsed}\t${status}\t${log}" >> "${PIPELINE_SUMMARY}"
}
set_backend_runner() {
  local backend="$1"
  case "${backend}" in
    libtorch_eager)
      export BEAM_STREAM1_EXECUTOR=libtorch_eager
      export BEAM_PRODUCTION_RUNNER_PATH="${BUILD_DIR}/production_runner_libtorch_stream1"
      ;;
    native_cuda_graph|cuda_graph)
      export BEAM_STREAM1_EXECUTOR=native_cuda_graph
      export BEAM_PRODUCTION_RUNNER_PATH="${BUILD_DIR}/production_runner"
      ;;
    *)
      echo "unknown_full_backend=${backend}"
      return 2
      ;;
  esac
}

run_full_config() {
  local stage="$1"
  local backend="$2"
  local beam_width="$3"
  local depth_limit="$4"
  local shard_count="$5"
  local b_micro="$6"
  local concurrency="$7"
  local ring_slots="$8"
  local final_chunk="$9"
  local capacity_scale="${10}"

  PUZZLE_ID="${PUZZLE_ID}"
  BEAM_WIDTH="${beam_width}"
  DEPTH_LIMIT="${depth_limit}"
  SHARD_COUNT="${shard_count}"
  BEAM_B_MICRO="${b_micro}"
  BEAM_STREAM1_CONCURRENCY="${concurrency}"
  BEAM_STREAM3_RING_SLOTS="${ring_slots}"
  BEAM_STREAM3_BATCH_CANDIDATES="${FULL_STREAM3_BATCH_CANDIDATES:-$((b_micro * 24 * ring_slots))}"
  BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${final_chunk}"
  SHARD_CAPACITY_SCALE_PPM="${capacity_scale}"
  STREAM4_TRIGGER_CANDIDATES=$((STREAM4_BATCH_CANDIDATES * STREAM4_TRIGGER_MULT))

  set_backend_runner "${backend}"
  beam_derive_shard_capacity

  local tag="${stage}_${backend}_bw${BEAM_WIDTH}_d${DEPTH_LIMIT}_sh${SHARD_COUNT}_b${BEAM_B_MICRO}_c${BEAM_STREAM1_CONCURRENCY}_r${BEAM_STREAM3_RING_SLOTS}_s3${STREAM3_BATCH_CANDIDATES}_fc${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}"
  local log="${TUNING_DIR}/${tag}.log"
  local gpu_log="${TUNING_DIR}/nvidia_smi_${tag}.log"
  local status="OK"
  local avg_last="NA	0"
  local gpu_max="NA	NA"

  if ! beam_validate_manual_config > "${TUNING_DIR}/${tag}.preflight" 2>&1; then
    status="INVALID"
    cat "${TUNING_DIR}/${tag}.preflight" | tee "${log}"
    echo -e "${stage}\t${backend}\t${status}\tNA\t0\tNA\tNA\t${BEAM_WIDTH}\t${DEPTH_LIMIT}\t${SHARD_COUNT}\t${BEAM_B_MICRO}\t${BEAM_STREAM1_CONCURRENCY}\t${BEAM_STREAM3_RING_SLOTS}\t${STREAM3_BATCH_CANDIDATES}\t${STREAM4_BATCH_CANDIDATES}\t${STREAM4_TRIGGER_CANDIDATES}\t${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}\t${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}\t${SHARD_CAPACITY_SCALE_PPM}\t${SHARD_CAPACITY_CANDIDATES}\t${LOGICAL_SHARD_SIZE}\t${BEAM_PRODUCTION_RUNNER_PATH}\t${log}" >> "${SUMMARY}"
    return 0
  fi

  beam_safe_clear_history_contents
  export BEAM_B_MICRO
  export BEAM_STREAM1_TRANSFORMER_MICRO
  export BEAM_STREAM1_TRANSFORMER_BLOCK51
  export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY
  export BEAM_STREAM1_CONCURRENCY
  export BEAM_STREAM3_RING_SLOTS
  export BEAM_SHARD_BUFFER_COUNT
  export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM
  export BEAM_PREDICT_STATS_PATH="${TUNING_DIR}/predict_stats_${tag}.jsonl"
  beam_export_manual_config
  beam_prepare_nccl_file "${tag}"

  echo "full_start stage=${stage} backend=${backend} beam_width=${BEAM_WIDTH} depth_limit=${DEPTH_LIMIT} b_micro=${BEAM_B_MICRO} transformer_micro=${BEAM_STREAM1_TRANSFORMER_MICRO} block51=${BEAM_STREAM1_TRANSFORMER_BLOCK51} final_cls_only=${BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY} concurrency=${BEAM_STREAM1_CONCURRENCY} stream3_batch=${STREAM3_BATCH_CANDIDATES} log=${log}"
  local rc=0
  set +e
  beam_torchrun_production "${tag}" "${log}" || rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    status="FAIL_${rc}"
  fi
  avg_last="$(parse_avg_depth_sec "${log}")"
  gpu_max="$(parse_gpu_maxima "${gpu_log}")"
  echo -e "${stage}\t${backend}\t${status}\t${avg_last}\t${gpu_max}\t${BEAM_WIDTH}\t${DEPTH_LIMIT}\t${SHARD_COUNT}\t${BEAM_B_MICRO}\t${BEAM_STREAM1_CONCURRENCY}\t${BEAM_STREAM3_RING_SLOTS}\t${STREAM3_BATCH_CANDIDATES}\t${STREAM4_BATCH_CANDIDATES}\t${STREAM4_TRIGGER_CANDIDATES}\t${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}\t${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}\t${SHARD_CAPACITY_SCALE_PPM}\t${SHARD_CAPACITY_CANDIDATES}\t${LOGICAL_SHARD_SIZE}\t${BEAM_PRODUCTION_RUNNER_PATH}\t${log}" >> "${SUMMARY}"
  return 0
}

write_best_stream1_env() {
  awk -F'\t' -v default_concurrency="${TARGET_CONCURRENCY_SWEEP%% *}" -v target_bmicro="${TARGET_B_MICRO_SWEEP%% *}" '
    NR == 1 { next }
    $1 == "isolated" && $4 == "OK" && $5 != "NA" {
      runner_backend=""; b=""; c="";
      if ($2 == "native_cutlass" && $3 == "graph") {
        runner_backend="native_cuda_graph"; b=$7; c=$8;
      } else if ($2 == "libtorch" && $3 == "eager") {
        runner_backend="libtorch_eager"; b=$7; c=$8;
      } else {
        next;
      }
      if (b == "" || b == "NA") { next; }
      if (c == "" || c == "NA") { c=default_concurrency; }
      if (best == "" || $5 + 0 > best + 0) {
        best=$5; backend=runner_backend; bench_backend=$2; mode=$3; batch=$6; bmicro=b; concurrency=c; source_log=$10;
      }
    }
    END {
      if (best == "") { exit 1 }
      print "export MEGAMINX_STREAM1_BACKEND=" backend;
      print "export BEAM_B_MICRO=" target_bmicro;
      print "export BEAM_STREAM1_TRANSFORMER_MICRO=" bmicro;
      print "export BEAM_STREAM1_TRANSFORMER_BLOCK51=1";
      print "export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1";
      print "export BEAM_STREAM1_CONCURRENCY=" concurrency;
      print "export BEST_STREAM1_BACKEND_BENCH=" bench_backend;
      print "export BEST_STREAM1_MODE=" mode;
      print "export BEST_STREAM1_BATCH=" batch;
      print "export BEST_STREAM1_ISOLATED_B_MICRO=" bmicro;
      print "export BEST_STREAM1_CANDIDATES_PER_SEC=" best;
      print "export BEST_STREAM1_SOURCE_LOG=" source_log;
    }
  ' "${ISOLATED_SUMMARY}" > "${BEST_STREAM1_ENV}.tmp" || return 1
  mv "${BEST_STREAM1_ENV}.tmp" "${BEST_STREAM1_ENV}"
}

write_best_env() {
  awk -F'\t' -v transformer_micro="${BEAM_STREAM1_TRANSFORMER_MICRO}" '
    NR == 1 { next }
    $1 == "target" && $3 == "OK" && $4 != "NA" {
      if (best == "" || $4 + 0 < best + 0) {
        best=$4; backend=$2; beam=$8; depth=$9; shard=$10; b=$11; c=$12; ring=$13;
        s3=$14; s4=$15; trig=$16; fc=$17; fs=$18; scale=$19; cap=$20; logical=$21; runner=$22; source_log=$23;
      }
    }
    END {
      if (best == "") { exit 1 }
      print "export MEGAMINX_STREAM1_BACKEND=" backend;
      print "export BEAM_WIDTH=" beam;
      print "export SHARD_COUNT=" shard;
      print "export BEAM_B_MICRO=" b;
      print "export BEAM_STREAM1_TRANSFORMER_MICRO=" transformer_micro;
      print "export BEAM_STREAM1_TRANSFORMER_BLOCK51=1";
      print "export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1";
      print "export BEAM_STREAM1_CONCURRENCY=" c;
      print "export BEAM_STREAM3_RING_SLOTS=" ring;
      print "export BEAM_STREAM3_BATCH_CANDIDATES=" s3;
      print "export STREAM4_BATCH_CANDIDATES=" s4;
      print "export STREAM4_TRIGGER_CANDIDATES=" trig;
      print "export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=" fc;
      print "export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=" fs;
      print "export SHARD_CAPACITY_SCALE_PPM=" scale;
      print "export BEST_MEGAMINX_AVG_DEPTH_SEC=" best;
      print "export BEST_MEGAMINX_SOURCE_LOG=" source_log;
    }
  ' "${SUMMARY}" > "${BEST_ENV}.tmp" || return 1
  mv "${BEST_ENV}.tmp" "${BEST_ENV}"
}

echo "benchmark_started_at=$(date -Is)"
echo "summary=${SUMMARY}"
echo "isolated_summary=${ISOLATED_SUMMARY}"
echo "pipeline_smoke_summary=${PIPELINE_SUMMARY}"
echo "weight_dir=${BEAM_WEIGHT_DIR}"
echo "torch_cmake_prefix=${CMAKE_PREFIX_PATH}"
echo "torch_lib_dir=${TORCH_LIB_DIR}"

if [ "${RUN_ISOLATED_STREAM1}" = "1" ]; then
  for b_micro in ${ISOLATED_B_MICRO_SWEEP}; do
    for concurrency in ${ISOLATED_CONCURRENCY_SWEEP}; do
      run_isolated_backend pytorch eager "${b_micro}" "${concurrency}"
      run_isolated_backend libtorch eager "${b_micro}" "${concurrency}"
      run_isolated_backend libtorch cuda_graph "${b_micro}" "${concurrency}"
      run_isolated_backend native_cutlass graph "${b_micro}" "${concurrency}"
    done
  done
  if write_best_stream1_env; then
    echo "best_megaminx_transformer_stream1_env=${BEST_STREAM1_ENV}"
    cat "${BEST_STREAM1_ENV}"
  else
    echo "best_megaminx_transformer_stream1_env=none"
  fi
fi

if [ "${RUN_PIPELINE_SMOKE}" = "1" ]; then
  for mode in ${PIPELINE_SMOKE_MODES}; do
    for window in ${PIPELINE_GRAPH_WINDOW_SWEEP}; do
      for b_micro in ${PIPELINE_B_MICRO_SWEEP}; do
        for concurrency in ${PIPELINE_CONCURRENCY_SWEEP}; do
          run_pipeline_smoke_config "${mode}" "${window}" "${b_micro}" "${concurrency}"
        done
      done
    done
  done
fi

if [ "${RUN_SELECTED_900M_AFTER_STREAM1}" = "1" ]; then
  if [ ! -f "${BEST_STREAM1_ENV}" ]; then
    echo "missing_best_stream1_env=${BEST_STREAM1_ENV}"
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${BEST_STREAM1_ENV}"
  echo "selected_900m_from_stream1_env=${BEST_STREAM1_ENV}"
  echo "selected_900m_backend=${MEGAMINX_STREAM1_BACKEND} b_micro=${BEAM_B_MICRO} transformer_micro=${BEAM_STREAM1_TRANSFORMER_MICRO} concurrency=${BEAM_STREAM1_CONCURRENCY}"
  run_full_config target "${MEGAMINX_STREAM1_BACKEND}" "${TARGET_BEAM_WIDTH}" "${TARGET_DEPTH_LIMIT}" "${TARGET_SHARD_COUNT_SWEEP%% *}" "${BEAM_B_MICRO}" "${BEAM_STREAM1_CONCURRENCY}" "${TARGET_RING_SLOTS_SWEEP%% *}" "${TARGET_FINAL_CHUNK_SWEEP%% *}" "${TARGET_SHARD_CAPACITY_SCALE_PPM}"
fi

if [ "${RUN_FULL_SMOKE}" = "1" ]; then
  for backend in ${SMOKE_BACKEND_SWEEP}; do
    run_full_config smoke "${backend}" "${SMOKE_BEAM_WIDTH}" "${SMOKE_DEPTH_LIMIT}" "${SMOKE_SHARD_COUNT}" "${SMOKE_B_MICRO}" "${SMOKE_CONCURRENCY}" "${BEAM_STREAM3_RING_SLOTS}" "${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}" "${SMOKE_SHARD_CAPACITY_SCALE_PPM}"
  done
fi

if [ "${RUN_TARGET_SWEEP}" = "1" ]; then
  for backend in ${TARGET_BACKEND_SWEEP}; do
    for shard in ${TARGET_SHARD_COUNT_SWEEP}; do
      for b_micro in ${TARGET_B_MICRO_SWEEP}; do
        for concurrency in ${TARGET_CONCURRENCY_SWEEP}; do
          for ring_slots in ${TARGET_RING_SLOTS_SWEEP}; do
            for final_chunk in ${TARGET_FINAL_CHUNK_SWEEP}; do
              run_full_config target "${backend}" "${TARGET_BEAM_WIDTH}" "${TARGET_DEPTH_LIMIT}" "${shard}" "${b_micro}" "${concurrency}" "${ring_slots}" "${final_chunk}" "${TARGET_SHARD_CAPACITY_SCALE_PPM}"
            done
          done
        done
      done
    done
  done
fi

if write_best_env; then
  echo "best_megaminx_transformer_env=${BEST_ENV}"
  cat "${BEST_ENV}"
else
  echo "best_megaminx_transformer_env=none"
fi

echo "benchmark_summary=${SUMMARY}"
echo "isolated_stream1_summary=${ISOLATED_SUMMARY}"
echo "pipeline_smoke_summary=${PIPELINE_SUMMARY}"
echo "finished_at=$(date -Is)"
