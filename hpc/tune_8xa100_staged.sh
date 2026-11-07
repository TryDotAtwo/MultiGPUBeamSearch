#!/bin/bash
#SBATCH --job-name=beam8a100-stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${SCRIPT_DIR}/mephi_8xa100_common.sh"

beam_setup_paths
PUZZLE_ID="${PUZZLE_ID:-992}"
DEPTH_LIMIT="${DEPTH_LIMIT:-12}"
BEAM_WIDTH="${BEAM_WIDTH:-260000000}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1250000}"
BEAM_STREAM4_ACTIVE_SORT_SLOTS="${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}"
BEAM_SHARD_BUFFER_COUNT="${BEAM_SHARD_BUFFER_COUNT:-2}"

STREAM1_CONFIG_ENV="${STREAM1_CONFIG_ENV:-${LOG_DIR}/best_stream1.env}"
if [ -f "${STREAM1_CONFIG_ENV}" ]; then
  source "${STREAM1_CONFIG_ENV}"
fi

BEAM_B_MICRO="${BEAM_B_MICRO:-8192}"
BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-8}"
SHARD_COUNT="${SHARD_COUNT:-8}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
if [ "${BEAM_STREAM3_RING_SLOTS}" -lt "${BEAM_STREAM1_CONCURRENCY}" ]; then
  BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM1_CONCURRENCY}"
fi
STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-524288}"
STREAM4_TRIGGER_MULT="${STREAM4_TRIGGER_MULT:-2}"
STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-$((STREAM4_BATCH_CANDIDATES * STREAM4_TRIGGER_MULT))}"
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-262144}"
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-$(beam_default_final_exchange_scale_ppm)}"

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
beam_configure_build production_runner
beam_export_common_runtime

SUMMARY="${TUNING_DIR}/pipeline_staged_${SLURM_JOB_ID:-manual}.tsv"
BEST_ENV="${TUNING_DIR}/best_pipeline.env"
echo -e "stage\tvalue\ttag\tstatus\tavg_full_depth_sec\tshard_count\tb_micro\tstream1_concurrency\tstream3_ring_slots\tstream3_batch\tstream4_batch\tstream4_trigger\tstream4_trigger_mult\tfinal_chunk\tfinal_exchange_scale_ppm\tshard_capacity\tlogical_shard\tlog" > "${SUMMARY}"

SHARD_COUNT_SWEEP="${SHARD_COUNT_SWEEP:-4 8 16 24}"
STREAM3_RING_SLOTS_SWEEP="${STREAM3_RING_SLOTS_SWEEP:-8}"
STREAM4_BATCH_SWEEP="${STREAM4_BATCH_SWEEP:-262144 524288 1048576}"
STREAM4_TRIGGER_MULT_SWEEP="${STREAM4_TRIGGER_MULT_SWEEP:-1 2 3 4}"
FINAL_MATERIALIZE_CHUNK_SWEEP="${FINAL_MATERIALIZE_CHUNK_SWEEP:-65536 131072 262144 524288}"
FINAL_MATERIALIZE_EXCHANGE_SCALE_SWEEP="${FINAL_MATERIALIZE_EXCHANGE_SCALE_SWEEP:-1000000 1100000 1200000}"
LAST_AVG=""

write_best_env() {
  local best_avg="$1"
  {
    echo "export SHARD_COUNT=${SHARD_COUNT}"
    echo "export SHARD_CAPACITY_SCALE_PPM=${SHARD_CAPACITY_SCALE_PPM}"
    echo "export STREAM4_BATCH_ALIGNMENT=${STREAM4_BATCH_ALIGNMENT}"
    echo "export STREAM4_BATCH_CANDIDATES=${STREAM4_BATCH_CANDIDATES}"
    echo "export STREAM4_TRIGGER_CANDIDATES=${STREAM4_TRIGGER_CANDIDATES}"
    echo "export STREAM4_TRIGGER_MULT=${STREAM4_TRIGGER_MULT}"
    echo "export BEAM_B_MICRO=${BEAM_B_MICRO}"
    echo "export BEAM_STREAM1_CONCURRENCY=${BEAM_STREAM1_CONCURRENCY}"
    echo "export BEAM_STREAM3_RING_SLOTS=${BEAM_STREAM3_RING_SLOTS}"
    echo "export BEAM_STREAM4_ACTIVE_SORT_SLOTS=${BEAM_STREAM4_ACTIVE_SORT_SLOTS}"
    echo "export BEAM_SHARD_BUFFER_COUNT=${BEAM_SHARD_BUFFER_COUNT}"
    echo "export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}"
    echo "export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}"
    echo "export BEST_PIPELINE_AVG_FULL_DEPTH_SEC=${best_avg}"
  } > "${BEST_ENV}"
  cp "${BEST_ENV}" "${LOG_DIR}/best_pipeline.env"
}

apply_stage_value() {
  local stage="$1"
  local value="$2"
  case "${stage}" in
    shard_count)
      SHARD_COUNT="${value}"
      ;;
    stream3_ring_slots)
      BEAM_STREAM3_RING_SLOTS="${value}"
      ;;
    stream4_batch)
      STREAM4_BATCH_CANDIDATES="${value}"
      STREAM4_TRIGGER_CANDIDATES=$((STREAM4_BATCH_CANDIDATES * STREAM4_TRIGGER_MULT))
      ;;
    stream4_trigger_mult)
      STREAM4_TRIGGER_MULT="${value}"
      STREAM4_TRIGGER_CANDIDATES=$((STREAM4_BATCH_CANDIDATES * STREAM4_TRIGGER_MULT))
      ;;
    final_chunk)
      BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${value}"
      ;;
    final_exchange_scale)
      BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${value}"
      ;;
    *)
      echo "unknown_stage=${stage}"
      return 2
      ;;
  esac
}

run_current_config() {
  local stage="$1"
  local value="$2"
  beam_derive_shard_capacity
  local tag="${stage}_${value}_sh${SHARD_COUNT}_r${BEAM_STREAM3_RING_SLOTS}_b${STREAM4_BATCH_CANDIDATES}_tm${STREAM4_TRIGGER_MULT}_fc${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}_fs${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}"
  local log="${TUNING_DIR}/${tag}.log"
  local status
  local avg

  if ! beam_validate_manual_config > "${TUNING_DIR}/${tag}.preflight" 2>&1; then
    status="INVALID"
    avg="NA"
    cat "${TUNING_DIR}/${tag}.preflight" | tee "${log}"
    echo -e "${stage}\t${value}\t${tag}\t${status}\t${avg}\t${SHARD_COUNT}\t${BEAM_B_MICRO}\t${BEAM_STREAM1_CONCURRENCY}\t${BEAM_STREAM3_RING_SLOTS}\t${STREAM3_BATCH_CANDIDATES}\t${STREAM4_BATCH_CANDIDATES}\t${STREAM4_TRIGGER_CANDIDATES}\t${STREAM4_TRIGGER_MULT}\t${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}\t${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}\t${SHARD_CAPACITY_CANDIDATES}\t${LOGICAL_SHARD_SIZE}\t${log}" >> "${SUMMARY}"
    return 1
  fi

  beam_safe_clear_history_contents
  export BEAM_B_MICRO
  export BEAM_STREAM1_CONCURRENCY
  export BEAM_STREAM3_RING_SLOTS
  export BEAM_SHARD_BUFFER_COUNT
  export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM
  beam_export_manual_config
  beam_prepare_nccl_file "${tag}"

  set +e
  beam_torchrun_production "${tag}" "${log}"
  local rc=$?
  set -e

  if [ "${rc}" -eq 0 ]; then
    status="OK"
  else
    status="FAIL_${rc}"
  fi
  avg="$(awk '/depth_done=/ && /depth_sec=/ {
    depth=""; sec="";
    for (i=1; i<=NF; ++i) {
      if ($i ~ /^depth_done=/) { split($i,a,"="); depth=a[2]; }
      if ($i ~ /^depth_sec=/) { split($i,a,"="); sec=a[2]; }
    }
    if (depth >= 7 && sec != "") { sum += sec; n += 1; }
  } END { if (n > 0) printf "%.6f", sum / n; else printf "NA"; }' "${log}")"
  echo -e "${stage}\t${value}\t${tag}\t${status}\t${avg}\t${SHARD_COUNT}\t${BEAM_B_MICRO}\t${BEAM_STREAM1_CONCURRENCY}\t${BEAM_STREAM3_RING_SLOTS}\t${STREAM3_BATCH_CANDIDATES}\t${STREAM4_BATCH_CANDIDATES}\t${STREAM4_TRIGGER_CANDIDATES}\t${STREAM4_TRIGGER_MULT}\t${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}\t${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}\t${SHARD_CAPACITY_CANDIDATES}\t${LOGICAL_SHARD_SIZE}\t${log}" >> "${SUMMARY}"

  if [ "${status}" = "OK" ] && [ "${avg}" != "NA" ]; then
    LAST_AVG="${avg}"
    return 0
  fi
  LAST_AVG=""
  return 1
}

tune_stage() {
  local stage="$1"
  local values="$2"
  local base_shard="${SHARD_COUNT}"
  local base_ring="${BEAM_STREAM3_RING_SLOTS}"
  local base_batch="${STREAM4_BATCH_CANDIDATES}"
  local base_trigger_mult="${STREAM4_TRIGGER_MULT}"
  local base_trigger="${STREAM4_TRIGGER_CANDIDATES}"
  local base_final_chunk="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}"
  local base_final_scale="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}"
  local best_avg=""
  local best_value=""

  echo "stage_start=${stage} values=${values}"
  for value in ${values}; do
    SHARD_COUNT="${base_shard}"
    BEAM_STREAM3_RING_SLOTS="${base_ring}"
    STREAM4_BATCH_CANDIDATES="${base_batch}"
    STREAM4_TRIGGER_MULT="${base_trigger_mult}"
    STREAM4_TRIGGER_CANDIDATES="${base_trigger}"
    BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${base_final_chunk}"
    BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${base_final_scale}"
    apply_stage_value "${stage}" "${value}"
    if run_current_config "${stage}" "${value}"; then
      local avg="${LAST_AVG}"
      if [ -z "${best_avg}" ] || awk "BEGIN { exit !(${avg} < ${best_avg}) }"; then
        best_avg="${avg}"
        best_value="${value}"
      fi
    fi
  done

  SHARD_COUNT="${base_shard}"
  BEAM_STREAM3_RING_SLOTS="${base_ring}"
  STREAM4_BATCH_CANDIDATES="${base_batch}"
  STREAM4_TRIGGER_MULT="${base_trigger_mult}"
  STREAM4_TRIGGER_CANDIDATES="${base_trigger}"
  BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${base_final_chunk}"
  BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${base_final_scale}"

  if [ -z "${best_value}" ]; then
    echo "stage_best_${stage}=none"
    return 1
  fi
  apply_stage_value "${stage}" "${best_value}"
  write_best_env "${best_avg}"
  echo "stage_best_${stage}=${best_value} avg=${best_avg}"
  cat "${LOG_DIR}/best_pipeline.env"
}

tune_stage shard_count "${SHARD_COUNT_SWEEP}"
tune_stage stream3_ring_slots "${STREAM3_RING_SLOTS_SWEEP}"
tune_stage stream4_batch "${STREAM4_BATCH_SWEEP}"
tune_stage stream4_trigger_mult "${STREAM4_TRIGGER_MULT_SWEEP}"
tune_stage final_chunk "${FINAL_MATERIALIZE_CHUNK_SWEEP}"
tune_stage final_exchange_scale "${FINAL_MATERIALIZE_EXCHANGE_SCALE_SWEEP}"

echo "pipeline_summary=${SUMMARY}"
echo "best_pipeline_env=${LOG_DIR}/best_pipeline.env"
cat "${LOG_DIR}/best_pipeline.env"
echo "finished_at=$(date -Is)"
