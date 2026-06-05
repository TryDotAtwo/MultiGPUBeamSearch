#!/bin/bash
#SBATCH --job-name=beam8a100-pipe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

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
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-0}"

STREAM1_CONFIG_ENV="${STREAM1_CONFIG_ENV:-${LOG_DIR}/best_stream1.env}"
if [ -f "${STREAM1_CONFIG_ENV}" ]; then
  source "${STREAM1_CONFIG_ENV}"
fi
BEAM_B_MICRO="${BEAM_B_MICRO:-4096}"
BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-2}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-4}"

beam_preflight
beam_configure_build production_runner
beam_export_common_runtime

SUMMARY="${TUNING_DIR}/pipeline_sweep_${SLURM_JOB_ID:-manual}.tsv"
BEST_ENV="${TUNING_DIR}/best_pipeline.env"
echo -e "tag\tstatus\tavg_full_depth_sec\tshard_count\tb_micro\tstream1_concurrency\tstream3_ring_slots\tstream3_batch\tstream4_batch\tstream4_trigger\tfinal_chunk\tshard_capacity\tlogical_shard\tlog" > "${SUMMARY}"

SHARD_COUNT_SWEEP="${SHARD_COUNT_SWEEP:-8 12 16 24 32}"
STREAM3_RING_SLOTS_SWEEP="${STREAM3_RING_SLOTS_SWEEP:-2 4}"
STREAM4_BATCH_SWEEP="${STREAM4_BATCH_SWEEP:-262144 393216 524288 786432 1048576}"
STREAM4_TRIGGER_MULT_SWEEP="${STREAM4_TRIGGER_MULT_SWEEP:-1 2}"
FINAL_MATERIALIZE_CHUNK_SWEEP="${FINAL_MATERIALIZE_CHUNK_SWEEP:-0}"

best_avg=""
best_status=""

for shard in ${SHARD_COUNT_SWEEP}; do
  for ring_slots in ${STREAM3_RING_SLOTS_SWEEP}; do
    for batch in ${STREAM4_BATCH_SWEEP}; do
      for trigger_mult in ${STREAM4_TRIGGER_MULT_SWEEP}; do
        for final_chunk in ${FINAL_MATERIALIZE_CHUNK_SWEEP}; do
        SHARD_COUNT="${shard}"
        BEAM_STREAM3_RING_SLOTS="${ring_slots}"
        STREAM4_BATCH_CANDIDATES="${batch}"
        STREAM4_TRIGGER_CANDIDATES=$((batch * trigger_mult))
        BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${final_chunk}"
        beam_derive_shard_capacity
        tag="sh${SHARD_COUNT}_r${BEAM_STREAM3_RING_SLOTS}_b${STREAM4_BATCH_CANDIDATES}_t${STREAM4_TRIGGER_CANDIDATES}_fc${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}"
        log="${TUNING_DIR}/${tag}.log"

        if ! beam_validate_manual_config > "${TUNING_DIR}/${tag}.preflight" 2>&1; then
          status="INVALID"
          avg="NA"
          cat "${TUNING_DIR}/${tag}.preflight" | tee "${log}"
          echo -e "${tag}\t${status}\t${avg}\t${SHARD_COUNT}\t${BEAM_B_MICRO}\t${BEAM_STREAM1_CONCURRENCY}\t${BEAM_STREAM3_RING_SLOTS}\t${STREAM3_BATCH_CANDIDATES}\t${STREAM4_BATCH_CANDIDATES}\t${STREAM4_TRIGGER_CANDIDATES}\t${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}\t${SHARD_CAPACITY_CANDIDATES}\t${LOGICAL_SHARD_SIZE}\t${log}" >> "${SUMMARY}"
          continue
        fi

        beam_safe_clear_history_contents
        export BEAM_B_MICRO
        export BEAM_STREAM1_CONCURRENCY
        export BEAM_STREAM3_RING_SLOTS
        export BEAM_SHARD_BUFFER_COUNT
        beam_export_manual_config
        beam_prepare_nccl_file "${tag}"

        set +e
        beam_torchrun_production "${tag}" "${log}"
        rc=$?
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
        echo -e "${tag}\t${status}\t${avg}\t${SHARD_COUNT}\t${BEAM_B_MICRO}\t${BEAM_STREAM1_CONCURRENCY}\t${BEAM_STREAM3_RING_SLOTS}\t${STREAM3_BATCH_CANDIDATES}\t${STREAM4_BATCH_CANDIDATES}\t${STREAM4_TRIGGER_CANDIDATES}\t${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}\t${SHARD_CAPACITY_CANDIDATES}\t${LOGICAL_SHARD_SIZE}\t${log}" >> "${SUMMARY}"

        if [ "${status}" = "OK" ] && [ "${avg}" != "NA" ]; then
          if [ -z "${best_avg}" ] || awk "BEGIN { exit !(${avg} < ${best_avg}) }"; then
            best_avg="${avg}"
            best_status="${status}"
            {
              echo "export SHARD_COUNT=${SHARD_COUNT}"
              echo "export SHARD_CAPACITY_SCALE_PPM=${SHARD_CAPACITY_SCALE_PPM}"
              echo "export STREAM4_BATCH_ALIGNMENT=${STREAM4_BATCH_ALIGNMENT}"
              echo "export STREAM4_BATCH_CANDIDATES=${STREAM4_BATCH_CANDIDATES}"
              echo "export STREAM4_TRIGGER_CANDIDATES=${STREAM4_TRIGGER_CANDIDATES}"
              echo "export BEAM_B_MICRO=${BEAM_B_MICRO}"
              echo "export BEAM_STREAM1_CONCURRENCY=${BEAM_STREAM1_CONCURRENCY}"
              echo "export BEAM_STREAM3_RING_SLOTS=${BEAM_STREAM3_RING_SLOTS}"
              echo "export BEAM_STREAM4_ACTIVE_SORT_SLOTS=${BEAM_STREAM4_ACTIVE_SORT_SLOTS}"
              echo "export BEAM_SHARD_BUFFER_COUNT=${BEAM_SHARD_BUFFER_COUNT}"
              echo "export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}"
              echo "export BEST_PIPELINE_AVG_FULL_DEPTH_SEC=${best_avg}"
            } > "${BEST_ENV}"
            cp "${BEST_ENV}" "${LOG_DIR}/best_pipeline.env"
          fi
        fi
        done
      done
    done
  done
done

echo "pipeline_summary=${SUMMARY}"
if [ -n "${best_status}" ]; then
  echo "best_pipeline_env=${LOG_DIR}/best_pipeline.env"
  cat "${LOG_DIR}/best_pipeline.env"
else
  echo "best_pipeline_env=none"
fi
echo "finished_at=$(date -Is)"
