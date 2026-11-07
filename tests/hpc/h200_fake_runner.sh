#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --runtime-library-info ]]; then
  # Synthetic library identity fixture, not an actual loaded NCCL assertion.
  "${H200_PYTHON:-python3}" - "$0" <<'PY'
import json, pathlib, sys
print(json.dumps(dict(schema_version=1, nccl_version=22501,
    nccl_library_path=str(pathlib.Path(sys.argv[1]).resolve()),
    communicator_initialized=False)))
PY
  exit 0
fi
if [[ ${1:-} == --build-info ]]; then
  echo '{"schema_version":1,"state_len":96,"state_storage_len":112,"move_count":24,"candidate_meta_bytes":32,"cuda_architectures":"'"${H200_TEST_BUILD_ARCH:-90a}"'"}'
  exit 0
fi
[[ -f ${H200_RESOLVED_PROFILE:-/nonexistent-profile} ]]
"${H200_PYTHON:-python3}" - "$H200_RESOLVED_PROFILE" <<'PY'
import json, os, sys
with open(sys.argv[1]) as stream:
    resolved = json.load(stream)
assert resolved['promotion_allowed'] is False
for key, value in resolved['parameters'].items():
    assert int(os.environ[key]) == value, key
assert resolved['derived']['effective_beam'] == 256376832
assert resolved['derived']['stream3_batch_candidates'] == 393216
PY
[[ $WORLD_SIZE == 8 && $RANK == "$LOCAL_RANK" ]]
[[ $BEAM_SHARD_CAPACITY_CANDIDATES == 500736 ]]
[[ $BEAM_B_MICRO == 2048 && $BEAM_STREAM1_TRANSFORMER_MICRO == 1024 ]]
[[ $BEAM_STREAM1_CONCURRENCY == 4 && $BEAM_STREAM3_RING_SLOTS == 8 ]]
[[ $BEAM_SHARD_COUNT == 64 && $BEAM_SHARD_BUFFER_COUNT == 2 ]]
[[ $BEAM_STREAM4_BATCH_CANDIDATES == 262144 && $BEAM_STREAM4_TRIGGER_CANDIDATES == 393216 ]]
[[ $BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM == 8000000 ]]
[[ $3 == 256000000 ]]
echo "mock_pass rank=$RANK"
