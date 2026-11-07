#!/usr/bin/env bash
set -eu
supervisord -c /workspace/guard_supervisord_54195079.conf
sleep 2
supervisorctl -c /workspace/guard_supervisord_54195079.conf status
cat /workspace/stopguard_54195079.log
mkdir -p /workspace/pipeline_audit_54195079
tar -xzf /workspace/source_ab_final_54186691.tar.gz -C /workspace/pipeline_audit_54195079
cp /workspace/test_stream4_blocking_slot_cuda.py /workspace/pipeline_audit_54195079/tests/
/venv/main/bin/python -m pip install pytest pytest-subtests pandas jsonschema > /workspace/pipeline_dependencies_54195079.log 2>&1
cd /workspace/pipeline_audit_54195079
set +e
BEAM_TEST_CUDA_EVENT_SKEW=1 /venv/main/bin/python -m pytest -q -s tests/test_stream4_blocking_slot_cuda.py \
  > /workspace/blocking_slot_RED_54195079.log 2>&1
result=$?
set -e
tail -35 /workspace/blocking_slot_RED_54195079.log
echo blocking_slot_RED_exit=$result
if [ "$result" -ne 1 ]; then exit 1; fi
