#!/bin/bash
set -a
. /etc/environment
set +a
test "${CONTAINER_ID:?}" = 54161582 || exit 3
deadline=1791139830
while (( $(date -u +%s) < deadline )); do sleep 15; done
for attempt in 1 2 3 4 5; do
  if vastai stop instance 54161582 --api-key "${CONTAINER_API_KEY:?}"; then exit 0; fi
  sleep 15
done
exit 1
