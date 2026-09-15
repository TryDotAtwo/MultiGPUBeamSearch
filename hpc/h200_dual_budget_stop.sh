#!/usr/bin/env bash
set -eu
test "${CONTAINER_ID:?}" = 51125653
# Three hours at $9.112/h leaves a margin inside the remaining phase budget.
sleep 10800
vastai stop instance 51125653 --api-key "${CONTAINER_API_KEY:?}"
