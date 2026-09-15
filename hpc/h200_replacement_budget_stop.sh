#!/usr/bin/env bash
# Bounded replacement-H200 session. Stop preserves disk; never destroys data.
set -eu
test "${CONTAINER_ID:?}" = 51120383
sleep 18000
vastai stop instance 51120383 --api-key "${CONTAINER_API_KEY:?}"
