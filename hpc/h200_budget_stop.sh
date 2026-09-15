#!/usr/bin/env bash
# Safety stop for this authorized 1-GPU session; storage remains billed.
set -eu
sleep 32400
vastai stop instance 51111383 --api-key "${CONTAINER_API_KEY:?}"
