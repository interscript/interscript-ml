#!/bin/bash
# Generic Modal retry wrapper — the resilience pattern this campaign
# re-implemented in /tmp seven times. Idempotent server-side functions
# make retries the resume mechanism.
#
#   scripts/modal_retry.sh <file::function> [args...] [retry_sleep_secs]
#
# Example:
#   scripts/modal_retry.sh src/gpu/modal_export.py::parity --model khm-latn 120
set -u
target="$1"; shift
sleep_secs="${!#}"  # last arg if numeric
if [[ "$sleep_secs" =~ ^[0-9]+$ ]]; then set -- "${@:1:$#-1}"; else sleep_secs=60; fi
until modal run --detach "$target" "$@"; do
  echo "[modal_retry] $target failed — retrying in ${sleep_secs}s ($(date))"
  sleep "$sleep_secs"
done
echo "[modal_retry] $target completed"
