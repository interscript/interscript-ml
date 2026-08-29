#!/bin/bash
# Watch a Modal volume marker with debounce, then act (or just report).
# The pattern behind every poller this campaign needed.
#
#   scripts/watch_marker.sh <volume> <remote_path> [check_interval_secs] [cmd...]
#
# Prints WAITING lines to stderr, exits 0 when the marker appears, then
# runs cmd if given. No debounce needed for a marker that only appears.
set -u
volume="$1"; marker="$2"; interval="${3:-600}"; shift 3 2>/dev/null || shift $#
while true; do
  rm -f /tmp/.watch_marker_$$
  if modal volume get "$volume" "$marker" "/tmp/.watch_marker_$$" >/dev/null 2>&1; then
    [ -f /tmp/.watch_marker_$$ ] && break
  fi
  sleep "$interval"
done
echo "[watch_marker] $volume:$marker present ($(date))"
if [ "$#" -gt 0 ]; then exec "$@"; fi
