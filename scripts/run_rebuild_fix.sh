#!/usr/bin/env bash
# Targeted --force rebuild of face-only videos for truncated/FAILed roots.
# Roots come from logs/rebuild_roots_for_fix.txt (affected roots + 19 FAIL roots).
set -uo pipefail
cd "$(dirname "$0")/.."
mapfile -t ROOTS < logs/rebuild_roots_for_fix.txt
ARGS=()
for r in "${ROOTS[@]}"; do
    [ -n "$r" ] || continue
    ARGS+=(--sample-root "$r")
done
echo "rebuild ${#ARGS[@]} sample roots (pairs=$(( ${#ARGS[@]} / 2 ))) with --force --jobs ${JOBS:-32}"
exec python3 scripts/build_verification_face_only_videos.py --force --jobs "${JOBS:-32}" "${ARGS[@]}" > logs/rebuild_fix.log 2>&1
