#!/usr/bin/env bash
# Wait for the verification-face-only batch to finish, then run consolidation
# for each of the four output roots. Logged to logs/watch_face_only_consolidate.log.
set -u

ROOT=/mnt/home/lxs/face_pipeline/Face_Pipeline_V2
LOG="$ROOT/logs/face_only_pass_batch.log"
WATCH_LOG="$ROOT/logs/watch_face_only_consolidate.log"

{
  echo "[watcher] started $(date +%F_%T)"

  # Wait until the batch process is gone. The pattern below only matches the
  # batch's own command line (this watcher's cmdline does not contain it).
  while pgrep -f 'python3 scripts/build_verification_face_only_videos.py' >/dev/null 2>&1; do
      sleep 30
  done
  echo "[watcher] batch process finished $(date +%F_%T)"
  sleep 5
  echo "----- tail of batch log -----"
  tail -12 "$LOG"

  for R in Talkvid_Chinese_videos_clips001 Talkvid_Chinese_videos_clips002 Talkvid_Chinese_videos_clips003 chinese_talking_videos_bilibili; do
      BATCH_DIR="$ROOT/outputs/$R"
      if [ ! -d "$BATCH_DIR" ]; then
          echo "[watcher] missing $BATCH_DIR — skip"
          continue
      fi
      echo "[watcher] consolidating $R @ $(date +%F_%T)"
      bash "$ROOT/scripts/consolidate_batch_outputs.sh" "$BATCH_DIR" \
          > "$ROOT/logs/consolidate_${R}.log" 2>&1
      rc=$?
      echo "[watcher] consolidate $R rc=$rc @ $(date +%F_%T)"
  done

  echo "[watcher] all done $(date +%F_%T)"
} >> "$WATCH_LOG" 2>&1