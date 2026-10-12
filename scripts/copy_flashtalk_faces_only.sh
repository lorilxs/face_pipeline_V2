#!/usr/bin/env bash
# Copy the newly-built *_faces_only.mp4 into the consolidated _new_pipeline/videos
# dir for the flashtalk batches, then remove the stale un-aligned verification
# mp4s (no _faces_only suffix) that an older consolidate copied from verification/.
# The verification/ source dirs are left untouched.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

for B in flashtalk_batch_000000 flashtalk_batch_010000; do
  NEW="$ROOT/outputs/$B/${B}_new_pipeline"
  [ -d "$NEW" ] || { echo "skip $B (no _new_pipeline)"; continue; }

  # 1) copy freshly built faces_only mp4s (skips files already present)
  copied=0
  while IFS= read -r -d '' f; do
    base=$(basename "$f")
    if [ -f "$NEW/videos/$base" ]; then
      : # already present -> leave as-is
    else
      cp "$f" "$NEW/videos/$base" && copied=$((copied+1))
    fi
  done < <(find "$ROOT/outputs/$B" -path '*/verification_face_only/*_faces_only.mp4' -print0)

  # 2) remove stale un-aligned verification mp4s (no _faces_only suffix)
  stale=0
  while IFS= read -r -d '' f; do
    case "$(basename "$f")" in
      *_faces_only.mp4) : ;;
      *) rm -f "$f"; stale=$((stale+1)) ;;
    esac
  done < <(find "$NEW/videos" -maxdepth 1 -type f -name '*.mp4' -print0)

  echo "$B: copied=$copied stale_removed=$stale videos_total=$(ls "$NEW/videos" | wc -l) faces_only=$(ls "$NEW/videos" | grep -c '_faces_only\.mp4$')"
done
