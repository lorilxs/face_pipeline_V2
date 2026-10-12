#!/usr/bin/env bash
# Remove stale *_faces_only.mp4 copies from *_new_pipeline/videos for the
# roots that were re-built (--force), so consolidate_batch_outputs.sh can
# refresh them (it skips existing files).
set -uo pipefail
cd "$(dirname "$0")/.."
DELETED=0
while IFS= read -r root; do
    [ -n "$root" ] || continue
    out_root="${root%%/*}"
    batch="${root#*/}"
    batch="${batch%%/*}"
    vid_dir="$out_root/$batch/${batch}_new_pipeline/videos"
    if [ ! -d "$vid_dir" ]; then
        continue
    fi
    while IFS= read -r -d '' f; do
        b="$(basename "$f")"
        if [ -e "$vid_dir/$b" ]; then
            rm -f "$vid_dir/$b"
            DELETED=$((DELETED + 1))
        fi
    done < <(find "$root/verification_face_only" -maxdepth 1 -type f -name '*_faces_only.mp4' ! -name '*_original.mp4' -print0 2>/dev/null)
done < logs/rebuild_roots_for_fix.txt
echo "deleted stale consolidated copies: $DELETED"
