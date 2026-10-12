#!/usr/bin/env bash
# Re-run of the 19,465 videos that FAILED Stage 3 in the 2026-09-06 batch run
# (min_face_confidence was 0.6 then; config now has 0.5).
#
# Previous run:
#   logs/batch_summary_20260906_170005.csv  (out-root=outputs/Chinese_LiPS_videos_train)
# Failed videos list: fail_videos_list_face_confidence0.5.txt (19,465 paths)
# New output root: outputs/Chinese_LiPS_videos_train_face_confidence0.5
set -u
cd /mnt/home/lxs/face_pipeline/Face_Pipeline_V2
bash scripts/run_batch_videos.sh \
  --list scripts/stage5_stage10_audit/fail_videos_list_face_confidence0.5.txt \
  --mode dataset \
  --out-root outputs/Chinese_LiPS_videos_train_face_confidence0.5 \
  --batch-size 200 \
  --parallel 16 \
  --skip-existing
