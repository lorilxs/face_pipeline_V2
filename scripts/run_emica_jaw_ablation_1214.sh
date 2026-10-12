#!/usr/bin/env bash
# EMICA-jaw ablation driver — re-run Stage12 (jaw_source=emica) + Stage13 + Stage14
# for all accepted clips into a NEW experimental base dir (production untouched).
#
# This is a TEMPORARY experiment only: Stage12 jaw_pose = EMICA (instead of the
# production SPECTRE Scheme-D) so the mesh team can compare both variants.
#
# Usage: bash scripts/run_emica_jaw_ablation_1214.sh [--smoke]
set -u
ROOT=/mnt/home/lxs/face_pipeline/Face_Pipeline_V2
cd "$ROOT"
SRC_BASE=outputs/stage0_17_e2e/dataset_validation   # production stage_inputs (aligned faces / AV alignment)
BASE=outputs/ablation_emica_jaw                     # NEW experimental base (do NOT touch production)
mkdir -p logs

CLIPS=$(ls "${SRC_BASE}"/stage_inputs/*_aligned_faces.json \
    | sed 's|.*/||; s/_aligned_faces.json//' | sort -u)

if [ "${1:-}" = "--smoke" ]; then
    CLIPS=$(echo "$CLIPS" | head -1)
    echo "SMOKE MODE: only ${CLIPS}"
fi

for clip in $CLIPS; do
  echo "########## $clip ##########"
  echo "[S12-EMICA] $clip starting $(date +%H:%M:%S)"
  docker exec face_pipeline_lxs bash -lc "cd /workspace/Face_Pipeline_V2 && /workspace/emica_ppu_env/bin/python scripts/stage0_17_e2e_stage12.py --input_metadata ${SRC_BASE}/stage_inputs/${clip}_aligned_faces.json --output_dir ${BASE}/stage12_flame --clip_id ${clip} --batch_size 8 --jaw_source emica" > "logs/ablation_stage12_${clip}.log" 2>&1
  echo "[S12-EMICA] $clip rc=$? $(tail -1 "logs/ablation_stage12_${clip}.log")"
  echo "[S13] $clip starting $(date +%H:%M:%S)"
  PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_13_gaze/runner.py \
      --input_metadata "${SRC_BASE}/stage_inputs/${clip}_aligned_faces.json" \
      --output_dir "${BASE}/stage13_gaze" \
      --model_path models/mediapipe/face_landmarker.task \
      --delegate cpu --clip_id "${clip}" > "logs/ablation_stage13_${clip}.log" 2>&1
  echo "[S13] $clip rc=$? $(tail -1 "logs/ablation_stage13_${clip}.log")"
  echo "[S14] $clip starting $(date +%H:%M:%S)"
  PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_14_audio_feature/runner.py \
      --input_metadata "${SRC_BASE}/stage_inputs/${clip}_av_alignment.json" \
      --output_dir "${BASE}/stage14_audio" \
      --model_path models/hubert/chinese-hubert-base \
      --device cpu --fps 25 --clip_id "${clip}" > "logs/ablation_stage14_${clip}.log" 2>&1
  echo "[S14] $clip rc=$? $(tail -1 "logs/ablation_stage14_${clip}.log")"
done
echo "ALL DONE $(date +%H:%M:%S)"
