#!/usr/bin/env bash
# Dataset-mode real-data validation driver for Stage12 (EMICA+SPECTRE, docker/CPU),
# Stage13 (gaze, host) and Stage14 (audio feature, host) across all accepted clips.
# Stage17 is run separately after these three complete.
#
# Usage: bash scripts/run_dataset_validation_121314.sh [--smoke]
set -u
ROOT=/mnt/home/lxs/face_pipeline/Face_Pipeline_V2
cd "$ROOT"
BASE=outputs/stage0_17_e2e/dataset_validation
mkdir -p logs

CLIPS=$(ls "${BASE}"/stage_inputs/*_aligned_faces.json \
    | sed 's|.*/||; s/_aligned_faces.json//' | sort -u)

if [ "${1:-}" = "--smoke" ]; then
    CLIPS=$(echo "$CLIPS" | head -1)
    echo "SMOKE MODE: only ${CLIPS}"
fi

for clip in $CLIPS; do
  echo "########## $clip ##########"
  echo "[S12] $clip starting $(date +%H:%M:%S)"
  docker exec face_pipeline_lxs bash -lc "cd /workspace/Face_Pipeline_V2 && /workspace/emica_ppu_env/bin/python scripts/stage0_17_e2e_stage12.py --input_metadata ${BASE}/stage_inputs/${clip}_aligned_faces.json --output_dir ${BASE}/stage12_flame --clip_id ${clip} --batch_size 8" > "logs/stage12_${clip}.log" 2>&1
  echo "[S12] $clip rc=$? $(tail -1 "logs/stage12_${clip}.log")"
  echo "[S13] $clip starting $(date +%H:%M:%S)"
  PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_13_gaze/runner.py \
      --input_metadata "${BASE}/stage_inputs/${clip}_aligned_faces.json" \
      --output_dir "${BASE}/stage13_gaze" \
      --model_path models/mediapipe/face_landmarker.task \
      --delegate cpu --clip_id "${clip}" > "logs/stage13_${clip}.log" 2>&1
  echo "[S13] $clip rc=$? $(tail -1 "logs/stage13_${clip}.log")"
  echo "[S14] $clip starting $(date +%H:%M:%S)"
  PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_14_audio_feature/runner.py \
      --input_metadata "${BASE}/stage_inputs/${clip}_av_alignment.json" \
      --output_dir "${BASE}/stage14_audio" \
      --model_path models/hubert/chinese-hubert-base \
      --device cpu --fps 25 --clip_id "${clip}" > "logs/stage14_${clip}.log" 2>&1
  echo "[S14] $clip rc=$? $(tail -1 "logs/stage14_${clip}.log")"
done
echo "ALL DONE $(date +%H:%M:%S)"
