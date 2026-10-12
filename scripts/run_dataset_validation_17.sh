#!/usr/bin/env bash
# Dataset-mode Stage17 (fusion) driver across all accepted clips.
# Requires stage12_flame / stage13_gaze / stage14_audio outputs to exist.
#
# Usage: bash scripts/run_dataset_validation_17.sh
set -u
ROOT=/mnt/home/lxs/face_pipeline/Face_Pipeline_V2
cd "$ROOT"
BASE=outputs/stage0_17_e2e/dataset_validation
mkdir -p logs

CLIPS=$(ls "${BASE}"/stage_inputs/*_aligned_faces.json \
    | sed 's|.*/||; s/_aligned_faces.json//' | sort -u)

for clip in $CLIPS; do
  if [ ! -f "${BASE}/stage12_flame/${clip}/flame_features.npz" ]; then
    echo "[S17] $clip SKIP (no stage12)"
    continue
  fi
  if [ ! -f "${BASE}/stage13_gaze/${clip}/gaze_features.npz" ]; then
    echo "[S17] $clip SKIP (no stage13)"
    continue
  fi
  if [ ! -f "${BASE}/stage14_audio/${clip}/audio_features.npz" ]; then
    echo "[S17] $clip SKIP (no stage14)"
    continue
  fi
  echo "[S17] $clip starting $(date +%H:%M:%S)"
  PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_17_fusion/runner.py \
      --clip_id "${clip}" \
      --flame_dir "${BASE}/stage12_flame" \
      --gaze_dir "${BASE}/stage13_gaze" \
      --audio_dir "${BASE}/stage14_audio" \
      --output_dir "${BASE}/stage17_fusion" \
      --fps 25 > "logs/stage17_${clip}.log" 2>&1
  echo "[S17] $clip rc=$? $(tail -1 "logs/stage17_${clip}.log")"
done
echo "ALL S17 DONE $(date +%H:%M:%S)"
