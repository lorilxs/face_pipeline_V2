#!/usr/bin/env bash
# Stage12-17 driver for the Static Video Stability Audit (AISDD diagnostic).
#
# Runs Stage12 (FLAME/EMICA, docker emica_env) / Stage13 (gaze) / Stage14
# (audio, hubert) / Stage17 (fusion) on the SAME 1_8s.mp4 static video, saving
# into outputs/static_video_audit/ (never touching production outputs).
#
# Prereq: scripts/stage5_stage10_static_video_audit.py prep already ran
#         (produces stage11 aligned_faces + av_alignment).
#
# Usage:
#     bash scripts/run_static_video_audit_stage12_17.sh [--smoke]
set -u

ROOT=/mnt/home/lxs/face_pipeline/Face_Pipeline_V2
cd "$ROOT"
CLIP=static_1_8s
OUT=outputs/static_video_audit
ALIGNED_META="$OUT/stage11/${CLIP}_aligned_faces.json"
AV_ALIGN="$OUT/stage14_audio/${CLIP}_av_alignment.json"

mkdir -p logs "$OUT"

if [ ! -f "$ALIGNED_META" ]; then
    echo "ERROR: $ALIGNED_META missing -> run prep first"; exit 1
fi
if [ ! -f "$AV_ALIGN" ] || [ ! -f "$OUT/stage14_audio/audio/${CLIP}.wav" ]; then
    echo "ERROR: av_alignment / wav missing -> run prep first"; exit 1
fi

echo "=== Static Video Audit Stage12-17 : CLIP=$CLIP ==="

echo "[S12] starting $(date +%H:%M:%S)"
docker exec face_pipeline_lxs bash -lc \
    "cd /workspace/Face_Pipeline_V2 && /workspace/emica_ppu_env/bin/python \
        scripts/stage0_17_e2e_stage12.py \
        --input_metadata ${ALIGNED_META} \
        --output_dir ${OUT}/stage12_flame \
        --clip_id ${CLIP} --batch_size 8" > "logs/static_audit_stage12_${CLIP}.log" 2>&1
echo "[S12] rc=$? $(tail -1 "logs/static_audit_stage12_${CLIP}.log")"

echo "[S13] starting $(date +%H:%M:%S)"
PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_13_gaze/runner.py \
    --input_metadata "$ALIGNED_META" \
    --output_dir "$OUT/stage13_gaze" \
    --model_path models/mediapipe/face_landmarker.task \
    --delegate cpu --clip_id "$CLIP" > "logs/static_audit_stage13_${CLIP}.log" 2>&1
echo "[S13] rc=$? $(tail -1 "logs/static_audit_stage13_${CLIP}.log")"

echo "[S14] starting $(date +%H:%M:%S)"
PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_14_audio_feature/runner.py \
    --input_metadata "$AV_ALIGN" \
    --output_dir "$OUT/stage14_audio" \
    --model_path models/hubert/chinese-hubert-base \
    --device cpu --fps 25 --clip_id "$CLIP" > "logs/static_audit_stage14_${CLIP}.log" 2>&1
echo "[S14] rc=$? $(tail -1 "logs/static_audit_stage14_${CLIP}.log")"

echo "[S17] starting $(date +%H:%M:%S)"
PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_17_fusion/runner.py \
    --clip_id "$CLIP" \
    --flame_dir "$OUT/stage12_flame" \
    --gaze_dir "$OUT/stage13_gaze" \
    --audio_dir "$OUT/stage14_audio" \
    --output_dir "$OUT/stage17_fusion" \
    --fps 25 > "logs/static_audit_stage17_${CLIP}.log" 2>&1
echo "[S17] rc=$? $(tail -1 "logs/static_audit_stage17_${CLIP}.log")"

echo "=== ALL DONE $(date +%H:%M:%S) ==="
echo "  flame : $OUT/stage12_flame/$CLIP/flame_features.npz"
echo "  sample: $OUT/stage17_fusion/$CLIP/training_sample.npz"
