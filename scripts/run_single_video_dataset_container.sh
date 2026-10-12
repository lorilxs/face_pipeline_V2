#!/usr/bin/env bash
# Container-side one-shot driver for a SINGLE video (dataset mode, EMICA jaw).
#
# Run this INSIDE the container's interactive shell, e.g.:
#     docker exec -it face_pipeline_lxs bash
#     bash scripts/run_single_video_dataset_container.sh <container_video_path> [--smoke]
#
# Runs: Stage0-11 (system python3.12) -> Stage12 (emica_env) -> Stage13 (emica_env,
# mediapipe) -> Stage14 (system python3.12, HuBERT) -> Stage17 fusion -> production
# export + validation. Outputs go under outputs/video_<id>/.
#
# The video path must already be visible inside the container. Since /mnt/home/lxs/videos
# is NOT mounted, copy it into the project first from the HOST:
#     cp <source.mp4> test_data/video_<id>/<id>.mp4
# then pass test_data/video_<id>/<id>.mp4 here.
set -u
cd /workspace/Face_Pipeline_V2

VIDEO_PATH="${1:?usage: run_single_video_dataset_container.sh <container_video_path> [--smoke]}"
SMOKE=0
[ "${2:-}" = "--smoke" ] && SMOKE=1

if [ ! -f "$VIDEO_PATH" ]; then
    echo "ERROR: video not found inside container: $VIDEO_PATH"
    echo "  Hint: copy it from the host first, then pass a container-visible path."
    exit 1
fi

VIDEO_ID=$(basename "$VIDEO_PATH" .mp4)
BASE="outputs/video_${VIDEO_ID}/stage0_11"
mkdir -p logs

echo "================================================================"
echo "VIDEO_ID = $VIDEO_ID"
echo "BASE     = $BASE"
echo "SMOKE    = $SMOKE"
echo "================================================================"

# ---- Stage 0-11 (system python3.12, dataset mode) ----
echo "[S0-11] $VIDEO_ID starting $(date +%H:%M:%S)"
python3 scripts/stage0_17_e2e_validate.py \
    --video "$VIDEO_PATH" \
    --config configs/config_dataset_mode.yaml \
    --test-base "$BASE" 2>&1 | tee "logs/stage0_11_${VIDEO_ID}.log"
S011_RC=${PIPESTATUS[0]}
echo "[S0-11] rc=$S011_RC"
if [ "$S011_RC" -ne 0 ]; then
    exit 1
fi

CLIPS=$(ls "$BASE"/stage_inputs/*_aligned_faces.json 2>/dev/null \
    | sed 's|.*/||; s/_aligned_faces.json//' | sort -u)
if [ -z "$CLIPS" ]; then
    echo "ERROR: no accepted clips under $BASE/stage_inputs/"
    exit 1
fi
if [ "$SMOKE" = "1" ]; then
    CLIPS=$(echo "$CLIPS" | head -1)
    echo "SMOKE MODE: only $CLIPS"
fi

# ---- Stage 12 (EMICA, emica_env) / 13 (gaze, emica_env) / 14 (HuBERT, python3.12) ----
for clip in $CLIPS; do
    echo "########## $clip ##########"
    echo "[S12] $clip starting $(date +%H:%M:%S)"
    /workspace/emica_ppu_env/bin/python scripts/stage0_17_e2e_stage12.py \
        --input_metadata "$BASE/stage_inputs/${clip}_aligned_faces.json" \
        --output_dir "$BASE/stage12_flame" \
        --clip_id="$clip" --batch_size 8 > "logs/stage12_${clip}.log" 2>&1
    echo "[S12] $clip rc=$? $(tail -1 "logs/stage12_${clip}.log")"

    echo "[S13] $clip starting $(date +%H:%M:%S)"
    PYTHONPATH=/workspace/Face_Pipeline_V2 /workspace/emica_env/bin/python \
        face_pipeline_v2/stages/stage_13_gaze/runner.py \
        --input_metadata "$BASE/stage_inputs/${clip}_aligned_faces.json" \
        --output_dir "$BASE/stage13_gaze" \
        --model_path models/mediapipe/face_landmarker.task \
        --delegate cpu --clip_id="$clip" > "logs/stage13_${clip}.log" 2>&1
    echo "[S13] $clip rc=$? $(tail -1 "logs/stage13_${clip}.log")"

    echo "[S14] $clip starting $(date +%H:%M:%S)"
    python3 face_pipeline_v2/stages/stage_14_audio_feature/runner.py \
        --input_metadata "$BASE/stage_inputs/${clip}_av_alignment.json" \
        --output_dir "$BASE/stage14_audio" \
        --model_path models/hubert/chinese-hubert-base \
        --device cpu --fps 25 --clip_id="$clip" > "logs/stage14_${clip}.log" 2>&1
    echo "[S14] $clip rc=$? $(tail -1 "logs/stage14_${clip}.log")"
done

# ---- Stage 17 (fusion, system python3.12) ----
for clip in $CLIPS; do
    if [ ! -f "$BASE/stage12_flame/${clip}/flame_features.npz" ] \
       || [ ! -f "$BASE/stage13_gaze/${clip}/gaze_features.npz" ] \
       || [ ! -f "$BASE/stage14_audio/${clip}/audio_features.npz" ]; then
        echo "[S17] $clip SKIP (missing stage12/13/14)"
        continue
    fi
    echo "[S17] $clip starting $(date +%H:%M:%S)"
    python3 face_pipeline_v2/stages/stage_17_fusion/runner.py \
        --clip_id="$clip" \
        --flame_dir "$BASE/stage12_flame" \
        --gaze_dir "$BASE/stage13_gaze" \
        --audio_dir "$BASE/stage14_audio" \
        --output_dir "$BASE/stage17_fusion" \
        --fps 25 > "logs/stage17_${clip}.log" 2>&1
    echo "[S17] $clip rc=$? $(tail -1 "logs/stage17_${clip}.log")"
done

# ---- Production multimodal export + validation (system python3.12) ----
echo "[EXPORT] starting $(date +%H:%M:%S)"
python3 scripts/export_production_multimodal.py \
    --base "$BASE" \
    --export-dir "outputs/video_${VIDEO_ID}/stage17_multimodal_export" 2>&1 \
    | tee "logs/export_${VIDEO_ID}.log"
EXPORT_RC=${PIPESTATUS[0]}

echo "================================================================"
echo "ALL DONE $(date +%H:%M:%S) | S0-11 rc=$S011_RC | export rc=$EXPORT_RC"
echo "  BASE   = $BASE"
echo "  EXPORT = outputs/video_${VIDEO_ID}/stage17_multimodal_export"
echo "================================================================"
exit "$EXPORT_RC"
