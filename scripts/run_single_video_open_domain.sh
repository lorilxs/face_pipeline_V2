#!/usr/bin/env bash
# Single-video OPEN-DOMAIN driver: FULL Stage0-17 chain (quality filtering ON).
#
# Mirrors scripts/run_single_video_dataset.sh, but runs the complete pipeline
# (configs/config.yaml -> pipeline_mode: open_domain, Stage 6/7/7.1/8/9 ENABLED)
# for ONE input video, with the EMICA jaw production scheme (jaw_source=emica),
# then the production multimodal export + validation.
#
# Difference vs run_single_video_dataset.sh:
#   - config:            configs/config.yaml  (full pipeline) instead of
#                        configs/config_dataset_mode.yaml (dataset, Stage6-9.5 off)
#   - output suffix:     outputs/video_<id>_open_domain/ instead of
#                        outputs/video_<id>/  (never collides with dataset runs)
#   - av_fix step:       repoints Stage14 aligned audio to the Stage9.5 exported
#                        asset (audio/<clip>.wav); without it Stage 9.5 cleanup
#                        leaves *_av_alignment.json pointing at a deleted file and
#                        Stage 14 fails. (Taken from run_batch_videos.sh.)
#
# Usage:
#     bash scripts/run_single_video_open_domain.sh <video_abs_path> [--smoke]
#
# Notes:
#   - If the input video already lives inside the project root it is used in
#     place (no copy); only videos outside the project are copied into
#     test_data/video_<id>/ so the docker container (mount: Face_Pipeline_V2
#     -> /workspace/Face_Pipeline_V2) can read them.
set -u

ROOT=/mnt/home/lxs/face_pipeline/Face_Pipeline_V2
cd "$ROOT"

VIDEO_PATH="${1:?usage: run_single_video_open_domain.sh <video_abs_path> [--smoke]}"
SMOKE=0
[ "${2:-}" = "--smoke" ] && SMOKE=1

if [ ! -f "$VIDEO_PATH" ]; then
    echo "ERROR: video not found: $VIDEO_PATH"
    exit 1
fi

VIDEO_ID=$(basename "$VIDEO_PATH" .mp4)
SUFFIX="_open_domain"
SRC_DIR="test_data/video_${VIDEO_ID}"
BASE="outputs/video_${VIDEO_ID}${SUFFIX}/stage0_11"
EXPORT_DIR="outputs/video_${VIDEO_ID}${SUFFIX}/stage17_multimodal_export"
mkdir -p logs

echo "======================================================================"
echo "MODE      = open_domain (FULL Stage0-17, Stage 6/7/7.1/8/9 enabled)"
echo "CONFIG    = configs/config.yaml"
echo "VIDEO_ID  = $VIDEO_ID"
echo "SRC_DIR   = $SRC_DIR"
echo "BASE      = $BASE"
echo "EXPORT    = $EXPORT_DIR"
echo "SMOKE     = $SMOKE"
echo "======================================================================"

# ---- resolve docker-visible video path (no copy when already in-project) ----
VIDEO_ABS=$(readlink -f "$VIDEO_PATH")
ROOT_ABS=$(readlink -f "$ROOT")
case "$VIDEO_ABS" in
    "$ROOT_ABS"|"$ROOT_ABS"/*)
        VIDEO_SRC=${VIDEO_ABS#"$ROOT_ABS"/}
        echo "[copy] in-project, no copy: $VIDEO_SRC"
        ;;
    *)
        mkdir -p "$SRC_DIR"
        if [ -f "$SRC_DIR/${VIDEO_ID}.mp4" ]; then
            echo "[copy] already present: $SRC_DIR/${VIDEO_ID}.mp4"
        else
            cp "$VIDEO_PATH" "$SRC_DIR/${VIDEO_ID}.mp4"
            echo "[copy] -> $SRC_DIR/${VIDEO_ID}.mp4"
        fi
        VIDEO_SRC="${SRC_DIR}/${VIDEO_ID}.mp4"
        ;;
esac

# ---- Stage 0-11 (FULL open_domain pipeline, docker python3.12) ----
echo "[S0-11] $VIDEO_ID starting $(date +%H:%M:%S)"
docker exec face_pipeline_lxs bash -lc \
    "cd /workspace/Face_Pipeline_V2 && python3 scripts/stage0_17_e2e_validate.py \
        --video ${VIDEO_SRC} \
        --config configs/config.yaml \
        --test-base ${BASE}" 2>&1 | tee "logs/stage0_11_${VIDEO_ID}${SUFFIX}.log"
S011_RC=${PIPESTATUS[0]}
echo "[S0-11] $VIDEO_ID rc=$S011_RC (see logs/stage0_11_${VIDEO_ID}${SUFFIX}.log)"
if [ "$S011_RC" -ne 0 ]; then
    echo "ERROR: Stage0-11 failed with rc=$S011_RC"
    exit 1
fi
mkdir -p "outputs/video_${VIDEO_ID}${SUFFIX}"
cp outputs/stage0_17_e2e/stage0_11_results.json \
   "outputs/video_${VIDEO_ID}${SUFFIX}/stage0_11_results_open_domain.json" 2>/dev/null || true

CLIPS=$(ls "${BASE}"/stage_inputs/*_aligned_faces.json 2>/dev/null \
    | sed 's|.*/||; s/_aligned_faces.json//' | sort -u)
if [ -z "$CLIPS" ]; then
    echo "ERROR: no accepted clips under ${BASE}/stage_inputs/"
    exit 1
fi
if [ "$SMOKE" = "1" ]; then
    CLIPS=$(echo "$CLIPS" | head -1)
    echo "SMOKE MODE: only $CLIPS"
fi

# ---- open_domain: repoint Stage14 aligned audio to exported asset ----
# (Stage 9.5 cleanup deletes stage_outputs/av_alignment/<clip>/, leaving
#  av_alignment.json pointing at a removed file -> Stage14 fails otherwise)
python3 - "$BASE" "$CLIPS" <<'PYEOF' || true
import json, os, sys
base, clips = sys.argv[1], sys.argv[2].split()
for clip in clips:
    p = f"{base}/stage_inputs/{clip}_av_alignment.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    t = d.get("aligned_audio_path")
    wav = f"{base}/audio/{clip}.wav"
    if t != wav:
        d["aligned_audio_path"] = wav
        d["corrected_audio_path"] = wav
        json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
        print(f"[avfix] {clip}: {t} -> {wav}")
PYEOF

# ---- Stage 12 (EMICA, docker/CPU) / 13 (gaze, host) / 14 (audio, host) ----
for clip in $CLIPS; do
    echo "########## $clip ##########"
    echo "[S12] $clip starting $(date +%H:%M:%S)"
    docker exec face_pipeline_lxs bash -lc \
        "cd /workspace/Face_Pipeline_V2 && /workspace/emica_ppu_env/bin/python \
            scripts/stage0_17_e2e_stage12.py \
            --input_metadata ${BASE}/stage_inputs/${clip}_aligned_faces.json \
            --output_dir ${BASE}/stage12_flame \
            --clip_id=${clip} --batch_size 8" > "logs/stage12_${clip}${SUFFIX}.log" 2>&1
    echo "[S12] $clip rc=$? $(tail -1 "logs/stage12_${clip}${SUFFIX}.log")"
    echo "[S13] $clip starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_13_gaze/runner.py \
        --input_metadata "${BASE}/stage_inputs/${clip}_aligned_faces.json" \
        --output_dir "${BASE}/stage13_gaze" \
        --model_path models/mediapipe/face_landmarker.task \
        --delegate cpu --clip_id="${clip}" > "logs/stage13_${clip}${SUFFIX}.log" 2>&1
    echo "[S13] $clip rc=$? $(tail -1 "logs/stage13_${clip}${SUFFIX}.log")"
    echo "[S14] $clip starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_14_audio_feature/runner.py \
        --input_metadata "${BASE}/stage_inputs/${clip}_av_alignment.json" \
        --output_dir "${BASE}/stage14_audio" \
        --model_path models/hubert/chinese-hubert-base \
        --device cpu --fps 25 --clip_id="${clip}" > "logs/stage14_${clip}${SUFFIX}.log" 2>&1
    echo "[S14] $clip rc=$? $(tail -1 "logs/stage14_${clip}${SUFFIX}.log")"
done

# ---- Stage 17 (fusion) ----
for clip in $CLIPS; do
    if [ ! -f "${BASE}/stage12_flame/${clip}/flame_features.npz" ] \
       || [ ! -f "${BASE}/stage13_gaze/${clip}/gaze_features.npz" ] \
       || [ ! -f "${BASE}/stage14_audio/${clip}/audio_features.npz" ]; then
        echo "[S17] $clip SKIP (missing stage12/13/14)"
        continue
    fi
    echo "[S17] $clip starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_17_fusion/runner.py \
        --clip_id="${clip}" \
        --flame_dir "${BASE}/stage12_flame" \
        --gaze_dir "${BASE}/stage13_gaze" \
        --audio_dir "${BASE}/stage14_audio" \
        --output_dir "${BASE}/stage17_fusion" \
        --fps 25 > "logs/stage17_${clip}${SUFFIX}.log" 2>&1
    echo "[S17] $clip rc=$? $(tail -1 "logs/stage17_${clip}${SUFFIX}.log")"
done

# ---- Production multimodal export + validation ----
echo "[EXPORT] starting $(date +%H:%M:%S)"
PYTHONPATH="$ROOT" python3 scripts/export_production_multimodal.py \
    --base "${BASE}" \
    --export-dir "${EXPORT_DIR}" 2>&1 \
    | tee "logs/export_${VIDEO_ID}${SUFFIX}.log"
EXPORT_RC=${PIPESTATUS[0]}

# ---- Verification videos (video + audio merged, for manual mesh review) ----
# Standalone build_verification_videos.py muxes the clip picture with the audio
# Stage 14 actually consumed (open_domain: aligned audio/<clip>.wav + original).
echo "[VERIFY] building verification videos $(date +%H:%M:%S)"
PYTHONPATH="$ROOT" python3 scripts/build_verification_videos.py \
    --sample-root "outputs/video_${VIDEO_ID}${SUFFIX}" > "logs/verify_${VIDEO_ID}${SUFFIX}.log" 2>&1
VERIFY_RC=$?
echo "[VERIFY] rc=$VERIFY_RC | $(tail -1 "logs/verify_${VIDEO_ID}${SUFFIX}.log")"

echo "======================================================================"
echo "ALL DONE $(date +%H:%M:%S) | S0-11 rc=$S011_RC | export rc=$EXPORT_RC"
echo "  BASE    = $BASE"
echo "  EXPORT  = $EXPORT_DIR"
echo "======================================================================"
exit $EXPORT_RC

