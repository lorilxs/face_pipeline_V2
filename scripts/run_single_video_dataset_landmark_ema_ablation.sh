#!/usr/bin/env bash
# Stage10 Landmark Temporal EMA (α=0.7) End-to-End Motion Preservation Ablation — entry driver.
#
# Baseline = production Stage10 raw 106 landmark (SCRFD raw bbox reused from
#            docs/stage5_roi_ablation/baseline_results.json, identical numbers).
# EMA      = inject an experiment-level causal EMA smoothing on the Stage10
#            landmark coordinates (landmark_ema[t] = 0.7*landmark_raw[t] +
#            0.3*landmark_ema[t-1], [106,2], frame 0 = raw, no future frames),
#            then re-run Stage11 align / Stage12 FLAME / Stage13 gaze / Stage14
#            audio / Stage17 fusion through the OFFICIAL production runners.
#
# The SCRFD bbox is IDENTICAL for both branches (only the landmark differs),
# so baseline and EMA are fully comparable.  Production code is NOT modified;
# only experiment scripts + outputs under outputs/ablation_landmark_ema/.
#
# Usage:
#     bash scripts/run_single_video_dataset_landmark_ema_ablation.sh [CLIP_ID] [--smoke]
#
# Env:
#     SRC_BASE   production Stage0-11 base (default outputs/stage0_17_e2e/stage0_11)
set -u

ROOT=/mnt/home/lxs/face_pipeline/Face_Pipeline_V2
cd "$ROOT"

SRC_BASE="${SRC_BASE:-outputs/stage0_17_e2e/stage0_11}"
OUT_BASE=outputs/ablation_landmark_ema
DEFAULT_CLIP="CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001"

CLIP="${1:-$DEFAULT_CLIP}"
SMOKE=0
[ "${2:-}" = "--smoke" ] && SMOKE=1
[ "${2:-}" = "" ] && [ "${3:-}" = "--smoke" ] && SMOKE=1

mkdir -p logs

echo "==================================================================="
echo "STAGE10 LANDMARK TEMPORAL EMA (alpha=0.7) END-TO-END ABLATION"
echo "CLIP     = $CLIP"
echo "SRC_BASE = $SRC_BASE   (production Stage0-11 run, baseline flow)"
echo "OUT_BASE = $OUT_BASE"
echo "SMOKE    = $SMOKE"
echo "==================================================================="

# ---- validate production baseline artifacts exist ----
CLIP_VIDEO="${SRC_BASE}/clips/${CLIP}.mp4"
ALIGNED_META="${SRC_BASE}/stage_inputs/${CLIP}_aligned_faces.json"

# Stage14 needs an av_alignment whose aligned audio actually exists.
_pick_av_alignment() {
    local clip="$1"
    for cand in "${SRC_BASE}/stage_inputs/${clip}_av_alignment.json" \
                "outputs/stage0_17_e2e/dataset_validation/stage_inputs/${clip}_av_alignment.json"; do
        if [ -f "$cand" ]; then
            local ap
            ap=$(/opt/hb/bin/python3 -c "import json; print(json.load(open('$cand')).get('aligned_audio_path') or '')" 2>/dev/null)
            if [ -n "$ap" ] && [ -f "$ap" ]; then
                echo "$cand"
                return 0
            fi
        fi
    done
    return 1
}
AV_ALIGN="$(_pick_av_alignment "$CLIP")" || AV_ALIGN=""
if [ ! -f "$ALIGNED_META" ] || [ ! -f "$CLIP_VIDEO" ] || [ -z "$AV_ALIGN" ]; then
    echo "ERROR: production Stage0-11 artifacts missing for clip ${CLIP}:"
    echo "  aligned_faces : $ALIGNED_META ($([ -f "$ALIGNED_META" ] && echo OK || echo MISSING))"
    echo "  clip video    : $CLIP_VIDEO ($([ -f "$CLIP_VIDEO" ] && echo OK || echo MISSING))"
    echo "  av_alignment  : ${AV_ALIGN:-NONE with valid aligned audio}"
    echo "Run the production flow first (scripts/run_single_video_dataset.sh)."
    exit 1
fi
echo "  AV_ALIGN  = $AV_ALIGN"
if [ ! -f docs/stage5_roi_ablation/baseline_results.json ]; then
    echo "ERROR: baseline SCRFD bbox source missing (docs/stage5_roi_ablation/baseline_results.json)"
    exit 1
fi

MAX_FRAMES=""
[ "$SMOKE" = "1" ] && MAX_FRAMES="--max-frames 30"

# =====================================================================
# 1) Stage5 (shared) / Stage10 raw+EMA landmark / Stage11 align (both branches)
# =====================================================================
echo "[PREP] starting $(date +%H:%M:%S)"
PYTHONPATH="$ROOT" python3 scripts/stage10_landmark_temporal_ema_ablation.py prep \
    --clip "$CLIP" --alpha 0.7 $MAX_FRAMES 2>&1 | tee "logs/lm_ema_prep_${CLIP}.log"
PREP_RC=${PIPESTATUS[0]}
if [ "$PREP_RC" -ne 0 ]; then
    echo "ERROR: prep failed rc=$PREP_RC (see logs/lm_ema_prep_${CLIP}.log)"
    exit 1
fi

# =====================================================================
# 2) Stage12 (EMICA, docker) / 13 (gaze) / 14 (audio) / 17 (fusion)
#    for BOTH branches (official production runners, unchanged params)
# =====================================================================
for branch in baseline ema_alpha07; do
    META="${OUT_BASE}/${branch}/stage11/${CLIP}_aligned_faces.json"
    echo "########## branch=$branch clip=$CLIP ##########"

    echo "[S12] $branch starting $(date +%H:%M:%S)"
    docker exec face_pipeline_lxs bash -lc \
        "cd /workspace/Face_Pipeline_V2 && /workspace/emica_ppu_env/bin/python \
            scripts/stage0_17_e2e_stage12.py \
            --input_metadata ${META} \
            --output_dir ${OUT_BASE}/${branch}/stage12_flame \
            --clip_id ${CLIP} --batch_size 8" > "logs/lm_ema_stage12_${branch}_${CLIP}.log" 2>&1
    echo "[S12] $branch rc=$? $(tail -1 "logs/lm_ema_stage12_${branch}_${CLIP}.log")"

    echo "[S13] $branch starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_13_gaze/runner.py \
        --input_metadata "${META}" \
        --output_dir "${OUT_BASE}/${branch}/stage13_gaze" \
        --model_path models/mediapipe/face_landmarker.task \
        --delegate cpu --clip_id "${CLIP}" > "logs/lm_ema_stage13_${branch}_${CLIP}.log" 2>&1
    echo "[S13] $branch rc=$? $(tail -1 "logs/lm_ema_stage13_${branch}_${CLIP}.log")"

    # Stage14 audio is bbox/landmark-independent -> identical for both branches
    echo "[S14] $branch starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_14_audio_feature/runner.py \
        --input_metadata "${AV_ALIGN}" \
        --output_dir "${OUT_BASE}/${branch}/stage14_audio" \
        --model_path models/hubert/chinese-hubert-base \
        --device cpu --fps 25 --clip_id "${CLIP}" > "logs/lm_ema_stage14_${branch}_${CLIP}.log" 2>&1
    echo "[S14] $branch rc=$? $(tail -1 "logs/lm_ema_stage14_${branch}_${CLIP}.log")"

    echo "[S17] $branch starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_17_fusion/runner.py \
        --clip_id "${CLIP}" \
        --flame_dir "${OUT_BASE}/${branch}/stage12_flame" \
        --gaze_dir "${OUT_BASE}/${branch}/stage13_gaze" \
        --audio_dir "${OUT_BASE}/${branch}/stage14_audio" \
        --output_dir "${OUT_BASE}/${branch}/stage17_fusion" \
        --fps 25 > "logs/lm_ema_stage17_${branch}_${CLIP}.log" 2>&1
    echo "[S17] $branch rc=$? $(tail -1 "logs/lm_ema_stage17_${branch}_${CLIP}.log")"
done

# =====================================================================
# 3) comparison metrics + overlay videos + report
# =====================================================================
echo "[ANALYZE] starting $(date +%H:%M:%S)"
PYTHONPATH="$ROOT" python3 scripts/stage10_landmark_temporal_ema_ablation.py analyze \
    2>&1 | tee "logs/lm_ema_analyze_${CLIP}.log"
ANA_RC=${PIPESTATUS[0]}

# Stage14 audio is identical for both branches -> verify byte-identical npz
if [ -f "${OUT_BASE}/baseline/stage14_audio/${CLIP}/audio_features.npz" ] && \
   [ -f "${OUT_BASE}/ema_alpha07/stage14_audio/${CLIP}/audio_features.npz" ]; then
    /opt/hb/bin/python3 -c "import numpy as np; a=np.load('${OUT_BASE}/baseline/stage14_audio/${CLIP}/audio_features.npz')['feature']; b=np.load('${OUT_BASE}/ema_alpha07/stage14_audio/${CLIP}/audio_features.npz')['feature']; print('stage14 audio identical:', bool(np.array_equal(a,b)), a.shape)"
fi

echo "==================================================================="
echo "ALL DONE $(date +%H:%M:%S) | prep rc=$PREP_RC | analyze rc=$ANA_RC"
echo "  OUT_BASE = $OUT_BASE"
echo "  REPORT   = docs/Stage10_Landmark_Temporal_EMA_Ablation_Report.md"
echo "==================================================================="
exit $ANA_RC
