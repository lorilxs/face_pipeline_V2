#!/usr/bin/env bash
# Stage12 EMICA vs SPECTRE full-parameter ablation driver (AISDD).
#
# Runs, on the SAME aligned-face inputs (reused Stage0-11 outputs):
#   Experiment A (EMICA):  stage12_flame with all sources = emica
#   Experiment B (SPECTRE): stage12_flame with all sources = spectre
# then Stage13 gaze + Stage14 audio (source-independent, computed once and
# copied to both trees), Stage17 fusion + multimodal export per variant.
#
# Outputs (never touch production outputs/stage12_flame | outputs/stage17_fusion):
#   outputs/ablation_stage12_emica_vs_spectre/
#       emica/{stage12_flame,stage13_gaze,stage14_audio,stage17_fusion,stage17_multimodal_export}
#       spectre/{stage12_flame,stage13_gaze,stage14_audio,stage17_fusion,stage17_multimodal_export}
#
# Usage:
#     bash scripts/run_stage12_emica_vs_spectre_ablation.sh \
#         [INPUT_BASE] [MAX_CLIPS]
#
#   INPUT_BASE : dir with stage_inputs/*_aligned_faces.json + *_av_alignment.json
#                (default: outputs/stage0_17_e2e/dataset_validation)
#   MAX_CLIPS  : 0 = all clips (default); N = only first N clips (smoke test)
set -u

ROOT=/mnt/home/lxs/face_pipeline/Face_Pipeline_V2
cd "$ROOT"

INPUT_BASE="${1:-outputs/stage0_17_e2e/dataset_validation}"
MAX_CLIPS="${2:-0}"
ABLATION="outputs/ablation_stage12_emica_vs_spectre"

mkdir -p logs "$ABLATION/emica" "$ABLATION/spectre"

CLIPS=$(ls "$INPUT_BASE"/stage_inputs/*_aligned_faces.json 2>/dev/null \
    | sed 's|.*/||; s/_aligned_faces.json//' | sort -u)
if [ -z "$CLIPS" ]; then
    echo "ERROR: no aligned_faces metadata under $INPUT_BASE/stage_inputs/"
    exit 1
fi
if [ "$MAX_CLIPS" != "0" ]; then
    CLIPS=$(echo "$CLIPS" | head -"$MAX_CLIPS")
fi
N_CLIPS=$(echo "$CLIPS" | wc -l)
echo "===================================================================="
echo "INPUT_BASE = $INPUT_BASE"
echo "CLIPS      = $N_CLIPS"
echo "ABLATION   = $ABLATION"
echo "===================================================================="

for clip in $CLIPS; do
    echo "########## $clip ##########"

    echo "[S12-EMICA] $clip starting $(date +%H:%M:%S)"
    docker exec face_pipeline_lxs bash -lc \
        "cd /workspace/Face_Pipeline_V2 && /workspace/emica_ppu_env/bin/python \
            scripts/stage0_17_e2e_stage12.py \
            --input_metadata ${INPUT_BASE}/stage_inputs/${clip}_aligned_faces.json \
            --output_dir ${ABLATION}/emica/stage12_flame \
            --clip_id ${clip} --batch_size 8" > "logs/ablation_emica_vs_spectre_s12_emica_${clip}.log" 2>&1
    echo "[S12-EMICA] $clip rc=$? $(tail -1 "logs/ablation_emica_vs_spectre_s12_emica_${clip}.log")"

    echo "[S12-SPECTRE] $clip starting $(date +%H:%M:%S)"
    docker exec face_pipeline_lxs bash -lc \
        "cd /workspace/Face_Pipeline_V2 && /workspace/emica_ppu_env/bin/python \
            scripts/stage0_17_e2e_stage12.py \
            --input_metadata ${INPUT_BASE}/stage_inputs/${clip}_aligned_faces.json \
            --output_dir ${ABLATION}/spectre/stage12_flame \
            --clip_id ${clip} --batch_size 8 \
            --expression_source spectre --head_source spectre --jaw_source spectre" > "logs/ablation_emica_vs_spectre_s12_spectre_${clip}.log" 2>&1
    echo "[S12-SPECTRE] $clip rc=$? $(tail -1 "logs/ablation_emica_vs_spectre_s12_spectre_${clip}.log")"

    echo "[S13] $clip starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_13_gaze/runner.py \
        --input_metadata "$INPUT_BASE/stage_inputs/${clip}_aligned_faces.json" \
        --output_dir "$ABLATION/emica/stage13_gaze" \
        --model_path models/mediapipe/face_landmarker.task \
        --delegate cpu --clip_id "$clip" > "logs/ablation_emica_vs_spectre_s13_${clip}.log" 2>&1
    echo "[S13] $clip rc=$? $(tail -1 "logs/ablation_emica_vs_spectre_s13_${clip}.log")"

    echo "[S14] $clip starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_14_audio_feature/runner.py \
        --input_metadata "$INPUT_BASE/stage_inputs/${clip}_av_alignment.json" \
        --output_dir "$ABLATION/emica/stage14_audio" \
        --model_path models/hubert/chinese-hubert-base \
        --device cpu --fps 25 --clip_id "$clip" > "logs/ablation_emica_vs_spectre_s14_${clip}.log" 2>&1
    echo "[S14] $clip rc=$? $(tail -1 "logs/ablation_emica_vs_spectre_s14_${clip}.log")"
done

# ---- Stage13/14 outputs are source-independent: copy into spectre tree ----
mkdir -p "$ABLATION/spectre"
cp -rn "$ABLATION/emica/stage13_gaze" "$ABLATION/spectre/" 2>/dev/null
cp -rn "$ABLATION/emica/stage14_audio" "$ABLATION/spectre/" 2>/dev/null
echo "[COPY] gaze/audio -> spectre tree done"

# ---- Stage17 fusion + multimodal export per variant ----
for variant in emica spectre; do
    for clip in $CLIPS; do
        if [ -f "$ABLATION/$variant/stage12_flame/$clip/flame_features.npz" ] \
           && [ -f "$ABLATION/$variant/stage13_gaze/$clip/gaze_features.npz" ] \
           && [ -f "$ABLATION/$variant/stage14_audio/$clip/audio_features.npz" ]; then
            echo "[S17-$variant] $clip starting $(date +%H:%M:%S)"
            PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_17_fusion/runner.py \
                --clip_id "$clip" \
                --flame_dir "$ABLATION/$variant/stage12_flame" \
                --gaze_dir "$ABLATION/$variant/stage13_gaze" \
                --audio_dir "$ABLATION/$variant/stage14_audio" \
                --output_dir "$ABLATION/$variant/stage17_fusion" \
                --fps 25 > "logs/ablation_emica_vs_spectre_s17_${variant}_${clip}.log" 2>&1
            echo "[S17-$variant] $clip rc=$? $(tail -1 "logs/ablation_emica_vs_spectre_s17_${variant}_${clip}.log")"
        fi
    done
    echo "[EXPORT-$variant] starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 scripts/stage12_emica_vs_spectre_export.py \
        --base "$ABLATION/$variant" \
        --export-dir "$ABLATION/$variant/stage17_multimodal_export" \
        > "logs/ablation_emica_vs_spectre_export_${variant}.log" 2>&1
    echo "[EXPORT-$variant] rc=$? $(tail -1 "logs/ablation_emica_vs_spectre_export_${variant}.log")"
done

echo "===================================================================="
echo "ALL DONE $(date +%H:%M:%S)"
echo "  EMICA   = $ABLATION/emica"
echo "  SPECTRE = $ABLATION/spectre"
echo "===================================================================="

