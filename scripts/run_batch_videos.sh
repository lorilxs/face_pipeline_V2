#!/usr/bin/env bash
# Batch driver for the single-video Stage0-17 chain (dataset | open_domain mode).
#
# Loops the logic of scripts/run_single_video_dataset.sh over MANY input videos,
# with configurable input, output root, and batched execution:
#   Stage0-11 (docker python3.12) -> Stage12 (emica_env) -> Stage13/14 (host) ->
#   Stage17 (host) -> production export + validation.
#
# Batching: use --batch-size N + --max-batches M to run M batches of N videos
# each and stop.  Every batch writes its own summary CSV + run log.
#
# Usage:
#     bash scripts/run_batch_videos.sh --src <video_dir> [options]
#     bash scripts/run_batch_videos.sh --list <paths_file> [options]
#     bash scripts/run_batch_videos.sh <video1.mp4> <video2.mp4> ... [options]
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

DOCKER_CONTAINER="${DOCKER_CONTAINER:-face_pipeline_lxs}"
MODE=dataset
OUT_ROOT="outputs"
BATCH_SIZE=0          # 0 -> all collected videos in one batch
MAX_BATCHES=0         # 0 -> run all batches
PARALLEL=1
LIMIT=
SMOKE=0
SKIP_EXISTING=0
DRY_RUN=0
SRC_DIR_ARG=
LIST_FILE=
VIDEO_ARGS=()

usage() {
    cat <<'EOF'
Batch driver for the single-video Stage0-17 chain (dataset | open_domain).

Usage:
  bash scripts/run_batch_videos.sh --src <video_dir> [--mode dataset|open_domain] [options]
  bash scripts/run_batch_videos.sh --list <paths_file> [options]
  bash scripts/run_batch_videos.sh <video1.mp4> <video2.mp4> ... [options]

Options:
  --src DIR         Input: process every *.mp4 under DIR (sorted).
  --list FILE       Input: one video path per line (# comment / blank ignored).
  --mode MODE       dataset (default) | open_domain
  --out-root DIR    Output base dir (relative to project root, default: outputs).
                    Must live INSIDE the project (docker mounts only the project).
  --batch-size N    Process N videos per batch (default: all videos in one batch).
  --max-batches N   Stop after running N batches (default: run all batches).
  --parallel N      Run up to N videos concurrently (default 1; N>1 -> s0_11 timing NA).
  --limit N         Global cap on total videos to process.
  --smoke           Only the first accepted clip per video.
  --skip-existing   Skip videos whose production export already exists.
  --dry-run         Print the plan only; do not execute.

Outputs (dataset mode mirrors run_single_video_dataset.sh exactly):
  <out-root>/video_<id>/...                 dataset mode
  <out-root>/video_<id>_open_domain/...     open_domain mode
  logs/batch_<ts>_b<K>.log                 run log of batch K
  logs/batch_summary_<ts>_b<K>.csv         one row per video of batch K
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --src)          SRC_DIR_ARG="${2:?--src needs a dir}"; shift 2 ;;
        --list)         LIST_FILE="${2:?--list needs a file}"; shift 2 ;;
        --mode)         MODE="${2:?--mode needs dataset|open_domain}"; shift 2 ;;
        --out-root)     OUT_ROOT="${2:?--out-root needs a dir}"; shift 2 ;;
        --batch-size)   BATCH_SIZE="${2:?--batch-size needs N}"; shift 2 ;;
        --max-batches)  MAX_BATCHES="${2:?--max-batches needs N}"; shift 2 ;;
        --parallel)     PARALLEL="${2:-1}"; shift 2 ;;
        --limit)        LIMIT="${2:?--limit needs N}"; shift 2 ;;
        --smoke)        SMOKE=1; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      usage ;;
        --) shift; while [ $# -gt 0 ]; do VIDEO_ARGS+=("$1"); shift; done ;;
        -*) echo "unknown option: $1"; usage ;;
        *)  VIDEO_ARGS+=("$1"); shift ;;
    esac
done

case "$MODE" in
    dataset)     CONFIG=configs/config_dataset_mode.yaml; SUFFIX="" ;;
    open_domain) CONFIG=configs/config.yaml;             SUFFIX="_open_domain" ;;
    *) echo "ERROR: --mode must be dataset|open_domain (got '$MODE')"; exit 1 ;;
esac

# numeric sanity for batch/parallel/limit
for v in BATCH_SIZE MAX_BATCHES PARALLEL; do
    case "${!v}" in ''|*[!0-9]*) echo "ERROR: --$v must be a non-negative integer (got '${!v}')"; exit 1 ;; esac
done

# ---- resolve output root (must live inside the project for docker visibility) ----
OUT_ROOT_ABS=$(readlink -f "$OUT_ROOT" 2>/dev/null || echo "$OUT_ROOT")
case "$OUT_ROOT_ABS" in
    "$ROOT"|"$ROOT"/*) : ;;
    *)
        echo "ERROR: --out-root must be inside the project: $ROOT"
        echo "  Got: $OUT_ROOT_ABS  (the docker container only mounts the project dir)"
        exit 1 ;;
esac
OUT_ROOT_REL=${OUT_ROOT_ABS#"$ROOT"/}
OUT_ROOT_REL=${OUT_ROOT_REL:-.}
mkdir -p "$OUT_ROOT_REL" logs

# ---- collect videos ----
VIDEOS=()
if [ -n "$SRC_DIR_ARG" ]; then
    [ -d "$SRC_DIR_ARG" ] || { echo "ERROR: --src dir not found: $SRC_DIR_ARG"; exit 1; }
    while IFS= read -r f; do VIDEOS+=("$f"); done < <(find "$SRC_DIR_ARG" -maxdepth 1 -type f -name '*.mp4' | sort)
elif [ -n "$LIST_FILE" ]; then
    [ -f "$LIST_FILE" ] || { echo "ERROR: --list file not found: $LIST_FILE"; exit 1; }
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        [ -n "$line" ] && VIDEOS+=("$line")
    done < "$LIST_FILE"
elif [ ${#VIDEO_ARGS[@]} -gt 0 ]; then
    VIDEOS=("${VIDEO_ARGS[@]}")
else
    echo "ERROR: nothing to run. Use --src <dir>, --list <file>, or pass video paths."
    usage
fi

if [ -n "$LIMIT" ]; then
    VIDEOS=("${VIDEOS[@]:0:$LIMIT}")
fi
if [ ${#VIDEOS[@]} -eq 0 ]; then echo "ERROR: no videos found."; exit 1; fi

for v in "${VIDEOS[@]}"; do
    [ -f "$v" ] || { echo "ERROR: video not found: $v"; exit 1; }
done

# ---- batch plan ----
if [ "$BATCH_SIZE" -gt 0 ]; then
    TOTAL_BATCHES=$(( (${#VIDEOS[@]} + BATCH_SIZE - 1) / BATCH_SIZE ))
    if [ "$MAX_BATCHES" -gt 0 ] && [ "$MAX_BATCHES" -lt "$TOTAL_BATCHES" ]; then
        TOTAL_BATCHES=$MAX_BATCHES
    fi
else
    TOTAL_BATCHES=1
    BATCH_SIZE=${#VIDEOS[@]}
fi

# sanity: docker container must be reachable (S0-11 and S12 run inside it)
if [ "$DRY_RUN" != "1" ]; then
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$DOCKER_CONTAINER"; then
        echo "ERROR: docker container '$DOCKER_CONTAINER' not running."
        echo "  Start it first, or set DOCKER_CONTAINER=<name>."
        exit 1
    fi
fi

TS=$(date +%Y%m%d_%H%M%S)
TMPDIR="logs/batch_tmp_${TS}"
mkdir -p "$TMPDIR"

# =====================================================================
# run_one_video: full Stage0-17 chain for a single video (mirrors
# run_single_video_dataset.sh), writes one CSV row to $part on exit.
#   $1 video path, $2 part-file, $3 batch number
# =====================================================================
run_one_video() {
    local video="$1" part="$2" batch_no="$3"
    local VIDEO_ID SRC_DIR BASE EXPORT_DIR VIDEO_SRC
    local S011_RC EXPORT_RC wall_start wall_end wall_s s0_total
    local CLIPS clip rc stage_fail status

    VIDEO_ID=$(basename "$video" .mp4)
    SRC_DIR="test_data/video_${VIDEO_ID}"
    BASE="${OUT_ROOT_REL}/video_${VIDEO_ID}${SUFFIX}/stage0_11"
    EXPORT_DIR="${OUT_ROOT_REL}/video_${VIDEO_ID}${SUFFIX}/stage17_multimodal_export"

    echo "[BATCH] ================== $VIDEO_ID ($MODE, batch $batch_no) =================="
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$EXPORT_DIR/manifest.csv" ]; then
        echo "[BATCH] SKIP existing export: $EXPORT_DIR"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$batch_no" "$VIDEO_ID" "$MODE" "SKIP_EXISTING" "NA" "0" "NA" "$BASE" "$EXPORT_DIR" "$(date +%Y-%m-%d\ %H:%M:%S)" > "$part"
        return 0
    fi

    wall_start=$(date +%s)

    # ---- resolve docker-visible video path (no copy when already in-project) ----
    # The docker container mounts the project root at /workspace/Face_Pipeline_V2,
    # so a video already inside the project is passed by its project-relative
    # path directly.  Only videos outside the project are copied into
    # test_data/video_<id>/ to make them docker-visible.
    VIDEO_ABS=$(readlink -f "$video")
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
                cp "$video" "$SRC_DIR/${VIDEO_ID}.mp4"
                echo "[copy] -> $SRC_DIR/${VIDEO_ID}.mp4"
            fi
            VIDEO_SRC="${SRC_DIR}/${VIDEO_ID}.mp4"
            ;;
    esac

    # ---- Stage 0-11 ----
    echo "[S0-11] $VIDEO_ID starting $(date +%H:%M:%S)"
    docker exec "$DOCKER_CONTAINER" bash -lc \
        "cd /workspace/Face_Pipeline_V2 && python3 scripts/stage0_17_e2e_validate.py \
            --video ${VIDEO_SRC} \
            --config ${CONFIG} \
            --test-base ${BASE}" 2>&1 | tee "logs/stage0_11_${VIDEO_ID}${SUFFIX}.log"
    S011_RC=${PIPESTATUS[0]}
    echo "[S0-11] $VIDEO_ID rc=$S011_RC"
    if [ "$S011_RC" -ne 0 ]; then
        # P3: S0-11 failures are never retried in this run -> drop their partial stage0_11
        # (diagnosis stays available via logs/stage0_11_<id>_open_domain.log)
        # NOTE: BASE already == <video_dir>/stage0_11 here.
        [ -d "${BASE}" ] && rm -rf -- "${BASE}" && echo "[clean] stage0_11 removed after S0-11 FAIL (${VIDEO_ID})"
        echo "[BATCH] $VIDEO_ID FAIL(S0-11:$S011_RC)"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$batch_no" "$VIDEO_ID" "$MODE" "FAIL(S0-11:$S011_RC)" "NA" "$(($(date +%s)-wall_start))" "NA" "$BASE" "$EXPORT_DIR" "$(date +%Y-%m-%d\ %H:%M:%S)" > "$part"
        return 1
    fi
    mkdir -p "${OUT_ROOT_REL}/video_${VIDEO_ID}${SUFFIX}"
    cp outputs/stage0_17_e2e/stage0_11_results.json \
       "${OUT_ROOT_REL}/video_${VIDEO_ID}${SUFFIX}/stage0_11_results_${MODE}.json" 2>/dev/null || true

    # ---- accepted clips ----
    CLIPS=$(ls "${BASE}"/stage_inputs/*_aligned_faces.json 2>/dev/null \
        | sed 's|.*/||; s/_aligned_faces.json//' | sort -u)
    if [ -z "$CLIPS" ]; then
        echo "[BATCH] $VIDEO_ID FAIL(no_clips)"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$batch_no" "$VIDEO_ID" "$MODE" "FAIL(no_clips)" "NA" "$(($(date +%s)-wall_start))" "0" "$BASE" "$EXPORT_DIR" "$(date +%Y-%m-%d\ %H:%M:%S)" > "$part"
        return 1
    fi
    if [ "$SMOKE" = "1" ]; then
        CLIPS=$(echo "$CLIPS" | head -1)
        echo "SMOKE MODE: only $CLIPS"
    fi

    # ---- open_domain: repoint Stage14 aligned audio to exported asset ----
    # (Stage 9.5 cleanup deletes stage_outputs/av_alignment/<clip>/, leaving
    #  av_alignment.json pointing at a removed file -> Stage14 fails otherwise)
    if [ "$MODE" = "open_domain" ]; then
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
    fi

    stage_fail=""

    # ---- Stage 12 (EMICA FLAME; inside container, PPU) — clips in parallel (-P 2 on GPU0) ----
    S12_FAILS=/tmp/s12fail_${VIDEO_ID}_$$
    rm -f "$S12_FAILS"
    stage12_one() {
        local clip="$1"
        echo "[S12] $clip starting $(date +%H:%M:%S)"
        docker exec "$DOCKER_CONTAINER" bash -lc \
            "cd /workspace/Face_Pipeline_V2 && /workspace/emica_ppu_env/bin/python \
                scripts/stage0_17_e2e_stage12.py \
                --input_metadata ${BASE}/stage_inputs/${clip}_aligned_faces.json \
                --output_dir ${BASE}/stage12_flame \
                --clip_id=${clip} --batch_size 8" > "logs/stage12_${clip}.log" 2>&1
        rc=$?
        echo "[S12] $clip rc=$rc $(tail -1 "logs/stage12_${clip}.log")"
        [ "$rc" -ne 0 ] && echo "$clip" >> "$S12_FAILS"
    }
    export -f stage12_one
    export DOCKER_CONTAINER BASE S12_FAILS
    echo "$CLIPS" | xargs -P 2 -I{} bash -c 'stage12_one "$1"' _ {}
    if [ -s "$S12_FAILS" ]; then
        stage_fail="S12:$(head -1 "$S12_FAILS")"
        echo "[S12] FAILED clips: $(wc -l < "$S12_FAILS")"
    fi
    rm -f "$S12_FAILS"

    # ---- Stage 13 (gaze; host) + Stage 14 (audio; host) — clips in parallel (-P 4) ----
    # P2: bump OMP threads for the CPU-bound host stages (machine has 256 cores).
    # Default 16; override externally to rebalance against --parallel, e.g.
    #   OMP_NUM_THREADS=8 bash scripts/run_batch_videos.sh ... --parallel 32
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
           MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" \
           OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}" \
           NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
    S13_FAILS=/tmp/s1314fail_${VIDEO_ID}_$$
    rm -f "$S13_FAILS"
    stage13_14_one() {
        local clip="$1"
        echo "[S13] $clip starting $(date +%H:%M:%S)"
        PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_13_gaze/runner.py \
            --input_metadata "${BASE}/stage_inputs/${clip}_aligned_faces.json" \
            --output_dir "${BASE}/stage13_gaze" \
            --model_path models/mediapipe/face_landmarker.task \
            --delegate cpu --clip_id="${clip}" > "logs/stage13_${clip}.log" 2>&1
        rc=$?
        echo "[S13] $clip rc=$rc $(tail -1 "logs/stage13_${clip}.log")"
        [ "$rc" -ne 0 ] && echo "S13:$clip" >> "$S13_FAILS"

        echo "[S14] $clip starting $(date +%H:%M:%S)"
        PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_14_audio_feature/runner.py \
            --input_metadata "${BASE}/stage_inputs/${clip}_av_alignment.json" \
            --output_dir "${BASE}/stage14_audio" \
            --model_path models/hubert/chinese-hubert-base \
            --device cpu --fps 25 --clip_id="${clip}" > "logs/stage14_${clip}.log" 2>&1
        rc=$?
        echo "[S14] $clip rc=$rc $(tail -1 "logs/stage14_${clip}.log")"
        [ "$rc" -ne 0 ] && echo "S14:$clip" >> "$S13_FAILS"
    }
    export -f stage13_14_one
    export ROOT BASE S13_FAILS
    echo "$CLIPS" | xargs -P 4 -I{} bash -c 'stage13_14_one "$1"' _ {}
    if [ -s "$S13_FAILS" ]; then
        stage_fail="$(head -1 "$S13_FAILS")"
        echo "[S13/S14] FAILED clips: $(wc -l < "$S13_FAILS")"
    fi
    rm -f "$S13_FAILS"

    # ---- Stage 17 (fusion) — clips in parallel (-P 6) ----
    S17_FAILS=/tmp/s17fail_${VIDEO_ID}_$$
    rm -f "$S17_FAILS"
    stage17_one() {
        local clip="$1"
        if [ ! -f "${BASE}/stage12_flame/${clip}/flame_features.npz" ] \
           || [ ! -f "${BASE}/stage13_gaze/${clip}/gaze_features.npz" ] \
           || [ ! -f "${BASE}/stage14_audio/${clip}/audio_features.npz" ]; then
            echo "[S17] $clip SKIP (missing stage12/13/14)"
            echo "missing:$clip" >> "$S17_FAILS"
            return 0
        fi
        echo "[S17] $clip starting $(date +%H:%M:%S)"
        PYTHONPATH="$ROOT" python3 face_pipeline_v2/stages/stage_17_fusion/runner.py \
            --clip_id="${clip}" \
            --flame_dir "${BASE}/stage12_flame" \
            --gaze_dir "${BASE}/stage13_gaze" \
            --audio_dir "${BASE}/stage14_audio" \
            --output_dir "${BASE}/stage17_fusion" \
            --fps 25 > "logs/stage17_${clip}.log" 2>&1
        rc=$?
        echo "[S17] $clip rc=$rc $(tail -1 "logs/stage17_${clip}.log")"
        [ "$rc" -ne 0 ] && echo "S17:$clip" >> "$S17_FAILS"
    }
    export -f stage17_one
    export ROOT BASE S17_FAILS
    echo "$CLIPS" | xargs -P 6 -I{} bash -c 'stage17_one "$1"' _ {}
    if [ -s "$S17_FAILS" ]; then
        [ -z "$stage_fail" ] && stage_fail="$(head -1 "$S17_FAILS")"
        echo "[S17] FAILED/SKIP clips: $(wc -l < "$S17_FAILS")"
    fi
    rm -f "$S17_FAILS"

    # ---- production export + validation ----
    echo "[EXPORT] starting $(date +%H:%M:%S)"
    PYTHONPATH="$ROOT" python3 scripts/export_production_multimodal.py \
        --base "${BASE}" \
        --export-dir "${EXPORT_DIR}" 2>&1 | tee "logs/export_${VIDEO_ID}${SUFFIX}.log"
    EXPORT_RC=${PIPESTATUS[0]}
    [ "$EXPORT_RC" -ne 0 ] && stage_fail="export:$EXPORT_RC"

    # ---- P3: free disk — drop the big stage0_11 face-frame intermediates after export ----
    # stage0_11/faces (+ aligned_faces in dataset mode) are pure intermediates:
    # neither stage17_multimodal_export nor the face_only verification rebuild
    # (clips/ + videos/ + stage17_fusion/ + stage14_audio/ + stage_inputs/) needs them.
    #   dataset     -> drop faces + aligned_faces (both rebuildable intermediates)
    #   open_domain -> keep aligned_faces (wanted later), drop only faces
    # S12-17 failures keep everything so their back-half can be retried later.
    # NOTE: BASE already == <video_dir>/stage0_11 here.
    case "${BASE}" in
        */video_*_open_domain/stage0_11)
            if [ -z "$stage_fail" ] && [ -d "${BASE}" ]; then
                [ -d "${BASE}/faces" ] && rm -rf -- "${BASE}/faces" && echo "[clean] stage0_11/faces removed (${VIDEO_ID})"
                rm -rf -- "${BASE}/tmp" 2>/dev/null
            fi
            ;;
        */video_*/stage0_11)
            if [ -z "$stage_fail" ] && [ -d "${BASE}" ]; then
                for d in faces aligned_faces; do
                    [ -d "${BASE}/$d" ] && rm -rf -- "${BASE}/$d" && echo "[clean] stage0_11/$d removed (${VIDEO_ID})"
                done
                rm -rf -- "${BASE}/tmp" 2>/dev/null
            fi
            ;;
        *)
            echo "[clean] SKIP: BASE pattern mismatch: ${BASE}"
            ;;
    esac

    if [ -z "$stage_fail" ]; then status="PASS"; else status="FAIL($stage_fail)"; fi

    wall_end=$(date +%s)
    wall_s=$(( wall_end - wall_start ))

    # per-stage S0-11 total from preserved results JSON (sequential only)
    if [ "$PARALLEL" = "1" ] && [ -f "${OUT_ROOT_REL}/video_${VIDEO_ID}${SUFFIX}/stage0_11_results_${MODE}.json" ]; then
        s0_total=$(python3 - "${OUT_ROOT_REL}/video_${VIDEO_ID}${SUFFIX}/stage0_11_results_${MODE}.json" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    tot = sum((s.get("elapsed_s") or 0) for s in d.get("stages", []))
    print(f"{tot:.2f}")
except Exception:
    print("NA")
PYEOF
        )
    else
        s0_total="NA"
    fi

    echo "[BATCH] $VIDEO_ID -> $status (wall ${wall_s}s, s0_11 ${s0_total}s, clips=$(echo "$CLIPS" | wc -l))"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$batch_no" "$VIDEO_ID" "$MODE" "$status" "$s0_total" "$wall_s" "$(echo "$CLIPS" | wc -l)" "$BASE" "$EXPORT_DIR" "$(date +%Y-%m-%d\ %H:%M:%S)" > "$part"
    return 0
}

# =====================================================================
# run_slice: process VIDEOS[start:end) as one batch, write its own log+CSV.
#   $1 start idx, $2 end idx, $3 batch number
# =====================================================================
run_slice() {
    local start_idx="$1" end_idx="$2" k="$3"
    local slice=("${VIDEOS[@]:$start_idx:$((end_idx-start_idx))}")
    local slog="logs/batch_${TS}_b${k}.log"
    local scsv="logs/batch_summary_${TS}_b${k}.csv"

    {
        echo "======================================================================"
        echo "BATCH $k/$TOTAL_BATCHES $MODE | videos=${#slice[@]} | parallel=$PARALLEL | smoke=$SMOKE | skip-existing=$SKIP_EXISTING"
        echo "config: $CONFIG | out-root: $OUT_ROOT_REL | log: $slog | csv: $scsv"
        echo "======================================================================"
        if [ "$PARALLEL" -le 1 ]; then
            for video in "${slice[@]}"; do
                vid=$(basename "$video" .mp4)
                run_one_video "$video" "$TMPDIR/${vid}.part" "$k"
            done
        else
            pool=()
            for video in "${slice[@]}"; do
                vid=$(basename "$video" .mp4)
                run_one_video "$video" "$TMPDIR/${vid}.part" "$k" &
                pool+=("$!")
                while [ ${#pool[@]} -ge "$PARALLEL" ]; do
                    wait -n || true
                    newpool=()
                    for p in "${pool[@]}"; do kill -0 "$p" 2>/dev/null && newpool+=("$p"); done
                    pool=("${newpool[@]}")
                done
            done
            wait
        fi
        echo "======================================================================"
        echo "BATCH $k DONE $(date +%H:%M:%S) | csv: $scsv"
        echo "======================================================================"
    } | tee "$slog"

    {
        printf '%s\n' "batch,video_id,mode,status,s0_11_elapsed_s,video_wall_s,clips,base,export_dir,finished_at"
        for video in "${slice[@]}"; do
            vid=$(basename "$video" .mp4)
            [ -f "$TMPDIR/${vid}.part" ] && cat "$TMPDIR/${vid}.part"
        done
    } > "$scsv"
}

# =====================================================================
# main driver
# =====================================================================
if [ "$DRY_RUN" = "1" ]; then
    echo "BATCH $MODE | videos=${#VIDEOS[@]} | batches=$TOTAL_BATCHES x ${BATCH_SIZE}"
    echo "config: $CONFIG | out-root: $OUT_ROOT_REL"
    echo "======================================================================"
    for (( k=1; k<=TOTAL_BATCHES; k++ )); do
        s=$(( (k-1)*BATCH_SIZE )); e=$(( k*BATCH_SIZE ))
        [ "$e" -gt "${#VIDEOS[@]}" ] && e=${#VIDEOS[@]}
        echo "DRY RUN batch $k/$TOTAL_BATCHES - would process $((e-s)) video(s):"
        for (( i=s; i<e; i++ )); do
            vid=$(basename "${VIDEOS[$i]}" .mp4)
            exp="${OUT_ROOT_REL}/video_${vid}${SUFFIX}/stage17_multimodal_export"
            if [ "$SKIP_EXISTING" = "1" ] && [ -f "$exp/manifest.csv" ]; then
                echo "    ${VIDEOS[$i]} -> $exp  (SKIP: existing export)"
            else
                echo "    ${VIDEOS[$i]} -> $exp"
            fi
        done
    done
    echo "(dry-run only; nothing executed)"
    rm -rf "$TMPDIR"
    exit 0
fi

# Emit a clip-level manifest summary snapshot even if the run is
# interrupted (SIGINT/SIGTERM). Kill -9 cannot trap: run the python script
# manually in that case.  NOTE: the summary is a full rescan of every
# pipeline_manifest.json under out-root, never an incremental append.
trap 'echo; echo "[SUMMARY] interrupted at $(date +%H:%M:%S) - emitting partial snapshot"; \
      PYTHONPATH="$ROOT" python3 scripts/generate_manifest_summary.py \
          --out-root "$OUT_ROOT_REL" \
          --output "$OUT_ROOT_REL/pipeline_manifest_summary_${TS}_partial.csv" 2>&1 | tail -3; \
      exit 130' INT TERM

echo "BATCH $MODE | videos=${#VIDEOS[@]} | batches=$TOTAL_BATCHES x ${BATCH_SIZE} | out-root: $OUT_ROOT_REL" | tee "logs/batch_${TS}.log"

for (( k=1; k<=TOTAL_BATCHES; k++ )); do
    s=$(( (k-1)*BATCH_SIZE )); e=$(( k*BATCH_SIZE ))
    [ "$e" -gt "${#VIDEOS[@]}" ] && e=${#VIDEOS[@]}
    run_slice "$s" "$e" "$k"
done

# combined summary across batches
if [ "$TOTAL_BATCHES" -gt 1 ]; then
    COMB="logs/batch_summary_${TS}.csv"
    {
        printf '%s\n' "batch,video_id,mode,status,s0_11_elapsed_s,video_wall_s,clips,base,export_dir,finished_at"
        for (( k=1; k<=TOTAL_BATCHES; k++ )); do
            tail -n +2 "logs/batch_summary_${TS}_b${k}.csv"
        done
    } > "$COMB"
    echo "Combined summary: $COMB"
fi

# ---------------------------------------------------------------------
# Verification videos (video + audio merged, for manual mesh review).
# Runs for every sample root under out-root that reached Stage12-14.
# Muxes each clip's video with the SAME audio Stage 14 consumed.
# (Mirrors the [VERIFY] block in run_single_video_*.sh)
# ---------------------------------------------------------------------
echo "[VERIFY] building verification videos for out-root: $OUT_ROOT_REL"
PYTHONPATH="$ROOT" python3 scripts/build_verification_videos.py \
    --glob "$OUT_ROOT_REL/video_*" --jobs "$PARALLEL" \
    > "logs/batch_${TS}_verify.log" 2>&1
VFY_RC=$?
echo "[VERIFY] rc=$VFY_RC | $(tail -1 "logs/batch_${TS}_verify.log" 2>/dev/null)"

# ---------------------------------------------------------------------
# P3: free disk — drop the full verification/ mp4s (video+audio) after they
# are built.  Only verification_face_only/ is kept, because consolidate copies
# its *_faces_only.mp4 into <out-root>_new_pipeline/videos.
# ---------------------------------------------------------------------
echo "[CLEAN] removing transient verification/ folders (keeping verification_face_only/)"
find "$OUT_ROOT_REL" -mindepth 2 -maxdepth 2 -type d -name verification -exec rm -rf -- {} +
echo "[CLEAN] verification/ removal done"

# ---------------------------------------------------------------------
# Verification face-only videos (face-only frames + audio merged, for review).
# The no-audio face-only stream is generated transiently inside the script and
# deleted after mux; only the final muxed *_faces_only.mp4 is persisted under
# verification_face_only/.
# ---------------------------------------------------------------------
echo "[VERIFY_FACE_ONLY] building verification face-only videos for out-root: $OUT_ROOT_REL"
PYTHONPATH="$ROOT" python3 scripts/build_verification_face_only_videos.py \
    --glob "$OUT_ROOT_REL/video_*" --jobs "$PARALLEL" \
    > "logs/batch_${TS}_verify_face_only.log" 2>&1
VFO_RC=$?
echo "[VERIFY_FACE_ONLY] rc=$VFO_RC | $(tail -1 "logs/batch_${TS}_verify_face_only.log" 2>/dev/null)"

# ---------------------------------------------------------------------
# Consolidate outputs (audio npy + flame npy + verification_face_only mp4) into
# <out-root>/<out-root-name>_new_pipeline/{audio,flame,videos}
# ---------------------------------------------------------------------
echo "[CONSOLIDATE] consolidating outputs into $OUT_ROOT_REL/${OUT_ROOT_REL##*/}_new_pipeline"
bash scripts/consolidate_batch_outputs.sh "$OUT_ROOT_REL" \
    > "logs/batch_${TS}_consolidate.log" 2>&1
CS_RC=$?
echo "[CONSOLIDATE] rc=$CS_RC | $(tail -1 "logs/batch_${TS}_consolidate.log" 2>/dev/null)"

# ---------------------------------------------------------------------
# Clip-level flat CSV from every pipeline_manifest.json under out-root.
# This is a full rescan (a complete snapshot of whatever manifests exist
# right now), so resuming a partial run and re-running the batch still
# captures new rows automatically -- it just writes a *new* file rather
# than appending to an old one.
# ---------------------------------------------------------------------
echo "[SUMMARY] generating clip-level manifest summary for out-root: $OUT_ROOT_REL"
PYTHONPATH="$ROOT" python3 scripts/generate_manifest_summary.py \
    --out-root "$OUT_ROOT_REL" \
    --output "$OUT_ROOT_REL/pipeline_manifest_summary_${TS}.csv" \
    > "logs/batch_${TS}_summary.log" 2>&1
SUM_RC=$?
echo "[SUMMARY] rc=$SUM_RC | $(tail -1 "logs/batch_${TS}_summary.log" 2>/dev/null)"

rm -rf "$TMPDIR"
echo "ALL BATCHES DONE $(date +%H:%M:%S)"
exit 0
