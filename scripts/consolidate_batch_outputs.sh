#!/usr/bin/env bash
# 整合分散的 pipeline 输出：audio npy / flame npy / mp4 视频。
#
# 规则：
#   以批次目录下每个子文件夹（一个视频）为单位做「一一对应」匹配；
#   只有同时满足下面三者，该视频才会被整合：
#     - stage17_multimodal_export/audio/*.npy  存在（非空）
#     - stage17_multimodal_export/flame/*.npy  存在（非空）
#     - verification_face_only/<clip_id>_faces_only.mp4  存在（非空，face-only 视频+音频合并，不含 *_original.mp4）
#   三者缺一不可，缺任意一类就整组跳过。
#   一个视频若有多个 audio/flame npy，全部保留；mp4 也全部保留。
#   使用复制(cp)，不删除原始分散文件。
#
# 用法：
#   bash consolidate_batch_outputs.sh <批次目录路径>
# 输出：
#   <批次目录>/<批次名>_new_pipeline/{audio,flame,videos}
#   （批次名由目录名自动推导，vivihead_batch_881、HDTFxxxx 等任意名称都适用）
set -u

usage() {
    cat <<EOF
整合分散的 pipeline 输出：audio npy / flame npy / mp4 视频。

用法:
  bash $(basename "$0") <批次目录路径>

说明:
  - 以批次目录下每个子文件夹(视频)为单位做「一一对应」匹配。
  - 一个视频必须同时有 audio npy、flame npy、mp4 三类文件，缺任意一类就整组跳过。
  - 视频 mp4 取自 verification_face_only/（face-only 视频+音频合并结果，不含 *_original.mp4）。
  - 一个视频若有多个 audio/flame npy，全部保留。
  - 使用复制(cp)，不删除原始分散文件。
  - 输出目录: <批次目录>/<批次名>_new_pipeline/{audio,flame,videos}

示例:
  bash $(basename "$0") /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/vivihead_batch_283_videos
  bash $(basename "$0") /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/vivihead_batch_881
  bash $(basename "$0") /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/HDTFxxxx
EOF
}

[ "$#" -ge 1 ] || { usage; exit 1; }
case "$1" in
  -h|--help) usage; exit 0 ;;
esac

BATCH_DIR="$1"
BATCH_DIR="$(cd "$BATCH_DIR" 2>/dev/null && pwd || echo "$BATCH_DIR")"
BATCH_DIR="${BATCH_DIR%/}"
BATCH_NAME="$(basename "$BATCH_DIR")"

if [ ! -d "$BATCH_DIR" ]; then
  echo "错误: 批次目录不存在: $BATCH_DIR" >&2
  exit 1
fi

OUT_DIR="$BATCH_DIR/${BATCH_NAME}_new_pipeline"
mkdir -p "$OUT_DIR/audio" "$OUT_DIR/flame" "$OUT_DIR/videos"
LOG_FILE="$OUT_DIR/consolidate.log"
: > "$LOG_FILE"

log() { printf '%s\n' "$*" | tee -a "$LOG_FILE"; }

# 判断目录下是否存在指定类型的文件（$1=目录 $2=文件名模式，$3...=额外 find 谓词）
has_files() {
  local dir="$1" pat="$2"; shift 2
  [ -d "$dir" ] || return 1
  find "$dir" -type f -name "$pat" "$@" -print -quit | grep -q .
}

# 复制单个文件到目标目录；返回: 0=已复制 2=已存在跳过 1=失败
copy_file() {  # $1=源文件 $2=目标目录
  local base dst
  base="$(basename "$1")"
  dst="$2/$base"
  if [ -e "$dst" ]; then
    return 2
  fi
  if cp "$1" "$2/"; then
    return 0
  fi
  return 1
}

total=0
valid=0
skip_audio=0
skip_flame=0
skip_video=0
audio_copied=0
flame_copied=0
video_copied=0
exist_skipped=0
fail=0

for d in "$BATCH_DIR"/*/; do
  [ -d "$d" ] || continue
  d="${d%/}"
  name="$(basename "$d")"
  case "$name" in
    .*|*_new_pipeline*) continue ;;   # 跳过隐藏目录和输出目录
  esac

  total=$((total+1))

  audio_dir="$d/stage17_multimodal_export/audio"
  flame_dir="$d/stage17_multimodal_export/flame"
  video_dir="$d/verification_face_only"

  has_audio=0; has_flame=0; has_video=0
  has_files "$audio_dir" '*.npy' && has_audio=1
  has_files "$flame_dir" '*.npy' && has_flame=1
  has_files "$video_dir" '*.mp4' -not -name '*_original.mp4' && has_video=1

  if [ "$has_audio" -eq 0 ] || [ "$has_flame" -eq 0 ] || [ "$has_video" -eq 0 ]; then
    reason=""
    if [ "$has_audio" -eq 0 ]; then reason="缺audio"; skip_audio=$((skip_audio+1)); fi
    if [ "$has_flame" -eq 0 ]; then reason="${reason:+$reason,}缺flame"; skip_flame=$((skip_flame+1)); fi
    if [ "$has_video" -eq 0 ]; then reason="${reason:+$reason,}缺video"; skip_video=$((skip_video+1)); fi
    log "SKIP  $name  [$reason]"
    continue
  fi

  valid=$((valid+1))
  log "OK    $name"

  while IFS= read -r -d '' f; do
    copy_file "$f" "$OUT_DIR/audio"
    rc=$?
    case $rc in
      0) audio_copied=$((audio_copied+1)) ;;
      2) exist_skipped=$((exist_skipped+1)) ;;
      *) log "FAIL  audio: $f"; fail=$((fail+1)) ;;
    esac
  done < <(find "$audio_dir" -type f -name '*.npy' -print0)

  while IFS= read -r -d '' f; do
    copy_file "$f" "$OUT_DIR/flame"
    rc=$?
    case $rc in
      0) flame_copied=$((flame_copied+1)) ;;
      2) exist_skipped=$((exist_skipped+1)) ;;
      *) log "FAIL  flame: $f"; fail=$((fail+1)) ;;
    esac
  done < <(find "$flame_dir" -type f -name '*.npy' -print0)

  while IFS= read -r -d '' f; do
    copy_file "$f" "$OUT_DIR/videos"
    rc=$?
    case $rc in
      0) video_copied=$((video_copied+1)) ;;
      2) exist_skipped=$((exist_skipped+1)) ;;
      *) log "FAIL  video: $f"; fail=$((fail+1)) ;;
    esac
  done < <(find "$video_dir" -maxdepth 1 -type f -name '*_faces_only.mp4' -print0)
done

log ""
log "================ 汇总 ================"
log "批次目录 : $BATCH_DIR"
log "输出目录 : $OUT_DIR"
log "扫描子目录: $total"
log "整合(有效): $valid"
log "跳过: 缺audio=$skip_audio, 缺flame=$skip_flame, 缺video=$skip_video"
log "复制文件: audio=$audio_copied, flame=$flame_copied, videos=$video_copied"
log "已存在跳过: $exist_skipped"
log "失败: $fail"
