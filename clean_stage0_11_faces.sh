#!/usr/bin/env bash
#
# clean_stage0_11_faces.sh
#
# 功能：
#   递归查找 BASE_DIR 下所有的 stage0_11 目录，
#   删除其中的 aligned_faces 和 faces 子目录，其余内容一律不动。
#
# 用法：
#   ./clean_stage0_11_faces.sh [选项] [基础目录]
#
# 选项：
#   --dry-run      只列出将要删除的路径，不真正删除（预览模式）
#   -h, --help     打印帮助
#   [基础目录]      可选，临时指定要清理的根目录（默认用下面的 BASE_DIR）
#
# 说明：
#   以后要清理别的路径，只需修改下面的 BASE_DIR，或运行时传入新的目录参数即可，
#   无需改动脚本主体逻辑。默认 BASE_DIR 指向批量输出目录，也可以直接传某个
#   单独的 video 文件夹来做单文件夹测试（脚本会递归找到其中的 stage0_11）。

set -u

# ===================== 可配置项 =====================

# 要清理的根目录默认列表（可被命令行目录参数覆盖）
# 默认指向本次需要清理的 5 个 out-root；运行时也可用目录参数临时覆盖。
BASE_DIRS=(
    "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/flashtalk_batch_010000"
    "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/flashtalk_batch_020000"
    "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/flashtalk_batch_030000"
    "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/flashtalk_batch_040000"
    "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/vivihead_Chinese_videos_all"
)

# stage 目录名（一般不需要改）
STAGE_DIR="stage0_11"

# 需要删除的子目录名（可增删，空格分隔，数组形式）
TARGETS=(aligned_faces faces)

# 默认是否预览：false=真正删除；true=只预览不删除（等价于 --dry-run）
DRY_RUN=false

# ====================================================

usage() {
    cat <<EOF
用法: $0 [--dry-run] [基础目录 ...]
  递归删除每个 BASE_DIR 下每个 ${STAGE_DIR} 目录里的: ${TARGETS[*]}
  默认 BASE_DIRS=${BASE_DIRS[*]}

选项:
  --dry-run    只列出将删除的路径，不真正删除
  -h, --help   打印本帮助
EOF
    exit 0
}

# ------------- 解析参数 -------------
BASE_DIRS_OVERRIDE=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage ;;
        *) BASE_DIRS_OVERRIDE+=("$arg") ;;
    esac
done
if [ ${#BASE_DIRS_OVERRIDE[@]} -gt 0 ]; then
    BASE_DIRS=("${BASE_DIRS_OVERRIDE[@]}")
fi

# ------------- 主体 -------------
found=0     # 找到的目标路径数量（含真实删除与预览）
deleted=0   # 真正删除的数量
missing=0   # 本就不存在的目标数量

for BASE_DIR in "${BASE_DIRS[@]}"; do
    if [ ! -d "$BASE_DIR" ]; then
        echo "[错误] 目录不存在: $BASE_DIR（跳过）" >&2
        continue
    fi
    echo "[清理] 处理目录: $BASE_DIR"

    # 递归查找所有 STAGE_DIR 目录（-print0 配合 read -d '' 可安全处理特殊字符路径）
    while IFS= read -r -d '' stage; do
        for t in "${TARGETS[@]}"; do
            target="$stage/$t"
            if [ -e "$target" ]; then
                found=$((found + 1))
                if [ "$DRY_RUN" = true ]; then
                    echo "[预览] 将删除: $target"
                else
                    rm -rf -- "$target"
                    if [ -e "$target" ]; then
                        echo "[失败] 未能删除: $target" >&2
                    else
                        echo "[已删除] $target"
                        deleted=$((deleted + 1))
                    fi
                fi
            else
                missing=$((missing + 1))
            fi
        done
    done < <(find "$BASE_DIR" -type d -name "$STAGE_DIR" -print0)
done

# ------------- 汇总 -------------
echo
if [ "$DRY_RUN" = true ]; then
    echo "预览结束：将删除 ${found} 个目标目录（未做任何删除）。"
    echo "去掉 --dry-run 参数即可真正执行。"
    if [ "$missing" -gt 0 ]; then
        echo "提示：另有 ${missing} 个目标本就不存在，无需处理。"
    fi
else
    echo "完成：共删除 ${deleted} 个目标目录。"
    if [ "$missing" -gt 0 ]; then
        echo "提示：有 ${missing} 个目标路径本就不存在，已跳过（未删除）。"
    fi
fi

exit 0