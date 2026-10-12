#!/usr/bin/env python3
"""Stage-17 之后的后处理：抽取"有脸帧"生成无音频 faces_only 视频（方案 A）。

背景
----
Stage 9.5 导出的 verification 视频 = clips/{clip_id}.mp4，包含该 clip 的**全部帧**
（含无人脸帧）。而 Stage 17 的最终参数 `training_sample.npz` 里 `T = 有 FLAME 的帧数`
（即只有有脸帧）。两者帧数对不上，给渲染同事的视频会多出一堆没有参数的帧。

本脚本按 `training_sample.npz` 里的 `frame_index`（有脸帧的原始绝对帧号），
从源视频里只抽这些帧，重编码成一个新的、无音频的 mp4，使
    faces_only.mp4 帧数与 flame/audio 参数帧数严格一一对应。

坐标约定
--------
`frame_index` 是 0-based 的**绝对帧号**（相对传入的源视频逐帧读取顺序号）。
因此源视频应传 Stage 1 标准化后的 std 视频（与 frame_index 同一坐标系）。
若传的是 Stage 9.5 的 clip mp4（clip 局部 0-based），请配合 `--frame_offset`
减去 clip 的 start_frame。

音频
----
本脚本只处理视频（-an），不碰音频。音频建议走"最终渲染时用原视频 + 映射表贴回"，
或用别的音频重对齐步骤单独处理。

用法示例
--------
本脚本只产出无音频的 faces_only 中间视频（-an），属临时产物、用完即删：
``build_verification_face_only_videos.py`` 会在 mux 时自行临场生成并删除它，
无需（也不应）把该中间 mp4 落盘到 verification_face_only/。下面示例直接写到系统临时目录：
python scripts/export_faces_only_video.py \\
    --npz outputs/.../stage17_fusion/{clip_id}/training_sample.npz \\
    --video outputs/.../videos/{video_id}/{video_id}_std.mp4 \\
    --output /tmp/{clip_id}_faces_only.mp4 \\
    [--fps 25]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def load_keep_frames(npz_path: Path):
    """读回 frame_index / timestamps / dropout_mask，并按 frame_index 升序。

    Returns:
        frame_index: (N,) int64 升序（含空隙 = 被跳过的无脸帧）
        info: {frame_index: (timestamp, dropout_mask)}
    """
    with np.load(npz_path, allow_pickle=False) as data:
        frame_index = data["frame_index"].astype(np.int64)
        timestamps = (
            data["timestamps"].astype(np.float64)
            if "timestamps" in data
            else np.zeros_like(frame_index, dtype=np.float64)
        )
        dropout = (
            data["dropout_mask"].astype(bool)
            if "dropout_mask" in data
            else np.zeros(len(frame_index), dtype=bool)
        )

    order = np.argsort(frame_index, kind="stable")
    frame_index = frame_index[order]
    timestamps = timestamps[order]
    dropout = dropout[order]

    info = {int(fi): (float(ts), bool(dp)) for fi, ts, dp in zip(frame_index, timestamps, dropout)}
    return frame_index, info


def infer_fps(npz_path: Path, fallback: float) -> float:
    """优先读同目录 metadata.json 的 fps，否则用 --fps / 默认 25。"""
    meta = npz_path.parent / "metadata.json"
    if meta.exists():
        try:
            with open(meta, encoding="utf-8") as fh:
                fps = json.load(fh).get("fps")
            if fps:
                return float(fps)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return fallback


def build_cmd(width: int, height: int, fps: float, out_path: Path) -> list:
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", f"{fps}",
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", f"{fps}",
        str(out_path),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="抽出有脸帧，生成无音频 faces_only 视频")
    ap.add_argument("--npz", required=True, help="training_sample.npz 路径")
    ap.add_argument("--video", required=True, help="源视频（std 视频，绝对帧号坐标系）")
    ap.add_argument("--output", required=True, help="输出 mp4 路径")
    ap.add_argument("--fps", type=float, default=25.0, help="输出帧率（默认 25，可被 metadata.json 覆盖）")
    ap.add_argument("--frame_offset", type=int, default=0,
                    help="若源视频是 clip mp4，则实际帧号 = frame_index - frame_offset")
    ap.add_argument("--csv", default=None, help="frame_map.csv 输出路径（默认与视频同名 .csv）")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    video_path = Path(args.video)
    out_path = Path(args.output)
    if not npz_path.exists():
        print(f"[error] npz 不存在: {npz_path}", file=sys.stderr)
        return 1
    if not video_path.exists():
        print(f"[error] 视频不存在: {video_path}", file=sys.stderr)
        return 1

    fps = infer_fps(npz_path, args.fps)
    frame_index, info = load_keep_frames(npz_path)
    if len(frame_index) == 0:
        print("[error] frame_index 为空，无帧可抽", file=sys.stderr)
        return 1

    # 应用 frame_offset 后的目标帧集合
    keep = sorted({int(fi) - args.frame_offset for fi in frame_index})
    keep = [k for k in keep if k >= 0]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] 无法打开视频: {video_path}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(build_cmd(width, height, fps, out_path), stdin=subprocess.PIPE)

    rows: list[tuple[int, int, float, int]] = []  # (order, frame_index, timestamp, dropout)
    keep_ptr = 0
    idx = 0  # 0-based 源视频帧号
    while keep_ptr < len(keep):
        ret, frame = cap.read()
        if not ret:
            break
        target = keep[keep_ptr]
        if idx == target:
            proc.stdin.write(frame.tobytes())
            fi = target + args.frame_offset
            ts, dp = info.get(fi, (0.0, False))
            rows.append((len(rows), fi, ts, int(dp)))
            keep_ptr += 1
        elif idx > target:
            # 目标帧号早于当前读取（源视频缺帧），跳过该目标
            keep_ptr += 1
            continue
        idx += 1

    cap.release()
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        print("[error] ffmpeg 编码失败", file=sys.stderr)
        return 1

    csv_path = Path(args.csv) if args.csv else out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "frame_index", "timestamp", "dropout_mask"])
        w.writerows(rows)

    dropped = total_src - len(rows)
    print(f"[done] 输出: {out_path}")
    print(f"[done] 映射表: {csv_path}")
    print(f"[stat] 源视频帧数 = {total_src}")
    print(f"[stat] 有脸帧(导出) = {len(rows)}")
    print(f"[stat] 无脸帧(丢弃) = {max(dropped, 0)}")
    print(f"[stat] 帧率 = {fps} fps, 输出时长 ≈ {len(rows) / fps:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())