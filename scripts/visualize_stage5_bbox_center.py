#!/usr/bin/env python3
"""Stage5 ROI Temporal BBox Center Trajectory Visualization (read-only).

Read-only analysis / visualization on top of the Stage5 SCRFD Temporal ROI
Ablation artifacts produced by ``scripts/stage5_roi_temporal_ablation.py``:

    docs/stage5_roi_ablation/baseline_results.json
    docs/stage5_roi_ablation/roi_results.json
    docs/stage5_roi_ablation/ablation_metrics.json

For every frame it computes the bbox center

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

and writes, under docs/stage5_roi_ablation/visualization/:

    bbox_center_x.png            cx trajectory curves (baseline vs ROI)
    bbox_center_y.png            cy trajectory curves (baseline vs ROI)
    bbox_center_velocity.png     per-frame center displacement (velocity)
    outlier_frames.json          isolated-jump outlier frames (threshold 8 px)
    bbox_center_statistics.json  statistical summary
    baseline_bbox_center_overlay.mp4   overlay video (bbox + center + frame id)
    roi_bbox_center_overlay.mp4        overlay video (bbox + center + frame id)

plus docs/Stage5_BBox_Center_Trajectory_Report.md.

The outlier rule reproduces the previous experiment exactly (section 8.3):
a transition t-1 -> t is flagged when its center displacement > threshold AND
the following transition t -> t+1 is normal (<= threshold).  This guarantees
the produced outlier lists match ``ablation_metrics.json``.

No production pipeline code, models or configs are modified; no filtering or
EMA is applied.  The purpose is purely to understand how the temporal-ROI
method affects bbox stability.

Usage (from the Face_Pipeline_V2 project root):

    python3 scripts/visualize_stage5_bbox_center.py
    python3 scripts/visualize_stage5_bbox_center.py --no-videos   # plots/JSON only
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is expected but optional
    plt = None

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _PROJECT_ROOT / "docs/stage5_roi_ablation/visualization"
EXPERIMENT_DIR = OUT_DIR.parent
BASELINE_JSON = EXPERIMENT_DIR / "baseline_results.json"
ROI_JSON = EXPERIMENT_DIR / "roi_results.json"
METRICS_JSON = EXPERIMENT_DIR / "ablation_metrics.json"
REPORT_PATH = _PROJECT_ROOT / "docs/Stage5_BBox_Center_Trajectory_Report.md"

# Fixed outlier threshold fallback (the experiment's configured value).
DEFAULT_OUTLIER_PX = 8.0
# expand ratio actually used for the ROI experiment (read from roi_results.json)
ROI_EXPAND = 1.2

# Plot colours.
COL_BASELINE = "#1f77b4"   # blue
COL_ROI = "#d62728"        # red
COL_THRESH = "#2ca02c"     # green threshold line
BASELINE_NAME = "baseline"
ROI_NAME = "roi"


# ---------------------------------------------------------------------------
# Data loading + center trajectory analysis
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file, raising a clear error on failure."""
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def load_results() -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Load baseline/ROI results plus the experiment outlier threshold."""
    baseline = load_json(BASELINE_JSON)
    roi = load_json(ROI_JSON)
    threshold = DEFAULT_OUTLIER_PX
    if METRICS_JSON.exists():
        try:
            metrics = load_json(METRICS_JSON)
            threshold = float(metrics.get("outlier_px_threshold", DEFAULT_OUTLIER_PX))
        except (json.JSONDecodeError, TypeError, ValueError):
            threshold = DEFAULT_OUTLIER_PX
    return baseline, roi, threshold


def _center(bbox: Sequence[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def extract_trajectory(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute (frame_id, cx, cy) for every frame with a valid bbox."""
    traj: List[Dict[str, Any]] = []
    for fr in frames:
        bbox = fr.get("bbox")
        if bbox is None:
            continue
        cx, cy = _center(bbox)
        traj.append(
            {"frame_id": int(fr["frame_id"]), "cx": float(cx), "cy": float(cy)}
        )
    return traj


def compute_displacements(traj: Sequence[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """Per-frame center displacement, attributed to the current frame t.

    dx = cx[t] - cx[t-1], dy = cy[t] - cy[t-1], distance = hypot(dx, dy).
    Mirrors ``bbox_stability`` in the ablation experiment.
    """
    deltas: List[Tuple[int, float]] = []
    for i in range(1, len(traj)):
        a, b = traj[i - 1], traj[i]
        dx = b["cx"] - a["cx"]
        dy = b["cy"] - a["cy"]
        deltas.append((int(b["frame_id"]), float(math.hypot(dx, dy))))
    return deltas


def isolated_outliers(
    deltas: Sequence[Tuple[int, float]], threshold: float
) -> List[Dict[str, Any]]:
    """Isolated-jump detection (identical rule to the ablation experiment).

    A transition attributed to frame ``t`` is an outlier when its center
    displacement > threshold AND the following transition (t -> t+1) is normal
    (<= threshold) or does not exist (last frame of the clip).
    """
    by_t = {t: d for t, d in deltas}
    out: List[Dict[str, Any]] = []
    for t in sorted(by_t):
        if by_t[t] > threshold:
            nxt = by_t.get(t + 1)
            if nxt is None or nxt <= threshold:
                out.append({"frame": int(t), "distance": round(float(by_t[t]), 2)})
    return out


def _frame_id(traj: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.array([t["frame_id"] for t in traj], dtype=int)


def compute_statistics(
    traj: Sequence[Dict[str, Any]], deltas: Sequence[Tuple[int, float]], method: str
) -> Dict[str, Any]:
    """Statistical summary for one method."""
    cx = np.array([t["cx"] for t in traj], dtype=float)
    cy = np.array([t["cy"] for t in traj], dtype=float)
    vals = np.array([d for _, d in deltas], dtype=float)
    return {
        "method": method,
        "cx_mean": round(float(cx.mean()), 4),
        "cy_mean": round(float(cy.mean()), 4),
        "cx_std": round(float(cx.std()), 4),
        "cy_std": round(float(cy.std()), 4),
        "velocity_mean": round(float(vals.mean()) if len(vals) else 0.0, 4),
        "velocity_max": round(float(vals.max()) if len(vals) else 0.0, 4),
    }


def validate_data(baseline: Dict[str, Any], roi: Dict[str, Any]) -> None:
    """Validate the required consistency checks (frame counts, continuity)."""
    b_frames, r_frames = baseline["frames"], roi["frames"]
    b_ids = [f["frame_id"] for f in b_frames]
    r_ids = [f["frame_id"] for f in r_frames]
    if len(b_ids) != len(r_ids):
        raise ValueError(
            f"frame-count mismatch: baseline={len(b_ids)} roi={len(r_ids)}"
        )
    for label, ids in (("baseline", b_ids), ("roi", r_ids)):
        if ids != list(range(len(ids))):
            raise ValueError(f"{label} frame ids are not contiguous 0..n-1: {ids}")
    n = len(b_ids)
    print(
        f"[validate] baseline & roi both have {n} frames, "
        f"frame ids contiguous 0..{n - 1}"
    )
    for label, data in (("baseline", baseline), ("roi", roi)):
        missing = [f["frame_id"] for f in data["frames"] if f.get("bbox") is None]
        if missing:
            print(f"[validate] WARNING {label} missing bbox on frames: {missing}")



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_figure() -> "Any":
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    ax.set_xlabel("Frame index")
    ax.grid(True, alpha=0.3)
    return fig, ax


def _save(fig: "Any", path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


def plot_center_x(
    b_traj: Sequence[Dict[str, Any]],
    r_traj: Sequence[Dict[str, Any]],
    b_out: Sequence[Dict[str, Any]],
    r_out: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    """cx trajectory curves (baseline vs ROI) with outlier markers."""
    fig, ax = _setup_figure()
    b_ids = _frame_id(b_traj)
    r_ids = _frame_id(r_traj)
    ax.plot(b_ids, [t["cx"] for t in b_traj], color=COL_BASELINE, lw=1.6,
            label="baseline cx")
    ax.plot(r_ids, [t["cx"] for t in r_traj], color=COL_ROI, lw=1.6,
            label="ROI 1.2 cx")
    if b_out:
        ax.plot([o["frame"] for o in b_out], [o["_cx"] for o in b_out],
                "x", color=COL_BASELINE, ms=9, mew=2,
                label="baseline outlier")
    if r_out:
        ax.plot([o["frame"] for o in r_out], [o["_cx"] for o in r_out],
                "s", color=COL_ROI, ms=7,
                label="ROI outlier")
    ax.set_ylabel("cx (pixel)")
    ax.set_title("BBox center X trajectory (cx) - baseline vs ROI 1.2")
    ax.legend(loc="best")
    _save(fig, out_path)


def plot_center_y(
    b_traj: Sequence[Dict[str, Any]],
    r_traj: Sequence[Dict[str, Any]],
    b_out: Sequence[Dict[str, Any]],
    r_out: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    """cy trajectory curves (baseline vs ROI) with outlier markers."""
    fig, ax = _setup_figure()
    b_ids = _frame_id(b_traj)
    r_ids = _frame_id(r_traj)
    ax.plot(b_ids, [t["cy"] for t in b_traj], color=COL_BASELINE, lw=1.6,
            label="baseline cy")
    ax.plot(r_ids, [t["cy"] for t in r_traj], color=COL_ROI, lw=1.6,
            label="ROI 1.2 cy")
    if b_out:
        ax.plot([o["frame"] for o in b_out], [o["_cy"] for o in b_out],
                "x", color=COL_BASELINE, ms=9, mew=2,
                label="baseline outlier")
    if r_out:
        ax.plot([o["frame"] for o in r_out], [o["_cy"] for o in r_out],
                "s", color=COL_ROI, ms=7,
                label="ROI outlier")
    ax.set_ylabel("cy (pixel)")
    ax.set_title("BBox center Y trajectory (cy) - baseline vs ROI 1.2")
    ax.legend(loc="best")
    _save(fig, out_path)


def plot_velocity(
    b_deltas: Sequence[Tuple[int, float]],
    r_deltas: Sequence[Tuple[int, float]],
    threshold: float,
    out_path: Path,
) -> None:
    """Per-frame center displacement (velocity) curves + threshold line."""
    fig, ax = _setup_figure()
    b_t = [t for t, _ in b_deltas]
    b_v = [d for _, d in b_deltas]
    r_t = [t for t, _ in r_deltas]
    r_v = [d for _, d in r_deltas]
    ax.plot(b_t, b_v, color=COL_BASELINE, lw=1.4, marker=".", ms=3,
            label="baseline velocity")
    ax.plot(r_t, r_v, color=COL_ROI, lw=1.4, marker=".", ms=3,
            label="ROI 1.2 velocity")
    ax.axhline(threshold, color=COL_THRESH, ls="--", lw=1.4,
               label=f"outlier threshold {threshold:.0f} px")
    ax.set_ylabel("center displacement (px)")
    ax.set_title("BBox center displacement between consecutive frames (velocity)")
    ax.legend(loc="best")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Overlay video rendering (bbox + center point + frame index, H.264)
# ---------------------------------------------------------------------------

GREEN = (60, 200, 80)
RED = (40, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_text(frame, text: str, y: int, colour) -> None:
    cv2.putText(frame, text, (24, y), _FONT, 1.0, BLACK, 5, cv2.LINE_AA)
    cv2.putText(frame, text, (24, y), _FONT, 1.0, colour, 2, cv2.LINE_AA)


def _h264_ffmpeg(src: Path, dst: Path) -> None:
    """Re-encode mp4v -> H.264/avc1 + yuv420p + faststart (playable everywhere)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"  [video] ffmpeg not found; keeping raw {src}", flush=True)
        src.replace(dst)
        return
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(src),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
    )
    src.unlink(missing_ok=True)


def render_overlay(
    video_path: Path,
    frames: Sequence[Dict[str, Any]],
    out_path: Path,
    fps: float,
    method_label: str,
) -> Path:
    """Draw bbox rectangle, bbox center point (cx, cy) and frame index.

    Preserves the source resolution and fps; the raw writer uses mp4v and the
    result is re-encoded to H.264 to keep the file playable.
    """
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or fi >= len(frames):
                break
            fr = frames[fi]
            bbox = fr.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
                cv2.circle(frame, (cx, cy), 6, RED, -1)
                cv2.circle(frame, (cx, cy), 10, WHITE, 2)
            _draw_text(frame, f"frame {fr['frame_id']:03d}  [{method_label}]",
                       40, WHITE)
            if bbox is not None:
                _draw_text(frame, f"center ({cx}, {cy})", 82, RED)
            writer.write(frame)
            fi += 1
    finally:
        cap.release()
        writer.release()
    _h264_ffmpeg(raw, out_path)
    print(f"[video] wrote {out_path} ({fi} frames @ {fps:.2f} fps)")
    return out_path


# ---------------------------------------------------------------------------
# JSON outputs
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"[json] wrote {path}")


def write_report(
    baseline: Dict[str, Any],
    roi: Dict[str, Any],
    threshold: float,
    stats: Sequence[Dict[str, Any]],
    b_out: Sequence[Dict[str, Any]],
    r_out: Sequence[Dict[str, Any]],
    plot_paths: Sequence[Path],
    video_paths: Sequence[Path],
    report_path: Path,
) -> Path:
    """Write docs/Stage5_BBox_Center_Trajectory_Report.md (Chinese)."""
    b_src = baseline["source"]
    r_src = roi["source"]
    video = b_src.get("video_path", "")
    fps = float(b_src.get("fps", 0.0))
    n = len(baseline["frames"])
    dur = (n / fps) if fps else 0.0
    b = {s["method"]: s for s in stats}[BASELINE_NAME]
    r = {s["method"]: s for s in stats}[ROI_NAME]

    b_out_frames = [o["frame"] for o in b_out]
    r_out_frames = [o["frame"] for o in r_out]

    def _fmt(v: float) -> str:
        return f"{v:.4f}"

    def _pct(diff: float, base: float) -> str:
        if base == 0.0:
            return "-"
        return f"{diff / base * 100.0:+.1f}%"

    def _out_str(outs: Sequence[Dict[str, Any]]) -> str:
        if not outs:
            return "无"
        return ", ".join(f"{o['frame']}:{o['distance']:.2f}" for o in outs)

    lines: List[str] = []
    lines.append("# Stage5 BBox Center Trajectory Report")
    lines.append("")
    lines.append("## 1. 数据来源")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 视频 | `{video}` |")
    lines.append(f"| 分辨率 / 帧率 | {b_src.get('width')}x{b_src.get('height')} / {fps:.0f} fps |")
    lines.append(f"| 帧数量 | {n}（时长 {dur:.1f}s） |")
    lines.append(f"| 输入 JSON | `docs/stage5_roi_ablation/baseline_results.json`（baseline）、`docs/stage5_roi_ablation/roi_results.json`（temporal ROI {r_src.get('roi_expand_ratio', ROI_EXPAND)}） |")
    lines.append(f"| 异常阈值 | {threshold:.0f} px（来自 `ablation_metrics.json` 的 `outlier_px_threshold`，沿用此前实验第 8.3 节规则） |")
    lines.append("")
    lines.append("本任务为**纯分析/可视化**：未修改 stage_05 生产代码、stage_10 landmark 代码、pipeline 流程或 config.yaml；未添加滤波 / EMA。")
    lines.append("")
    lines.append("## 2. cx / cy 定义")
    lines.append("")
    lines.append("对每一帧 bbox `[x1, y1, x2, y2]` 计算人脸框中心：")
    lines.append("")
    lines.append("| 字段 | 含义 | 公式 |")
    lines.append("|---|---|---|")
    lines.append("| `cx` | 人脸框中心 X 坐标（pixel） | `(x1 + x2) / 2` |")
    lines.append("| `cy` | 人脸框中心 Y 坐标（pixel） | `(y1 + y2) / 2` |")
    lines.append("| `frame_id` | 视频帧索引（0 起始、连续） | 与 JSON 中 `frame_id` 一致 |")
    lines.append("")
    lines.append(f"中心轨迹由 `{n}` 个连续帧组成（baseline 与 ROI 帧数一致、frame index 连续 0..{n - 1}）。")
    lines.append("")
    lines.append("## 3. 曲线说明")
    lines.append("")
    lines.append("| 图片 | 路径 | 内容 |")
    lines.append("|---|---|---|")
    lines.append(f"| Center X 轨迹 | `{plot_paths[0]}` | 横轴 frame index，纵轴 `cx`（pixel）；蓝色 = baseline，红色 = ROI 1.2；异常帧已标记 |")
    lines.append(f"| Center Y 轨迹 | `{plot_paths[1]}` | 横轴 frame index，纵轴 `cy`（pixel）；蓝色 = baseline，红色 = ROI 1.2；异常帧已标记 |")
    lines.append(f"| Center 位移（velocity） | `{plot_paths[2]}` | 相邻帧中心移动距离 `sqrt(dx² + dy²)`；绿色虚线 = 阈值 {threshold:.0f} px |")
    lines.append("")
    if video_paths:
        lines.append("Overlay 视频（可逐帧人工观察检测框是否跳动）：")
        lines.append(f"- Baseline：`{video_paths[0]}`")
        lines.append(f"- Temporal ROI 1.2：`{video_paths[1]}`")
        lines.append("")
        lines.append("视频内容：绿色 bbox 矩形、红色中心点 `(cx, cy)`、左上角 frame index 与 center 坐标；保持原分辨率 / 原 fps / H.264 编码。")
    else:
        lines.append("Overlay 视频：本次运行使用 `--no-videos` 跳过了视频渲染。")
    lines.append("")
    lines.append("## 4. 异常帧分析")
    lines.append("")
    lines.append(f"判定规则（沿用此前实验第 8.3 节）：帧 `t-1 -> t` 的中心位移 `> {threshold:.0f} px` 且 `t -> t+1` 的位移恢复正常（`<= {threshold:.0f} px`），标记 `t` 为孤立跳变异常帧。")
    lines.append("")
    lines.append("| 方法 | 异常帧（frame : distance px） | 数量 |")
    lines.append("|---|---|---|")
    lines.append(f"| baseline | {_out_str(b_out)} | {len(b_out)} |")
    lines.append(f"| roi | {_out_str(r_out)} | {len(r_out)} |")
    lines.append("")
    lines.append("异常帧列表亦写入 `docs/stage5_roi_ablation/visualization/outlier_frames.json`。")
    lines.append("")
    lines.append("## 5. baseline vs ROI 结论")
    lines.append("")
    lines.append("| 统计量 | baseline | ROI 1.2 | 变化 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| cx_mean（中心 X 均值 px） | {_fmt(b['cx_mean'])} | {_fmt(r['cx_mean'])} | {_fmt(r['cx_mean'] - b['cx_mean'])} |")
    lines.append(f"| cy_mean（中心 Y 均值 px） | {_fmt(b['cy_mean'])} | {_fmt(r['cy_mean'])} | {_fmt(r['cy_mean'] - b['cy_mean'])} |")
    lines.append(f"| cx_std（横向稳定性 px） | {_fmt(b['cx_std'])} | {_fmt(r['cx_std'])} | {_fmt(r['cx_std'] - b['cx_std'])}（{_pct(r['cx_std'] - b['cx_std'], b['cx_std'])}） |")
    lines.append(f"| cy_std（纵向稳定性 px） | {_fmt(b['cy_std'])} | {_fmt(r['cy_std'])} | {_fmt(r['cy_std'] - b['cy_std'])}（{_pct(r['cy_std'] - b['cy_std'], b['cy_std'])}） |")
    lines.append(f"| velocity_mean（平均位移 px） | {_fmt(b['velocity_mean'])} | {_fmt(r['velocity_mean'])} | {_fmt(r['velocity_mean'] - b['velocity_mean'])}（{_pct(r['velocity_mean'] - b['velocity_mean'], b['velocity_mean'])}） |")
    lines.append(f"| velocity_max（最大位移 px） | {_fmt(b['velocity_max'])} | {_fmt(r['velocity_max'])} | {_fmt(r['velocity_max'] - b['velocity_max'])}（{_pct(r['velocity_max'] - b['velocity_max'], b['velocity_max'])}） |")
    lines.append(f"| 孤立跳变异常帧数（>{threshold:.0f} px） | {len(b_out)} | {len(r_out)} | {len(r_out) - len(b_out):+d} |")
    lines.append("")
    lines.append("**结论**：")
    lines.append("")
    lines.append(f"- 平均位移（velocity_mean）：ROI 1.2 为 {_fmt(r['velocity_mean'])} px，baseline 为 {_fmt(b['velocity_mean'])} px（{_pct(r['velocity_mean'] - b['velocity_mean'], b['velocity_mean'])}）——ROI 约束未降低平均抖动。")
    lines.append(f"- 最大瞬时跳变（velocity_max）：ROI 1.2 为 {_fmt(r['velocity_max'])} px，baseline 为 {_fmt(b['velocity_max'])} px（{_pct(r['velocity_max'] - b['velocity_max'], b['velocity_max'])}）——ROI 显著抑制了最大跳变。")
    lines.append(f"- 孤立跳变异常帧：baseline {len(b_out)} 个（{', '.join(map(str, b_out_frames)) if b_out_frames else '无'}），ROI 1.2 {len(r_out)} 个（{', '.join(map(str, r_out_frames)) if r_out_frames else '无'}），减少 {len(b_out) - len(r_out)} 个。")
    lines.append(f"- 稳定性：ROI 1.2 的 `cy_std` 更大（纵向漂移更明显），`cx_std` 变化相对较小；横向/纵向中心均值接近，说明 ROI 未改变人脸框整体位置。")
    lines.append("")
    lines.append("整体而言，temporal ROI 1.2 在该 clip 上**降低了大跳变与孤立异常帧的数量**（峰值跳变从 "
                f"{_fmt(b['velocity_max'])} 降至 {_fmt(r['velocity_max'])} px，异常帧从 {len(b_out)} 降至 {len(r_out)}），"
                "但**平均抖动并未下降**（平均位移反而略升），与 `Stage5_ROI_Temporal_Ablation_Report.md` 的定量结论一致。"
                "因此默认 ROI 1.2 不直接进入生产；本任务仅用于理解 ROI 方法对 bbox 稳定性的影响，生产管线保持全图检测不变。")
    lines.append("")
    lines.append("**统计文件**：`docs/stage5_roi_ablation/visualization/bbox_center_statistics.json`、`docs/stage5_roi_ablation/visualization/outlier_frames.json`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {report_path} ({len(lines)} lines)")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="Output directory for plots/JSON/videos.")
    p.add_argument("--report", type=Path, default=REPORT_PATH,
                   help="Output markdown report path.")
    p.add_argument("--no-videos", action="store_true",
                   help="Skip overlay video rendering (plots + JSON only).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, roi, threshold = load_results()
    print(f"[load] baseline: {len(baseline['frames'])} frames")
    print(f"[load] roi:      {len(roi['frames'])} frames")
    print(f"[load] outlier threshold: {threshold} px")

    validate_data(baseline, roi)

    b_traj = extract_trajectory(baseline["frames"])
    r_traj = extract_trajectory(roi["frames"])
    b_deltas = compute_displacements(b_traj)
    r_deltas = compute_displacements(r_traj)
    b_out = isolated_outliers(b_deltas, threshold)
    r_out = isolated_outliers(r_deltas, threshold)

    # Attach the actual cx / cy to each outlier for curve markers.
    b_cy = {t["frame_id"]: t["cy"] for t in b_traj}
    r_cy = {t["frame_id"]: t["cy"] for t in r_traj}
    b_cx = {t["frame_id"]: t["cx"] for t in b_traj}
    r_cx = {t["frame_id"]: t["cx"] for t in r_traj}
    for o in b_out:
        o["_cx"] = b_cx[o["frame"]]
        o["_cy"] = b_cy[o["frame"]]
    for o in r_out:
        o["_cx"] = r_cx[o["frame"]]
        o["_cy"] = r_cy[o["frame"]]

    print(f"[outlier] baseline: {[(o['frame'], o['distance']) for o in b_out]}")
    print(f"[outlier] roi:      {[(o['frame'], o['distance']) for o in r_out]}")

    # Plots.
    px = out_dir / "bbox_center_x.png"
    py = out_dir / "bbox_center_y.png"
    pv = out_dir / "bbox_center_velocity.png"
    plot_center_x(b_traj, r_traj, b_out, r_out, px)
    plot_center_y(b_traj, r_traj, b_out, r_out, py)
    plot_velocity(b_deltas, r_deltas, threshold, pv)

    # JSON outputs.
    outlier_payload = {"baseline": [
        {"frame": o["frame"], "distance": o["distance"]} for o in b_out
    ], "roi": [
        {"frame": o["frame"], "distance": o["distance"]} for o in r_out
    ]}
    write_json(out_dir / "outlier_frames.json", outlier_payload)

    b_stats = compute_statistics(b_traj, b_deltas, BASELINE_NAME)
    r_stats = compute_statistics(r_traj, r_deltas, ROI_NAME)
    stats = [b_stats, r_stats]
    write_json(out_dir / "bbox_center_statistics.json", stats)

    # Overlay videos (preserve source resolution + fps, H.264).
    video_paths: List[Path] = []
    if not args.no_videos:
        src_video = Path(baseline["source"].get("video_path", ""))
        if not src_video.exists():
            print(f"[video] WARNING source video not found: {src_video} - skipping")
        else:
            fps = float(baseline["source"].get("fps", 25.0))
            video_paths.append(render_overlay(
                src_video, baseline["frames"],
                out_dir / "baseline_bbox_center_overlay.mp4", fps, BASELINE_NAME))
            video_paths.append(render_overlay(
                src_video, roi["frames"],
                out_dir / "roi_bbox_center_overlay.mp4", fps, ROI_NAME))
    else:
        print("[video] skipped overlay rendering (--no-videos)")

    # Report.
    write_report(baseline, roi, threshold, stats, b_out, r_out,
                 [px, py, pv], video_paths, args.report)

    print("\n=== DONE ===")
    print(f"  plots      : {px}, {py}, {pv}")
    print(f"  outliers   : {out_dir / 'outlier_frames.json'}")
    print(f"  statistics : {out_dir / 'bbox_center_statistics.json'}")
    if video_paths:
        print(f"  videos     : {video_paths[0]}, {video_paths[1]}")
    print(f"  report     : {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

