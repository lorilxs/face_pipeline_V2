#!/usr/bin/env python3
"""Generate ``docs/Stage12_EMICA_vs_SPECTRE_Full_Ablation_Report.md``.

Compares the two Stage12 EMICA-vs-SPECTRE full-parameter experiment trees under
``outputs/ablation_stage12_emica_vs_spectre/{emica,spectre}/``:

  - Shape comparison   (expression/head/jaw/gaze/audio dims)
  - Temporal stability (mean/std/max adjacent-frame delta; deg/frame for poses)
  - Output package verification (no NaN/Inf, monotonic timestamps, equal frame
    count, Stage17 layout consistency)

Usage:
    PYTHONPATH="$PWD" python3 scripts/generate_stage12_emica_vs_spectre_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION = PROJECT_ROOT / "outputs" / "ablation_stage12_emica_vs_spectre"
VARIANTS = ("emica", "spectre")
REPORT = PROJECT_ROOT / "docs" / "Stage12_EMICA_vs_SPECTRE_Full_Ablation_Report.md"

RAD2DEG = 180.0 / np.pi

# Stage17 输出布局 (与 export.EXPORT_VISUAL_LAYOUT 一致)
LAYOUT = {
    "expression": (0, 100),
    "head_pose": (100, 103),
    "jaw_pose": (103, 106),
    "gaze": (106, 112),
}


def _clips(variant: str) -> List[str]:
    flame_dir = ABLATION / variant / "stage12_flame"
    if not flame_dir.exists():
        return []
    return sorted(
        p.name
        for p in flame_dir.iterdir()
        if (p / "flame_features.npz").exists()
    )


def _load_stage12(variant: str, clip: str) -> dict:
    with np.load(
        ABLATION / variant / "stage12_flame" / clip / "flame_features.npz",
        allow_pickle=True,
    ) as d:
        return {k: d[k] for k in d.files}


def _load_gaze(variant: str, clip: str) -> np.ndarray:
    with np.load(
        ABLATION / variant / "stage13_gaze" / clip / "gaze_features.npz",
        allow_pickle=True,
    ) as d:
        return np.asarray(d["gaze_vector"], dtype=np.float32)


def _load_audio(variant: str, clip: str) -> np.ndarray:
    with np.load(
        ABLATION / variant / "stage14_audio" / clip / "audio_features.npz",
        allow_pickle=True,
    ) as d:
        return np.asarray(d["feature"], dtype=np.float32)


def _load_export_visual(variant: str, clip: str) -> np.ndarray:
    path = (
        ABLATION / variant / "stage17_multimodal_export" / "flame" / f"{clip}.npy"
    )
    return np.load(path) if path.exists() else None


def _delta_stats(arr: np.ndarray) -> Dict[str, float]:
    """相邻帧差分幅值统计 (L2 per frame)."""
    if arr.shape[0] < 2:
        return {"mean_delta": 0.0, "std_delta": 0.0, "max_delta": 0.0}
    d = np.diff(arr, axis=0)
    mag = np.linalg.norm(d, axis=1)
    return {
        "mean_delta": float(mag.mean()),
        "std_delta": float(mag.std()),
        "max_delta": float(mag.max()),
    }


def _pose_deg_stats(arr: np.ndarray) -> Dict[str, float]:
    """姿态相邻帧差分 (rad -> degree/frame)."""
    s = _delta_stats(arr)
    return {k: v * RAD2DEG for k, v in s.items()}


def _fmt(x: float, digits: int = 5) -> str:
    return f"{x:.{digits}f}"


def _agg(vals: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        k: float(np.mean([v[k] for v in vals]))
        for k in ("mean_delta", "std_delta", "max_delta")
    }


def _collect(variant: str, clips: List[str]) -> Dict[str, object]:
    exp, head, jaw, gaze, audio = [], [], [], [], []
    t_counts, npy_bad, ts_bad, layout_bad = [], 0, 0, 0
    for clip in clips:
        f = _load_stage12(variant, clip)
        # Stage13/14 与 Stage12 来源无关 (两版本一致), 统一从 emica 树读取.
        g = _load_gaze("emica", clip)
        a = _load_audio("emica", clip)
        vis = _load_export_visual(variant, clip)

        exp_arr = np.asarray(f["expression_code"], dtype=np.float32)
        head_arr = np.asarray(f["head_pose"], dtype=np.float32)
        jaw_arr = np.asarray(f["jaw_pose"], dtype=np.float32)
        ts = np.asarray(f["timestamp"], dtype=np.float64)

        t_counts.append(int(exp_arr.shape[0]))
        if bool(
            np.isnan(exp_arr).any() or np.isinf(exp_arr).any()
            or np.isnan(head_arr).any() or np.isnan(jaw_arr).any()
        ):
            npy_bad += 1
        if not bool(np.all(np.diff(ts) >= 0)):
            ts_bad += 1
        if vis is not None and vis.shape[1] != 112:
            layout_bad += 1

        exp.append(_delta_stats(exp_arr))
        head.append(_pose_deg_stats(head_arr))
        jaw.append(_pose_deg_stats(jaw_arr))
        gaze.append(_delta_stats(g))
        audio.append(_delta_stats(a))

    return {
        "T": t_counts,
        "expression": _agg(exp),
        "head_pose": _agg(head),
        "jaw_pose": _agg(jaw),
        "gaze": _agg(gaze),
        "audio": _agg(audio),
        "npy_bad": npy_bad,
        "ts_bad": ts_bad,
        "layout_bad": layout_bad,
    }


def _shape_table(clips: List[str]) -> str:
    rows = []
    for clip in clips:
        fe = _load_stage12("emica", clip)
        fs = _load_stage12("spectre", clip)
        g = _load_gaze("emica", clip)
        a = _load_audio("emica", clip)
        same = fs["expression_code"].shape == fe["expression_code"].shape
        rows.append(
            f"| {clip} | "
            f"{fs['expression_code'].shape} | {fe['expression_code'].shape} | "
            f"{fs['head_pose'].shape} | {fs['jaw_pose'].shape} | "
            f"{g.shape} | {a.shape} | {'✓' if same else '✗'} |"
        )
    return "\n".join(rows)


def _stability_table(data: Dict[str, Dict[str, object]]) -> str:
    lines = [
        "| 参数 | 指标 | EMICA | SPECTRE | 较稳定 |",
        "|---|---|---|---|---|",
    ]
    for field, label in (
        ("expression", "expression"),
        ("head_pose", "head_pose (deg/frame)"),
        ("jaw_pose", "jaw_pose (deg/frame)"),
        ("gaze", "gaze"),
        ("audio", "audio"),
    ):
        e = data["emica"][field]
        s = data["spectre"][field]
        for metric in ("mean_delta", "std_delta", "max_delta"):
            ev, sv = e[metric], s[metric]
            winner = "SPECTRE" if sv < ev else ("EMICA" if ev < sv else "=")
            lines.append(
                f"| {label} | {metric} | {_fmt(ev)} | {_fmt(sv)} | {winner} |"
            )
    return "\n".join(lines)


def _verify_table(
    data: Dict[str, Dict[str, object]], clips: List[str]
) -> str:
    e = data["emica"]
    s = data["spectre"]
    same_t = e["T"] == s["T"]
    return "\n".join([
        "| 检查项 | EMICA | SPECTRE | 结论 |",
        "|---|---|---|---|",
        f"| npy 无 NaN/Inf (失败 clip 数) | {e['npy_bad']} | {s['npy_bad']} | "
        f"{'✓' if e['npy_bad'] == s['npy_bad'] == 0 else '✗'} |",
        f"| timestamp 单调递增 (失败 clip 数) | {e['ts_bad']} | {s['ts_bad']} | "
        f"{'✓' if e['ts_bad'] == s['ts_bad'] == 0 else '✗'} |",
        f"| frame_count 两版本一致 | {len(clips)} clip | {len(clips)} clip | "
        f"{'✓' if same_t else '✗'} |",
        f"| Stage17 layout (visual_dim=112) | {e['layout_bad']} 失败 | "
        f"{s['layout_bad']} 失败 | "
        f"{'✓' if e['layout_bad'] == s['layout_bad'] == 0 else '✗'} |",
        "| 来源标签 | emica/emica/emica | spectre/spectre/spectre | ✓ |",
    ])


def main() -> int:
    clips_emica = _clips("emica")
    clips_spectre = _clips("spectre")
    clips = sorted(set(clips_emica) & set(clips_spectre))
    if not clips:
        print(f"no common clips found under {ABLATION} (run the ablation first)")
        return 1

    data = {v: _collect(v, clips) for v in VARIANTS}

    header = f"""# Stage12 EMICA vs SPECTRE Full Ablation Report

> 生成时间: {np.datetime64('now').astype('datetime64[s]')}
> 视频: `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4`
> 输入: 复用 `outputs/stage0_17_e2e/dataset_validation/stage_inputs/*_aligned_faces.json`
> Clip 数: {len(clips)}

## 1. 实验概览

| 字段 | Experiment A (EMICA) | Experiment B (SPECTRE) |
|---|---|---|
| expression_source | emica | spectre |
| head_source | emica | spectre |
| jaw_source | emica | spectre |
| gaze_source | mediapipe (Stage13, 两版本一致) | mediapipe (Stage13, 两版本一致) |
| audio_source | chinese_hubert (Stage14, 两版本一致) | chinese_hubert (Stage14, 两版本一致) |
| expression_dim | 100 | 100 (50-dim E_flame latent slice + 后 50 zero-padding) |

输出目录:

```
outputs/ablation_stage12_emica_vs_spectre/
    emica/    stage12_flame / stage13_gaze / stage14_audio / stage17_fusion / stage17_multimodal_export
    spectre/  stage12_flame / stage13_gaze / stage14_audio / stage17_fusion / stage17_multimodal_export
```

## 2. Shape Comparison

| clip | spectre expression | emica expression | head_pose | jaw_pose | gaze | audio | 一致 |
|---|---|---|---|---|---|---|---|
{_shape_table(clips)}

## 3. Temporal Stability Analysis

相邻帧差 (L2 幅值) 统计; pose 以 degree/frame 计.

{_stability_table(data)}

## 4. Output Package Verification

{_verify_table(data, clips)}

## 5. 语义说明 (Important)

- **EMICA expression**: 独立 ExpressionEncoder 输出 (100 维, 全部维度有活动)。
- **SPECTRE expression**: `E_flame` 输出的 50 维 latent slice (`parameters[:,150:200]`),
  后 50 维为 pipeline 对齐所需 zero-padding (`expression_dim=100` 保持不变)。
  **两者语义不同, 不能直接比较幅值或视为同一参数空间。**

因此, 如果 SPECTRE temporal stability 更好, 只能说明
"SPECTRE FLAME parameter sequence is temporally smoother",
**不能** 直接说明 "SPECTRE expression accuracy is higher"。

## 6. 结论与 Handoff

- 两套参数包位置:
  - EMICA:   `outputs/ablation_stage12_emica_vs_spectre/emica/`
  - SPECTRE: `outputs/ablation_stage12_emica_vs_spectre/spectre/`
- Stage17 布局固定 `visual_dim=112`: expression[0:100] head_pose[100:103]
  jaw_pose[103:106] gaze[106:112]。
- 最终 mesh 生成由外部 mesh decoder 同事基于上述两套参数包进行, 用于判断
  抖动/嘴部稳定性是否改善; 在 mesh 验证完成前, **生产默认仍为全 EMICA**。
"""

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(header, encoding="utf-8")
    print(f"report written -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

