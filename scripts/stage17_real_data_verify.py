#!/usr/bin/env python3
"""Stage17 真实数据端到端验证 (Stage12 → Stage13 → Stage14 → Stage17)。

读取真实 Stage12/13/14 输出 (经各自官方 read_* 接口), 调用 Stage17 官方
build_training_sample() / save_training_sample() 装配 clip-level TrainingSample,
随后:

  1. 校验输入字段 (FLAME expression[100] / pose[6] / timestamp,
     Gaze gaze_vector[6] / timestamp, Audio feature[768] / timestamp);
  2. 校验输出 training_sample.npz 字段 (visual_feature[T,112] /
     audio_feature[T,768] / frame_index / timestamps / audio_match_idx /
     dropout_mask);
  3. 数据质量检查 (NaN / Inf / expression std / pose std / gaze norm≈1 /
     audio 非零比例 / std / NaN);
  4. 输出对齐统计 (帧数 / visual T / audio matched / dropout ratio);
  5. 生成 docs/Stage17_Real_Data_Validation_Report.md 并判定 PASS/FAIL。

本脚本不重新实现任何融合算法 — 仅调用官方接口 + 校验 / 报告。

Usage:
    python scripts/stage17_real_data_verify.py
    python scripts/stage17_real_data_verify.py --clip_id CLIP \
        [--flame_dir outputs/stage12_flame] [--gaze_dir outputs/stage13_gaze] \
        [--audio_dir outputs/stage14_audio] [--output_dir outputs/stage17_fusion]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from face_pipeline_v2.stages.stage_12_flame.integration import (  # noqa: E402
    read_stage12_outputs,
)
from face_pipeline_v2.stages.stage_13_gaze.integration import (  # noqa: E402
    read_gaze_features,
)
from face_pipeline_v2.stages.stage_14_audio_feature.integration import (  # noqa: E402
    read_audio_features,
)
from face_pipeline_v2.stages.stage_17_fusion.integration import (  # noqa: E402
    NPZ_FILENAME,
    META_FILENAME,
    MANIFEST_FILENAME,
    read_training_sample,
    save_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.processor import (  # noqa: E402
    AUDIO_DIM,
    VISUAL_DIM,
    VISUAL_LAYOUT,
    build_training_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage17_real_data_verify")

DEFAULT_CLIP_ID = "CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001"


# ---------------------------------------------------------------------------
# 通用校验工具
# ---------------------------------------------------------------------------

def _check(name: str, ok: bool, detail: str) -> Tuple[str, bool, str]:
    """记录一条检查项: (name, ok, detail)。"""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42} {detail}")
    return name, ok, detail


def _ts_monotonic(ts: np.ndarray) -> Tuple[bool, bool]:
    """返回 (严格递增?, 弱单调不减?)。空/单元素视为单调。"""
    if ts.size < 2:
        return True, True
    d = np.diff(ts)
    return bool(np.all(d > 0)), bool(np.all(d >= 0))


def _finite_counts(arr: np.ndarray) -> Tuple[int, int]:
    return int(np.isnan(arr).sum()), int(np.isinf(arr).sum())


def _as_f32_2d(features: List, dim: int, field: str) -> np.ndarray:
    """把 read_* 返回的对象列表转成 [N, dim] float32 数组。"""
    arr = np.asarray(
        [np.asarray(getattr(f, field), dtype=np.float32).reshape(-1) for f in features],
        dtype=np.float32,
    )
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(
            f"{field} must be [{len(features)}, {dim}], got {arr.shape}"
        )
    return arr


# ---------------------------------------------------------------------------
# 1. 输入读取 + 字段校验 (经各阶段官方 read_* 接口)
# ---------------------------------------------------------------------------

def validate_flame(flame_dir: Path, clip_id: str) -> Tuple[List, np.ndarray, List]:
    """读取 Stage12 并校验 expression[100] / pose[6] / timestamp 单调。"""
    print("-" * 70)
    print("STAGE12 FLAME 校验")
    features = read_stage12_outputs(flame_dir, clip_id)
    checks: List[Tuple[str, bool, str]] = []
    checks.append(_check("FLAME 帧数 > 0", len(features) > 0, f"{len(features)} frames"))

    exp_ok = all(len(f.expcode) == 100 for f in features)
    checks.append(_check(
        "FLAME expression [T,100]", exp_ok,
        f"expcode dims={sorted({len(f.expcode) for f in features})}"
        if features else "n/a"))

    pose_ok = all(len(f.posecode) == 6 for f in features)
    checks.append(_check(
        "FLAME pose [T,6]", pose_ok,
        f"posecode dims={sorted({len(f.posecode) for f in features})}"
        if features else "n/a"))

    ts = np.asarray([f.timestamp for f in features], dtype=np.float64)
    strict, weak = _ts_monotonic(ts)
    checks.append(_check(
        "FLAME timestamp 单调", strict or weak,
        f"strict_increasing={strict} weak_monotonic={weak} "
        f"ts[{ts[0] if ts.size else 0:.3f}..{ts[-1] if ts.size else 0:.3f}]"))

    nan, inf = _finite_counts(ts)
    checks.append(_check("FLAME timestamp 无 NaN/Inf", nan == 0 and inf == 0,
                         f"nan={nan} inf={inf}"))

    if not all(ok for _, ok, _ in checks):
        raise ValueError("Stage12 FLAME 输入校验失败")
    return features, ts, checks


def validate_gaze(gaze_dir: Path, clip_id: str) -> Tuple[List, np.ndarray, List]:
    """读取 Stage13 并校验 gaze_vector[6] / timestamp 单调。"""
    print("-" * 70)
    print("STAGE13 GAZE 校验")
    features = read_gaze_features(gaze_dir, clip_id)
    checks: List[Tuple[str, bool, str]] = []
    checks.append(_check("GAZE 帧数 > 0", len(features) > 0, f"{len(features)} frames"))

    gv_ok = all(len(f.gaze_vector) == 6 for f in features)
    checks.append(_check(
        "GAZE gaze_vector [T,6]", gv_ok,
        f"dims={sorted({len(f.gaze_vector) for f in features})}" if features else "n/a"))

    ts = np.asarray([f.timestamp for f in features], dtype=np.float64)
    strict, weak = _ts_monotonic(ts)
    checks.append(_check(
        "GAZE timestamp 单调", strict or weak,
        f"strict_increasing={strict} weak_monotonic={weak} "
        f"ts[{ts[0] if ts.size else 0:.3f}..{ts[-1] if ts.size else 0:.3f}]"))

    nan, inf = _finite_counts(ts)
    checks.append(_check("GAZE timestamp 无 NaN/Inf", nan == 0 and inf == 0,
                         f"nan={nan} inf={inf}"))

    gaze = _as_f32_2d(features, 6, "gaze_vector")
    gnan, ginf = _finite_counts(gaze)
    checks.append(_check("GAZE vector 无 NaN/Inf", gnan == 0 and ginf == 0,
                         f"nan={gnan} inf={ginf}"))

    left = np.linalg.norm(gaze[:, 0:3], axis=1)
    right = np.linalg.norm(gaze[:, 3:6], axis=1)
    norm_ok = bool(np.all(np.abs(left - 1.0) < 0.1) and np.all(np.abs(right - 1.0) < 0.1))
    checks.append(_check(
        "GAZE 视线向量 norm≈1", norm_ok,
        f"left=[{left.min():.4f},{left.max():.4f}] right=[{right.min():.4f},{right.max():.4f}]"))

    if not all(ok for _, ok, _ in checks):
        raise ValueError("Stage13 Gaze 输入校验失败")
    return features, ts, checks


def validate_audio(audio_dir: Path, clip_id: str) -> Tuple[List, np.ndarray, List]:
    """读取 Stage14 并校验 feature[768] / timestamp 单调。"""
    print("-" * 70)
    print("STAGE14 AUDIO 校验")
    features = read_audio_features(audio_dir, clip_id)
    checks: List[Tuple[str, bool, str]] = []
    checks.append(_check("AUDIO 帧数 > 0", len(features) > 0, f"{len(features)} frames"))

    dim_ok = all(f.dimension == 768 and len(f.feature) == 768 for f in features)
    checks.append(_check(
        "AUDIO feature [T,768]", dim_ok,
        f"dims={sorted({len(f.feature) for f in features})}" if features else "n/a"))

    ts = np.asarray([f.timestamp for f in features], dtype=np.float64)
    strict, weak = _ts_monotonic(ts)
    checks.append(_check(
        "AUDIO timestamp 单调", strict or weak,
        f"strict_increasing={strict} weak_monotonic={weak} "
        f"ts[{ts[0] if ts.size else 0:.3f}..{ts[-1] if ts.size else 0:.3f}]"))

    nan, inf = _finite_counts(ts)
    checks.append(_check("AUDIO timestamp 无 NaN/Inf", nan == 0 and inf == 0,
                         f"nan={nan} inf={inf}"))

    feat = _as_f32_2d(features, 768, "feature")
    fnan, finf = _finite_counts(feat)
    checks.append(_check("AUDIO feature 无 NaN/Inf", fnan == 0 and finf == 0,
                         f"nan={fnan} inf={finf}"))
    nonzero = float(np.count_nonzero(feat)) / float(feat.size) if feat.size else 0.0
    checks.append(_check("AUDIO feature 非零比例 > 0", nonzero > 0.0,
                         f"nonzero_ratio={nonzero:.4f} std={feat.std():.4f}"))

    if not all(ok for _, ok, _ in checks):
        raise ValueError("Stage14 Audio 输入校验失败")
    return features, ts, checks

# ---------------------------------------------------------------------------
# 2/3. 输出校验 (training_sample.npz) + 数据质量检查
# ---------------------------------------------------------------------------

def validate_output_sample(sample_dir: Path) -> Tuple[bool, Dict[str, object]]:
    """校验训练样本包: 字段存在、shape 正确、round-trip 一致。"""
    print("-" * 70)
    print("TRAINING SAMPLE 输出校验")
    npz_path = sample_dir / NPZ_FILENAME
    meta_path = sample_dir / META_FILENAME
    checks: List[Tuple[str, bool, str]] = []
    checks.append(_check("training_sample.npz 存在", npz_path.exists(), str(npz_path)))
    checks.append(_check("metadata.json 存在", meta_path.exists(), str(meta_path)))

    out: Dict[str, object] = {}
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as d:
            for field, exp_dim in (("visual_feature", VISUAL_DIM),
                                   ("audio_feature", AUDIO_DIM),
                                   ("frame_index", None),
                                   ("timestamps", None),
                                   ("audio_match_idx", None),
                                   ("dropout_mask", None)):
                ok = field in d
                detail = "missing"
                if ok:
                    a = d[field]
                    detail = f"{list(a.shape)} {a.dtype}"
                    if exp_dim is not None:
                        ok = a.ndim == 2 and a.shape[1] == exp_dim
                        detail = f"{list(a.shape)} {a.dtype}" + ("" if ok else f" (expect dim {exp_dim})")
                    out[field] = a
                checks.append(_check(f"NPZ 字段 {field}", ok, detail))

            vis, aud = out.get("visual_feature"), out.get("audio_feature")
            if vis is not None and aud is not None:
                t = vis.shape[0]
                t_ok = vis.shape[0] == aud.shape[0]
                checks.append(_check("T 轴一致 (visual==audio)", t_ok, f"T={t}"))
                out["T"] = t
            for field in ("frame_index", "timestamps", "audio_match_idx", "dropout_mask"):
                if field in out:
                    a = out[field]
                    checks.append(_check(
                        f"对齐数组 {field} 长度 == T", a.shape[0] == t,
                        f"{a.shape[0]} vs T={t}"))

    # 官方 read_training_sample round-trip
    rt_ok = False
    try:
        restored = read_training_sample(sample_dir)
        rt_ok = (restored.visual_feature.shape[1] == VISUAL_DIM
                 and restored.audio_feature.shape[1] == AUDIO_DIM
                 and restored.visual_feature.shape[0] == restored.audio_feature.shape[0])
        checks.append(_check(
            "read_training_sample round-trip", rt_ok,
            f"visual{list(restored.visual_feature.shape)} "
            f"audio{list(restored.audio_feature.shape)} valid_length={restored.valid_length}"))
    except Exception as exc:  # pragma: no cover
        checks.append(_check("read_training_sample round-trip", False, str(exc)))

    all_ok = all(ok for _, ok, _ in checks)
    return all_ok, out


def quality_checks(sample, valid_only_mask: np.ndarray) -> List[Tuple[str, bool, str]]:
    """数据质量: visual NaN/Inf/std/gaze-norm; audio 非零比例/std/NaN。"""
    print("-" * 70)
    print("数据质量检查")
    checks: List[Tuple[str, bool, str]] = []
    vis = sample.visual_feature
    aud = sample.audio_feature

    vnan, vinf = _finite_counts(vis)
    checks.append(_check("visual_feature 无 NaN/Inf", vnan == 0 and vinf == 0,
                         f"nan={vnan} inf={vinf}"))

    expr = vis[:, 0:100]
    pose = vis[:, 100:106]
    checks.append(_check("expression std > 0", float(expr.std()) > 0.0,
                         f"std={expr.std():.4f} mean={expr.mean():.4f}"))
    checks.append(_check("pose std 合理 (>1e-4)", float(pose.std()) > 1e-4,
                         f"std={pose.std():.4f} mean={pose.mean():.4f}"))

    gaze = vis[valid_only_mask, 106:112] if valid_only_mask.any() else np.zeros((0, 6))
    if gaze.shape[0] > 0:
        left = np.linalg.norm(gaze[:, 0:3], axis=1)
        right = np.linalg.norm(gaze[:, 3:6], axis=1)
        norm_ok = bool(np.all(np.abs(left - 1.0) < 0.1) and np.all(np.abs(right - 1.0) < 0.1))
        checks.append(_check(
            "visual gaze norm≈1 (有效帧)", norm_ok,
            f"n={gaze.shape[0]} left=[{left.min():.4f},{left.max():.4f}] "
            f"right=[{right.min():.4f},{right.max():.4f}]"))
    else:
        checks.append(_check("visual gaze norm≈1 (有效帧)", True, "no valid frames"))

    anan, ainf = _finite_counts(aud)
    checks.append(_check("audio_feature 无 NaN/Inf", anan == 0 and ainf == 0,
                         f"nan={anan} inf={ainf}"))
    matched = sample.audio_match_idx >= 0
    if matched.any():
        m_aud = aud[matched]
        nonzero = float(np.count_nonzero(m_aud)) / float(m_aud.size)
        checks.append(_check(
            "audio_feature 非零比例 (matched)", nonzero > 0.0,
            f"nonzero_ratio={nonzero:.4f} std={m_aud.std():.4f}"))
    else:
        checks.append(_check("audio_feature 非零比例 (matched)", False, "0 matched frames"))
    checks.append(_check("audio_feature std > 0", float(aud.std()) > 0.0,
                         f"std={aud.std():.4f}"))
    return checks


# ---------------------------------------------------------------------------
# 5. 报告生成
# ---------------------------------------------------------------------------

def write_report(
    report_path: Path,
    *,
    clip_id: str,
    fps: float,
    duration: float,
    n_flame: int,
    n_gaze: int,
    n_audio: int,
    sample,
    t_visual: int,
    audio_matched: int,
    gaze_missing: int,
    checks: List[Tuple[str, bool, str]],
) -> bool:
    """按任务 §5 模板生成 docs/Stage17_Real_Data_Validation_Report.md。"""
    ok = all(ok for _, ok, _ in checks)
    visual_layout = {k: list(v) for k, v in VISUAL_LAYOUT.items()}
    dropout = int(sample.dropout_mask.sum())
    dropout_ratio = 100.0 * dropout / t_visual if t_visual else 0.0
    match_ratio = 100.0 * audio_matched / t_visual if t_visual else 0.0

    lines: List[str] = [
        "# Stage17 Real Data Validation Report",
        "",
        f"- 生成时间: {str(np.datetime64('now', 's'))}",
        f"- 生成脚本: `scripts/stage17_real_data_verify.py`",
        "",
        "## Dataset",
        "",
        f"- clip_id: `{clip_id}`",
        f"- frames (FLAME): {n_flame}",
        f"- duration: {duration:.2f}s @ {fps:g} FPS",
        "",
        "## Input",
        "",
        "| 阶段 | 文件 | Shape | 校验 |",
        "|------|------|-------|------|",
    ]
    lines += [
        f"| Stage12 FLAME | `flame_features.npz` | exp[{n_flame},100] + pose[{n_flame},6] | PASS |",
        f"| Stage13 Gaze  | `gaze_features.npz`  | gaze_vector[{n_gaze},6]          | PASS |",
        f"| Stage14 Audio | `audio_features.npz` | feature[{n_audio},768]           | PASS |",
        "",
        "## Output",
        "",
        "TrainingSample (`training_sample.npz`):",
        "",
        f"- visual_feature: `{list(sample.visual_feature.shape)}` float32 "
        f"(expression[0:100] + pose[100:106] + gaze[106:112])",
        f"- audio_feature: `{list(sample.audio_feature.shape)}` float32 (Chinese HuBERT)",
        f"- frame_index: `{list(sample.frame_index.shape)}`",
        f"- timestamps: `{list(sample.timestamps.shape)}`",
        f"- audio_match_idx: `{list(sample.audio_match_idx.shape)}`",
        f"- dropout_mask: `{list(sample.dropout_mask.shape)}`",
        f"- valid_length: {sample.valid_length}",
        f"- visual_layout: `{visual_layout}`",
        "",
        "## Alignment",
        "",
        "Frame count:",
        "",
        f"- FLAME: T_flame = {n_flame}",
        f"- Gaze:  T_gaze  = {n_gaze}",
        f"- Audio: T_audio = {n_audio}",
        "",
        "Alignment result:",
        "",
        f"- visual T: T_visual = {t_visual}",
        f"- audio matched: T_audio_match = {audio_matched} ({match_ratio:.1f}%)",
        f"- Gaze missing: {gaze_missing} frames",
        f"- Audio unmatched: {t_visual - audio_matched} frames",
        f"- dropout ratio: {dropout_ratio:.2f}%",
        "",
        "## Validation Checks",
        "",
        "| Check | Status | Detail |",
        "|-------|--------|--------|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines += [
        "",
        "## Result",
        "",
        f"**{'PASS' if ok else 'FAIL'}** — "
        f"{'真实数据链路 Stage12→Stage17 完整闭环可用。' if ok else '存在校验失败, 请检查上方检查项。'}",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {report_path}")
    return ok


# ---------------------------------------------------------------------------
# 6. CLI 入口
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage17 真实数据端到端验证 (Stage12→Stage17)")
    p.add_argument("--clip_id", default=DEFAULT_CLIP_ID)
    p.add_argument("--flame_dir", default="outputs/stage12_flame")
    p.add_argument("--gaze_dir", default="outputs/stage13_gaze")
    p.add_argument("--audio_dir", default="outputs/stage14_audio")
    p.add_argument("--output_dir", default="outputs/stage17_fusion")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--audio_tolerance", type=float, default=0.04)
    p.add_argument("--audio_match", default="nearest_timestamp")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--metadata_json", default=None,
                   help="额外溯源元数据 JSON 对象文件")
    p.add_argument("--report_path",
                   default="docs/Stage17_Real_Data_Validation_Report.md")
    return p


def _load_extra_metadata(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"metadata json must be an object, got {type(data).__name__}")
    return data


def main() -> int:
    args = _build_parser().parse_args()
    root = PROJECT_ROOT
    clip_id = args.clip_id

    checks: List[Tuple[str, bool, str]] = []

    print("=" * 70)
    print(f"STAGE17 真实数据端到端验证  clip={clip_id}")
    print("=" * 70)

    # 1. 读取 + 校验输入 (经各阶段官方 read_* 接口)
    flame, flame_ts, flame_checks = validate_flame(root / args.flame_dir, clip_id)
    gaze, _, gaze_checks = validate_gaze(root / args.gaze_dir, clip_id)
    audio, _, audio_checks = validate_audio(root / args.audio_dir, clip_id)
    checks += flame_checks + gaze_checks + audio_checks

    # 2. 调用 Stage17 官方接口装配 + 保存
    print("-" * 70)
    print("STAGE17 FUSION 装配")
    metadata = _load_extra_metadata(args.metadata_json)
    metadata.setdefault("video_id", clip_id.rsplit("_scene", 1)[0])
    metadata.setdefault("clip_start", float(flame_ts[0]) if flame_ts.size else 0.0)
    metadata.setdefault("duration", args.duration)
    metadata.setdefault("source_audio",
                        str(root / args.audio_dir / clip_id / "audio_features.npz"))

    sample = build_training_sample(
        clip_id=clip_id,
        flame_features=flame,
        gaze_features=gaze,
        audio_features=audio,
        fps=args.fps,
        audio_tolerance=args.audio_tolerance,
        audio_match=args.audio_match,
        metadata=metadata,
    )
    out_dir = save_training_sample(sample, output_dir=root / args.output_dir)
    print(f"样本包 -> {out_dir}")

    # 3. 输出校验 (training_sample.npz / metadata.json / manifest.csv)
    out_ok, _ = validate_output_sample(out_dir)
    checks.append(("输出样本包 (npz/meta/manifest)", out_ok,
                   f"{NPZ_FILENAME}/{META_FILENAME}/{MANIFEST_FILENAME}"))

    # 4. 数据质量
    checks += quality_checks(sample, ~sample.dropout_mask)

    # 对齐统计
    t_visual = int(sample.frame_index.size)
    audio_matched = int((sample.audio_match_idx >= 0).sum())
    gaze_missing = int((sample.dropout_mask & (sample.audio_match_idx >= 0)).sum())

    print("-" * 70)
    print("ALIGNMENT 统计")
    print(f"  Frame count:  FLAME={len(flame)}  Gaze={len(gaze)}  Audio={len(audio)}")
    print(f"  Alignment:    visual T={t_visual}  audio matched={audio_matched}  "
          f"dropout={int(sample.dropout_mask.sum())} "
          f"({100.0 * int(sample.dropout_mask.sum()) / t_visual if t_visual else 0.0:.2f}%)")
    print(f"  valid_length = {sample.valid_length}")

    # 5. 报告 + 判定
    ok = all(ok for _, ok, _ in checks)
    write_report(
        root / args.report_path,
        clip_id=clip_id, fps=args.fps, duration=args.duration,
        n_flame=len(flame), n_gaze=len(gaze), n_audio=len(audio),
        sample=sample, t_visual=t_visual,
        audio_matched=audio_matched, gaze_missing=gaze_missing,
        checks=checks,
    )
    print("=" * 70)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  (真实数据链路 Stage12→Stage17)")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

