#!/usr/bin/env python3
"""Stage17 Multimodal Export — 真实数据端到端验证脚本。

读取真实 Stage12/13/14 输出 (经各阶段官方 read_* 接口), 重新执行 Stage17 fusion
(build_training_sample + save_training_sample), 随后执行 Stage17 export
(export_multimodal_sample) 生成最终交付目录:

  outputs/stage17_multimodal_export/
    flame/{clip_id}.npy       # [T,112] float32 视觉运动特征 (FLAME + Gaze)
    audio/{clip_id}.npy       # [T,768] float32  HuBERT embedding
    timestamp/{clip_id}.npz   # frame_index / timestamps / audio_match_idx / dropout_mask
    metadata/{clip_id}.json   # 溯源元数据
    manifest.csv              # 数据集级索引

随后校验 (export_structure / flame_shape / audio_shape / timestamp_alignment /
metadata_check / round-trip) 并生成 docs/Stage17_Multimodal_Export_Report.md。

本脚本不重新实现任何融合/导出算法 — 仅调用官方接口 + 校验 / 报告。
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
    read_training_sample,
    save_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.processor import (  # noqa: E402
    AUDIO_DIM,
    VISUAL_DIM,
    build_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.export import (  # noqa: E402
    AUDIO_DIR,
    FLAME_DIR,
    MANIFEST_FILENAME,
    METADATA_DIR,
    REQUIRED_METADATA_FIELDS,
    TIMESTAMP_ARRAY_FIELDS,
    TIMESTAMP_DIR,
    export_multimodal_sample,
    read_multimodal_export,
    validate_export_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage17_export_verify")

DEFAULT_CLIP_ID = "CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001"


# ---------------------------------------------------------------------------
# 通用校验工具
# ---------------------------------------------------------------------------

def _check(name: str, ok: bool, detail: str) -> Tuple[str, bool, str]:
    """记录一条检查项: (name, ok, detail)。"""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<24} {detail}")
    return name, ok, detail


def _ts_monotonic(ts: np.ndarray) -> Tuple[bool, bool]:
    """返回 (严格递增?, 弱单调不减?)。空/单元素视为单调。"""
    if ts.size < 2:
        return True, True
    d = np.diff(ts)
    return bool(np.all(d > 0)), bool(np.all(d >= 0))


def _export_file_paths(root: Path, clip_id: str) -> Dict[str, Path]:
    return {
        "flame": root / FLAME_DIR / f"{clip_id}.npy",
        "audio": root / AUDIO_DIR / f"{clip_id}.npy",
        "timestamp": root / TIMESTAMP_DIR / f"{clip_id}.npz",
        "metadata": root / METADATA_DIR / f"{clip_id}.json",
        "manifest": root / MANIFEST_FILENAME,
    }


def _validate_export(
    export_root: Path,
    clip_id: str,
    sample,
) -> List[Tuple[str, bool, str]]:
    """校验交付目录: structure / flame_shape / audio_shape / timestamp_alignment /
    metadata_check / round-trip。"""
    checks: List[Tuple[str, bool, str]] = []
    paths = _export_file_paths(export_root, clip_id)

    # 1. export_structure — 目录结构检查
    missing = [p.name for p in paths.values() if not p.exists()]
    ok = not missing
    checks.append(_check(
        "export_structure",
        ok,
        "flame/audio/timestamp/metadata/manifest 齐全"
        if ok else f"缺失: {', '.join(missing)}",
    ))

    # 2/3. flame_shape / audio_shape
    flame = np.load(paths["flame"])
    audio = np.load(paths["audio"])
    t_visual = flame.shape[0]
    ok_f = flame.shape == (t_visual, VISUAL_DIM) and flame.dtype == np.float32
    checks.append(_check(
        "flame_shape", ok_f, f"flame={list(flame.shape)} float32 (期望 [T,{VISUAL_DIM}])"
    ))
    ok_a = audio.shape == (t_visual, AUDIO_DIM) and audio.dtype == np.float32
    checks.append(_check(
        "audio_shape", ok_a, f"audio={list(audio.shape)} float32 (期望 [T,{AUDIO_DIM}])"
    ))

    # 4. timestamp_alignment — frame 对齐检查
    with np.load(paths["timestamp"], allow_pickle=False) as d:
        frame_index = d["frame_index"]
        timestamps = d["timestamps"]
        audio_match_idx = d["audio_match_idx"]
        dropout_mask = d["dropout_mask"]
    len_ok = all(
        a.shape[0] == t_visual
        for a in (frame_index, timestamps, audio_match_idx, dropout_mask)
    )
    strict, weak = _ts_monotonic(timestamps)
    fi_ok = np.array_equal(frame_index, sample.frame_index)
    dm_ok = np.array_equal(dropout_mask, sample.dropout_mask)
    ami_ok = np.array_equal(audio_match_idx, sample.audio_match_idx)
    ok_ts = len_ok and weak and fi_ok and dm_ok and ami_ok
    checks.append(_check(
        "timestamp_alignment",
        ok_ts,
        f"T={t_visual}, 4 数组长度一致={len_ok}, timestamp 单调={strict}, "
        f"frame_index 一致={fi_ok}, dropout 一致={dm_ok}, audio_match 一致={ami_ok}",
    ))

    # 5. metadata_check — 元数据字段完整性
    with open(paths["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    ok_m = validate_export_metadata(meta)
    missing_m = [k for k in REQUIRED_METADATA_FIELDS if k not in meta]
    checks.append(_check(
        "metadata_check",
        ok_m,
        f"必含字段 {len(REQUIRED_METADATA_FIELDS) - len(missing_m)}/"
        f"{len(REQUIRED_METADATA_FIELDS)}, "
        f"layout={meta.get('visual_layout')}, "
        f"head={meta.get('head_source')}, jaw={meta.get('jaw_source')}, "
        f"gaze={meta.get('gaze_source')}, audio={meta.get('audio_source')}",
    ))

    # 6. round-trip — 读取导出结果后 shape 一致 / 数据无变化
    loaded = read_multimodal_export(export_root, clip_id)
    ok_rt = (
        np.array_equal(loaded["visual_feature"], sample.visual_feature)
        and np.array_equal(loaded["audio_feature"], sample.audio_feature)
        and np.array_equal(loaded["frame_index"], sample.frame_index)
        and np.array_equal(loaded["timestamps"], sample.timestamps)
        and np.array_equal(loaded["audio_match_idx"], sample.audio_match_idx)
        and np.array_equal(loaded["dropout_mask"], sample.dropout_mask)
    )
    checks.append(_check(
        "round_trip",
        ok_rt,
        "flame/audio/timestamp 与 fusion 样本逐元素一致"
        if ok_rt else "round-trip 数据不一致",
    ))

    return checks


def write_report(
    report_path: Path,
    *,
    clip_id: str,
    fps: float,
    duration: float,
    t_visual: int,
    audio_matched: int,
    checks: List[Tuple[str, bool, str]],
) -> bool:
    """生成 docs/Stage17_Multimodal_Export_Report.md (含任务 §5 汇总表)。"""
    ok = all(ok for _, ok, _ in checks)
    match_ratio = 100.0 * audio_matched / t_visual if t_visual else 0.0

    lines: List[str] = [
        "# Stage17 Multimodal Export Report",
        "",
        f"- 生成时间: {str(np.datetime64('now', 's'))}",
        f"- 生成脚本: `scripts/stage17_multimodal_export_verify.py`",
        "",
        "## Dataset",
        "",
        f"- clip_id: `{clip_id}`",
        f"- 帧数 (visual T): {t_visual}",
        f"- 时长: {duration:.2f}s @ {fps:g} FPS",
        f"- audio 匹配帧: {audio_matched} ({match_ratio:.1f}%)",
        "",
        "## 交付格式 (Data Contract)",
        "",
        "| 文件 | Shape | dtype | 内容 |",
        "|------|-------|-------|------|",
        "| `flame/{clip_id}.npy` | `[T,112]` | float32 | 视觉运动特征 (FLAME expression[0:100] + head/jaw pose[100:106] + Gaze[106:112]) |",
        "| `audio/{clip_id}.npy` | `[T,768]` | float32 | HuBERT 音频 embedding |",
        "| `timestamp/{clip_id}.npz` | `[T]` × 4 | int32/float32/bool | frame_index / timestamps / audio_match_idx / dropout_mask |",
        "| `metadata/{clip_id}.json` | - | JSON | 溯源元数据 (含 visual_layout / *_source) |",
        "| `manifest.csv` | - | CSV | 数据集级索引 |",
        "",
        "## Summary",
        "",
        "| Field Name | 中文名称 | Type | Field Description |",
        "|------------|----------|------|--------------------|",
    ]
    summary_rows = [
        ("export_structure", "输出结构", "string", "目录结构检查"),
        ("flame_shape", "visual维度", "string", "[T,112]"),
        ("audio_shape", "audio维度", "string", "[T,768]"),
        ("timestamp_alignment", "时间同步", "string", "frame对齐检查"),
        ("metadata_check", "元数据检查", "bool", "字段完整性"),
        ("final_result", "最终结果", "string", "PASS/FAIL"),
    ]
    for fname, cn, typ, desc in summary_rows:
        status = (
            ok if fname == "final_result"
            else next((ok for n, ok, _ in checks if n == fname), False)
        )
        lines.append(
            f"| {fname} | {cn} | {typ} | {'PASS' if status else 'FAIL'} — {desc} |"
        )
    lines += [
        "",
        "## Validation Checks",
        "",
        "| Check | Status | Detail |",
        "|-------|--------|--------|",
    ]
    for name, ok_c, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok_c else 'FAIL'} | {detail} |")
    lines += [
        "",
        "## Result",
        "",
        f"**{'PASS' if ok else 'FAIL'}** — "
        f"{'Stage17 多模态导出层完整闭环可用, 交付格式符合数据契约。' if ok else '存在校验失败, 请检查上方检查项。'}",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {report_path}")
    return ok


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage17 多模态导出层端到端验证 (Stage12→Stage17 fusion → export)")
    p.add_argument("--clip_id", default=DEFAULT_CLIP_ID)
    p.add_argument("--flame_dir", default="outputs/stage12_flame")
    p.add_argument("--gaze_dir", default="outputs/stage13_gaze")
    p.add_argument("--audio_dir", default="outputs/stage14_audio")
    p.add_argument("--fusion_dir", default="outputs/stage17_fusion")
    p.add_argument("--export_dir", default="outputs/stage17_multimodal_export")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--audio_tolerance", type=float, default=0.04)
    p.add_argument("--audio_match", default="nearest_timestamp")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--report_path",
                   default="docs/Stage17_Multimodal_Export_Report.md")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    root = PROJECT_ROOT
    clip_id = args.clip_id

    # 1. 读取真实输入 (经各阶段官方 read_* 接口)
    logger.info("读取 Stage12/13/14 真实输出 ...")
    flame = read_stage12_outputs(root / args.flame_dir, clip_id)
    gaze = read_gaze_features(root / args.gaze_dir, clip_id)
    audio = read_audio_features(root / args.audio_dir, clip_id)
    n_flame, n_gaze, n_audio = len(flame), len(gaze), len(audio)
    logger.info(f"输入: FLAME={n_flame} GAZE={n_gaze} AUDIO={n_audio}")

    # 2. 重新执行 Stage17 fusion (官方接口)
    logger.info("重新执行 Stage17 fusion (build_training_sample) ...")
    metadata: Dict[str, object] = {
        "video_id": clip_id.split("_scene", 1)[0],
        "clip_start": 4.0,
        "duration": args.duration,
        "audio_backend": "Chinese HuBERT",
        "audio_match": args.audio_match,
        "audio_tolerance": args.audio_tolerance,
    }
    sample = build_training_sample(
        flame_features=flame,
        gaze_features=gaze,
        audio_features=audio,
        fps=args.fps,
        audio_tolerance=args.audio_tolerance,
        audio_match=args.audio_match,
        clip_id=clip_id,
        metadata=metadata,
    )
    save_training_sample(sample, output_dir=root / args.fusion_dir)
    logger.info(f"fusion 样本: visual={list(sample.visual_feature.shape)} "
                f"audio={list(sample.audio_feature.shape)}")

    # 3. 执行 Stage17 export (官方接口)
    logger.info("执行 Stage17 export (export_multimodal_sample) ...")
    export_root = export_multimodal_sample(sample, root / args.export_dir)
    logger.info(f"交付目录: {export_root}")

    # 4. 校验 + 报告
    checks = _validate_export(export_root, clip_id, sample)
    audio_matched = int((sample.audio_match_idx >= 0).sum())
    duration = float(sample.timestamps[-1] - sample.timestamps[0]) + 1.0 / args.fps
    ok = write_report(
        root / args.report_path,
        clip_id=clip_id,
        fps=args.fps,
        duration=duration,
        t_visual=int(sample.visual_feature.shape[0]),
        audio_matched=audio_matched,
        checks=checks,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())



