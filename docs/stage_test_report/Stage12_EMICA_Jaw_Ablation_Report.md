# Stage12 EMICA-Jaw Ablation Report

> **✅ 已正式采用 (ADOPTED)** — 2026-08-31 该 EMICA-jaw 方案经 FLAME mesh 重建
> 验证后正式成为**最终生产方案**，并已重新生成生产输出
> （`outputs/stage0_17_e2e/dataset_validation/`、`outputs/stage17_multimodal_export/`，
> `jaw_source=emica`）。本报告为当时对照实验记录，保留作为历史；最终方案见
> `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。

**Status:** Temporary experiment only — jaw_pose source switched from SPECTRE (production Scheme D) to EMICA for FLAME-mesh comparison. **Not a production decision.** Production outputs (`outputs/stage0_17_e2e/dataset_validation/`, `outputs/stage17_multimodal_export/`) were NOT modified.

- Clips: 11 (CNFACE_00001_BV18Qg96YE1W_000000)
- Ablation Stage12/13/14/17: `outputs/ablation_emica_jaw/`
- Ablation export: `outputs/stage17_multimodal_export_emica_jaw/`
- Driver: `scripts/run_emica_jaw_ablation_1214.sh` + `scripts/run_emica_jaw_ablation_17.py`; validation: `scripts/validate_emica_jaw_ablation.py`

## 1. Source matrix (unchanged except jaw)

| Modality | Production | Ablation |
|---|---|---|
| expression | EMICA | EMICA (identical) |
| head_pose | EMICA global | EMICA global (identical) |
| jaw_pose | **SPECTRE** | **EMICA** |
| gaze | MediaPipe | MediaPipe (identical) |
| audio | Chinese HuBERT | Chinese HuBERT (identical) |

## 2. Consistency checks (per clip, all PASS)

| Check | Meaning |
|---|---|
| T / frame_index / timestamp same | same temporal grid as production |
| expression / head / eye identical | max diff = 0 vs production |
| prod jaw == spectre_jawpose | production really is SPECTRE jaw |
| ablation jaw == emica_jawpose | ablation really is EMICA jaw |
| gaze / audio identical | Stage13/14 unchanged |
| export: audio & exp/head/gaze identical; jaw block differs | Stage17 only carries the jaw change |
| no NaN/Inf anywhere | numerical sanity |

## 3. Jaw statistics (rad, all clips combined)

| Metric | SPECTRE (production) | EMICA (ablation) |
|---|---|---|
| mean | 0.0090 | 0.0311 |
| std | 0.0203 | 0.0677 |
| min / max | -0.0476 / 0.0861 | -0.0745 / 0.2491 |
| mean\|jaw\| | 0.0170 | 0.0596 |
| velocity std (frame-to-frame) | 0.0085 | 0.0145 |
| rest-bias (mean min\|jaw\| / axis) | 0.0001 | 0.0023 |

- global max |SPECTRE - EMICA| jaw diff = 0.2327 rad (13.33 deg)

## 4. Per-clip jaw diff (SPECTRE vs EMICA)

| clip | T | max \|Δjaw\| rad | max \|Δjaw\| deg |
|---|---|---|---|
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0000 | 191 | 0.2327 | 13.33 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001 | 200 | 0.2327 | 13.33 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0002 | 200 | 0.2014 | 11.54 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0003 | 200 | 0.1940 | 11.11 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0004 | 200 | 0.2147 | 12.30 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000 | 200 | 0.2282 | 13.08 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0001 | 200 | 0.2046 | 11.72 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0002 | 200 | 0.2046 | 11.72 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0003 | 200 | 0.2026 | 11.61 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0004 | 200 | 0.2157 | 12.36 |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0005 | 200 | 0.2157 | 12.36 |

## 5. Note

This modification is used only for the mesh comparison experiment and does NOT represent the final production scheme. Production remains `jaw_source=spectre` (config default).
