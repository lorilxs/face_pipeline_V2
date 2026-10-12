# Stage12 SPECTRE Jaw Adoption Report (Scheme D)

> **⚠️ 已被取代 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案
> （FLAME mesh 重建验证：SPECTRE jaw 版本嘴部区域存在扭曲，EMICA 版本效果更好）。
> 本报告保留作为历史记录；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。
> 生产代码 `jaw_pose` 现固定为 EMICA（`jaw_fusion_lambda` 已废弃）。

**Date:** 2026-08-29
**Status:** ✅ EXECUTED — production code changed, regenerated, and validated
**Decision (AISDD-approved):** `jaw_pose = SPECTRE` (pure, no fusion); keep
`head_pose = EMICA global` and `expression = EMICA`. `emica_jawpose` is
preserved on the output for debug/analysis.

---

## 1. Scope

| Component | Changed? | Note |
|---|---|---|
| Stage12 `jaw_pose` source | ✅ YES | `jaw_pose = spectre_jaw` (Scheme D, no fusion) |
| Stage12 `head_pose` source | ❌ NO | stays EMICA `globalpose` |
| Stage12 `expression` source | ❌ NO | stays EMICA `expcode` [100] |
| FLAME interface (`head [T,3] / jaw [T,3] / exp [T,100]`) | ❌ NO | unchanged |
| Stage13 / Stage14 / Stage17 | ❌ NO | untouched |
| FLAME decoder / mesh / pytorch3d | ❌ NO | untouched |
| Docker / emica_env | ❌ NO | untouched |
| Preprocessing normalization | ❌ NO | already `/255.0` in `runner.py:preprocess_image`, `emica.py`, `spectre.py` (no `/127.5-1.0` remains) |

---

## 2. What changed (file-by-file)

### Production code
| File | Change |
|---|---|
| `face_pipeline_v2/stages/stage_12_flame/runner.py` | Fusion → `fused_jaw = spectre_jaw` (was `(1-λ)·emica + λ·spectre`); result dict now also carries `emica_jawpose`; `save_outputs` writes `emica_jawpose` to npz + metadata (`jaw_source/head_source/expression_source`, `emica_jawpose_dim`, updated notes); `--jaw_fusion_lambda` default 1.0, help notes it is RETIRED/audit-only; **model loaders `_load_emica`/`_load_spectre` switched to the production wrappers** (`EMICAWrapper`/`SPECTREWrapper`) — see §6; repo root added to `sys.path` |
| `face_pipeline_v2/stages/stage_12_flame/processor.py` | Docstring updated (Scheme D); fusion → `fused_jaw = spectre_jaw`; `FLAMEFeature` now gets `emica_jawpose` |
| `face_pipeline_v2/data/models.py` | `FLAMEFeature` docstring updated (jaw source = SPECTRE); new optional field `emica_jawpose: Optional[Tuple[float,...]] = None` (mirrors `spectre_jawpose`; default → non-breaking) |
| `face_pipeline_v2/stages/stage_12_flame/integration.py` | `read_stage12_outputs` reads optional `emica_jawpose` (absent → `None`) |
| `face_pipeline_v2/stages/stage_12_flame/stage.py` | Module docstring updated (jaw_pose = SPECTRE, Scheme D) |
| `configs/config.yaml` | `stage12.jaw_fusion_lambda: 0.6 → 1.0` with comment (Scheme D; value now audit-only) |
| `scripts/stage0_17_e2e_stage12.py` | `_feature_to_result` now carries `emica_jawpose` (required by the new `save_outputs`) |

### Tests
| File | Change |
|---|---|
| `tests/test_stage12_processor.py` | `test_single_face_fusion` → `test_single_face_jaw_from_spectre` (expects jaw = SPECTRE, asserts `emica_jawpose` preserved); `test_lambda_zero_pure_emica_jaw` → `test_jaw_source_spectre_regardless_of_lambda` |
| `tests/test_stage12_pipeline_integration.py` | `test_fusion_lambda_0_6` / `test_fusion_lambda_0` rewritten to assert jaw = SPECTRE regardless of λ; `emica_jawpose` asserted |
| `tests/test_stage12_fusion.py` | Helper `_fuse_jaw` → `_stage12_jaw` (returns spectre_jaw); T1 parametrized tests now assert jaw == spectre for any λ; `test_fused_jaw_matches_formula` → `test_jaw_matches_stage12_design`; added `emica_jawpose` tests |
| `tests/test_datamodel.py` | Added `test_emica_jawpose_default_none`; `test_full_create_with_spectre` includes `emica_jawpose` |

---

## 3. Before vs After metrics (same 50 frames, 100–149)

Clip: `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001`

| Quantity | STALE pre-fix npz (before) | REGENERATED (after, Scheme D) |
|---|---|---|
| `jaw_pose` | mean **400.24°** std 6.06° max 416.88° | mean **1.619°** std **0.440°** max 2.792° (0.0283 rad mean) |
| `spectre_jawpose` | mean 669.06° (±7 rad artifact) | mean 1.619° std 0.440° (identical to jaw) |
| `head_pose` | mean 75.44° (implausible) | mean **9.514°** std 1.077° (plausible) |
| `emica_jawpose` | (not stored) | mean 8.250° std 2.907° — preserved for debug |
| NaN / Inf | 0 | 0 |
| jaw velocity | — | mean **0.556°**/frame, std 0.470°, max 2.411° |
| head velocity | — | mean 0.694°/frame, std 0.413° |

### Scheme-D identity & sanity checks (all pass)
- `jaw_pose == spectre_jawpose` → **max|diff| = 0.0 rad** (pure SPECTRE, no fusion)
- `jaw_pose` max magnitude = **0.049 rad < 1 rad** ✅
- **Bit-for-bit match vs validated D-preview reference** (`outputs/stage12_pose_validation/<clip>/D_jaw_pose.npy`, `D_head_pose.npy`): jaw max|diff| = **0.000000**, head max|diff| = **0.000000**
- Expression non-zero (mean −0.032, std 0.462) → real EMICA output, not zeros

---

## 4. Regeneration method

Ran the **modified production runner** inside `face_pipeline_lxs` (emica_env,
Docker) on frames 100–149 (50 frames, timestamps = frame_index/25.0):

```
python face_pipeline_v2/stages/stage_12_flame/runner.py \
  --input_metadata outputs/tmp/stage12_input_meta_<clip>.json \
  --output_dir outputs/stage12_flame \
  --emica_checkpoint  models/EMICA/.../model-epoch=38-val/....ckpt \
  --spectre_checkpoint third_party/inferno/spectre/spectre/data/deca_model.tar \
  --jaw_fusion_lambda 1.0 --batch_size 8 --device cuda --clip_id <clip>
```

Log: `[INFO] Inference complete: 50/50 frames processed`; wrote
`flame_features.npz` + `flame_metadata.json` + per-array `.npy`.
(CUDA not exposed via `docker exec`; runner fell back to CPU — values verified
bit-identical to the reference, so device does not affect results.)

---

## 5. Test results

| Suite | Result |
|---|---|
| `test_stage12_processor.py` + `test_stage12_pipeline_integration.py` + `test_stage12_fusion.py` + `test_stage12_output_metadata.py` | ✅ **51 passed** |
| `test_stage17_datamodel.py` + `test_stage17_fusion_processor.py` | ✅ **40 passed** |
| `test_datamodel.py` | ⚠️ 1 **pre-existing** failure (unrelated): `TestAllModelsImport::test_all_models_instantiable` — `GazeFeature` 4-arg vs 6-field (documented F1 in `docs/Stage12_14_DataFlow_Compatibility_Report.md`); not Stage12/FLAMEFeature, untouched |
| `read_stage12_outputs` smoke on regenerated npz | ✅ 50 features; `jaw==spectre_jawpose`; `emica_jawpose` present; posecode 6 / exp 100 / shape 300 |

### Downstream chain consistency (unchanged structure)
- Stage12: T=50, frames 100–149 · Stage13 gaze: T=50 · Stage14 audio: T=200
  (full clip, merged by timestamp) · Stage17 manifest: `num_frames=50,
  valid_length=50` (unchanged).
- Because the regenerated output is **bit-identical** to the previously
  validated Scheme-D preview, Stage17 end-to-end behavior is unchanged from
  the prior validation session.

---

## 6. Notes / issues found & fixed during implementation

1. **Pre-existing runner model-load bug (fixed).** `runner.py::_load_emica` /
   `_load_spectre` used the full-inferno `FaceReconstructionBase` / `SPECTRE`
   paths which pull in the MICA encoder (`loguru`, etc.) absent from emica_env
   and were never functional (this was already documented in
   `scripts/stage0_17_e2e_stage12.py` docstring). Both loaders now use the
   production wrappers (`EMICAWrapper` / `SPECTREWrapper`) — the same verified
   official route used by direct mode and the D-preview. This was required to
   regenerate the npz through the real runner CLI.
2. **`scripts/stage0_17_e2e_stage12.py::_feature_to_result`** updated to carry
   `emica_jawpose` (the new `save_outputs` requires the key).
3. **`jaw_fusion_lambda` is now vestigial** (value ignored by code) but kept on
   the interface / config / metadata for backward compatibility and audit. It
   is set to `1.0` in `config.yaml` so configs remain self-describing
   (Scheme D ≡ λ=1.0).
4. **`emica_jawpose` npz key** closes a previously documented schema gap
   (`scripts/stage12_pose_ablation_analyze.py` already expected this key).
   Adding it is non-breaking: downstream consumers read named keys only, and
   `read_stage12_outputs` treats it as optional.
5. **`tests/manual/test_stage12_real_validation.py`** (dev-only, not run by
   `pytest`) still demonstrates the old fusion formula; intentionally left
   unchanged as a historical reference harness.
6. **Latent CWD assumption (not fixed, out of scope):**
   `integration.invoke_stage12_runner` does not `cd` into the repo root before
   invoking the runner, while checkpoints are passed as repo-root-relative
   paths; this only works if the container CWD is the repo root. The
   regeneration used `cd /workspace/Face_Pipeline_V2` explicitly.

---

## 7. Revert

- Production: set `fused_jaw` back to `(1-λ)·emica_jaw + λ·spectre_jaw` in
  `runner.py` / `processor.py` and restore `configs/config.yaml`
  `jaw_fusion_lambda: 0.6` (the model-loader / metadata / npz additions are
  additive and non-breaking and can remain).
- Pre-edit originals are backed up at `/tmp/stage12_backup/`.

## 8. Files produced / validated

- `outputs/stage12_flame/<clip>/flame_features.npz` (regenerated, T=50)
- `outputs/stage12_flame/<clip>/flame_metadata.json` (jaw_source: spectre)
- `outputs/tmp/stage12_input_meta_<clip>.json` (regeneration input)


