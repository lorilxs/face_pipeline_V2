# Stage12 Pipeline End-to-End Integration Report

**Date**: 2026-08-25  
**Status**: ✅ Complete  
**Tests**: 18/18 passed

---

## 1. Pipeline Call Chain

```
Stage 0  (Video Integrity Check)
   ↓
Stage 1  (AV Standardization)
   ↓
Stage 2  (Scene Segmentation)
   ↓
...
   ↓
Stage 11 (Face Alignment)
   ↓   List[AlignedFaceInfo]
Stage 12 (FLAME Extraction)  ← NEW
   ↓   List[FLAMEFeature]
Stage 13 (Gaze)  [future]
```

---

## 2. Stage12 Runtime Environment

**Problem**: EMICA/SPECTRE depend on `torch` (CUDA), `pytorch-lightning`, `inferno`, `omegaconf`, `yacs`, `kornia` — none available in host Python.

**Solution**: Docker Subprocess with `emica_env`.

```
Host Python (Stage 0-11)
    │
    └── docker exec face_pipeline_lxs bash -c
            "source /workspace/emica_env/bin/activate &&
             python stage12_runner.py"
```

Two runtime modes:
---

## 3. EMICA / SPECTRE Loading

- **EMICA**: `inferno.models.FaceReconstruction.FaceRecBase`, `EmicaEncoder` with `ExpressionEncoder` (ResNet50)
  - Outputs: `expcode` [100], `jawpose` [3], `globalpose` [3]
- **SPECTRE**: `spectre.src.spectre.SPECTRE`, `E_flame` encoder
  - Outputs: `pose` [6] (head_pose[3] + jaw_pose[3])
- **Jaw Fusion**: `fused = (1-λ)*EMICA.jaw + λ*SPECTRE.jaw`, λ=0.6

---

## 4. Modified Files

| File | Type | Description |
|---|---|---|
| `configs/config.yaml` | Modified | Added `runtime_mode`, `batch_size`, `device`, `output_dir` |
| `stage_12_flame/stage.py` | Modified | Added `_process_subprocess()` and `_process_direct()` |
| `stage_12_flame/runner.py` | **Created** | Standalone CLI runner for Docker |
| `stage_12_flame/integration.py` | **Created** | Subprocess bridge (serialize/invoke/deserialize) |
| `tests/test_stage12_pipeline_integration.py` | **Created** | 18 integration tests |
| `docs/Stage12_Runtime_Environment_Analysis.md` | **Created** | Environment analysis |
| `docs/Stage12_Pipeline_Integration_Report.md` | **Created** | This report |

### Unchanged
- `processor.py`, `emica.py`, `spectre.py`, `models.py` — no changes

---

## 5. Test Results

**18/18 tests passed** (2.76s):
- 6 dimension tests: expcode[100], jawpose[3], globalpose[3], eye_pose[6], shapecode[300], spectre_jawpose[3]
- 4 fusion tests: λ=0.6, λ=0, λ=1, batch processing (20 frames)
- 8 validation tests: input validation, output validation, empty handling

**Previous real-data validation**: 50/50 frames, 0 failures, no NaN/Inf

---

## 6. Output File Structure

```
outputs/stage12_flame/{clip_id}/
    flame_features.npz    # all arrays
    flame_metadata.json   # metadata + frame indices
    expression.npy        # [T, 100]
    jaw_pose.npy          # [T, 3]
    head_pose.npy         # [T, 3]
    eye_pose.npy          # [T, 6]
```

---

## 7. Current Limitations

| Limitation | Status | Plan |
|---|---|---|
| MICA not loaded | `shapecode = zeros(300)` | Future: add MICA shape fitting |
| Eye pose zeros | `eye_pose = zeros(6)` | Stage 13 (Gaze) placeholder |
| No FLAME decoder | Encoder-only | Future: renderer for visualization |
| No temporal smoothing | Per-frame output | Future: Stage 15 post-processing |
| Requires Docker | `face_pipeline_lxs` must be running | OK for current deployment |

---

## 8. Next Steps

1. Run subprocess mode on real Stage11 output (requires Docker running)
2. Add MICA shape fitting for non-zero shapecode
3. Integrate Stage 13 (Gaze) for non-zero eye_pose
4. Add Stage12 to CI pipeline
5. Test different `jaw_fusion_lambda` values
- `"subprocess"` (default): Docker subprocess
- `"direct"`: In-process via ModelManager