# Stage12 Downstream Compatibility Report

**Date**: 2025-08-26 (updated 2025-08-26)
**Review Scope**: Stage12 FLAMEFeature output interface → Stage13-17 data needs  
**Status**: ✅ Issues 1 & 2 resolved; Issue 3: Stage13 → independent GazeFeature

---

## 1. Executive Summary

Stage12 output interface is **compatible** with downstream stages after fixes:

| # | Issue | Severity | Status |

|---|-------|----------|--------|
| 1 | `timestamp` and `clip_id` missing from `FLAMEFeature` | 🔴 HIGH | ✅ RESOLVED |
| 2 | `read_stage12_outputs()` discards `cam` from DecaEncoder | 🟡 MEDIUM | ✅ RESOLVED |
| 3 | Stage13 input spec: `AlignmentInfo` vs `FLAMEFeature` | 🟡 MEDIUM | ✅ Stage13 → independent GazeFeature |

---
## 2. Current Stage12 Output Interface

### 2.1 FLAMEFeature Dataclass (models.py:607-670)

```python
@dataclass(frozen=True)
class FLAMEFeature:
    frame_index: int                              # ✅
    shapecode: Tuple[float, ...]                  # [300]
    expcode: Tuple[float, ...]                    # [100]
    jawpose: Tuple[float, ...]                    # [3]
    globalpose: Tuple[float, ...]                 # [3]
    cam: Tuple[float, ...]                        # [3]
    eye_pose: Tuple[float, ...] = (0,)*6          # [6] placeholder
    texcode: Optional[Tuple[float, ...]] = None   # [50] unused
    lightcode: Optional[Tuple[float, ...]] = None # [27] unused
    spectre_jawpose: Optional[Tuple[float, ...]] = None  # [3] audit

    @property
    def posecode(self) -> Tuple[float, ...]:      # [6] = globalpose + jawpose
```

### 2.2 Runner Output Dict (runner.py:250-260)

```python
result = {
    "frame_index": face["frame_index"],
    "timestamp": face["timestamp"],        # ← in runner, NOT in FLAMEFeature
    "clip_id": face["clip_id"],            # ← in runner, NOT in FLAMEFeature
    "expression_code": emica_exp.tolist(),
    "jaw_pose": fused_jaw.tolist(),
    "head_pose": emica_global.tolist(),
    "eye_pose": [0.0] * 6,
    "spectre_jawpose": spectre_jaw.tolist(),
    "shapecode": [0.0] * 300,
}
```

### 2.3 Subprocess Bridge (integration.py:145-157)

```python
feature = FLAMEFeature(
    frame_index=int(data["frame_index"][i]),
    shapecode=tuple(data["shapecode"][i].tolist()),
    expcode=tuple(data["expression_code"][i].tolist()),
    jawpose=tuple(data["jaw_pose"][i].tolist()),
    globalpose=tuple(data["head_pose"][i].tolist()),
    cam=tuple([0.0] * 3),                # ← HARDCODED, EMICA cam is discarded
    eye_pose=tuple(data["eye_pose"][i].tolist()),
    spectre_jawpose=tuple(data["spectre_jawpose"][i].tolist()),
)
# NOTE: timestamp and clip_id from npz are NOT passed to FLAMEFeature
```
---
## 3. Detailed Downstream Analysis

### 3.1 Stage13: Gaze Parameter Extraction

#### AISDD Spec

| Item | Spec |
|------|------|
| Input | `AlignmentInfo` (AlignedFaceInfo) |
| Output | `GazeFeature` |
| Model | MediaPipe Face Landmarker / Gaze360 |

#### GazeFeature Dataclass (models.py:674-688)

```python
class GazeFeature:
    frame_index: int
    left_eye: Tuple[float, float, float]      # [3]
    right_eye: Tuple[float, float, float]     # [3]
    gaze_vector: Tuple[float, ...]            # [6]
```

#### Issue 3: Input Spec Mismatch

AISDD says Stage13 input is `AlignmentInfo` (the aligned face image). However,
`eye_pose` is a **placeholder field on FLAMEFeature**, implying Stage13 should
modify FLAMEFeature or produce something that merges into it.

**Option A: Modify FLAMEFeature.eye_pose directly**
- ❌ Violates immutable dataclass principle (AISDD 2.4.1)
- ❌ Stage13 would need to return a new FLAMEFeature with updated eye_pose
- ❌ Stage13 would need knowledge of all FLAMEFeature fields

**Option B: Independent GazeFeature** ✅ RECOMMENDED
- ✅ Consistent with AISDD spec (Stage13 output = `GazeFeature`)
- ✅ Consistent with immutable dataclass principle
- ✅ `GazeFeature.gaze_vector[6]` maps 1:1 to `FLAMEFeature.eye_pose[6]`
- ✅ Stage17 handles the merge: `TrainingSample.gaze_feature = gaze_vector`
- ✅ Stage13 only needs frame images (AlignedFaceInfo), not FLAME parameters

**Recommendation**: Stage13 takes `List[AlignedFaceInfo]` as input, produces
`List[GazeFeature]`. Stage17 merges them. FLAMEFeature.eye_pose remains
zeros(6) placeholder.

#### Dimension Compatibility

| FLAMEFeature.eye_pose | GazeFeature.gaze_vector | Match? |
|---|---|---|
| [6] | [6] | ✅ Compatible |
### 3.2 Stage14: Audio Feature Extraction

#### AISDD Spec

| Item | Spec |
|------|------|
| Input | `CorrectedAudioInfo` (from Stage 7.1) |
| Output | `AudioFeature` |
| Model | wav2vec2 / HuBERT |
| Feature dim | [T, 768] |

#### AudioFeature Dataclass (models.py:691-706)

```python
class AudioFeature:
    timestamp: float              # corrected audio timestamp
    feature: Tuple[float, ...]    # [768] embedding
    dimension: int                # 768
    source_audio: Path
```

#### Issue 1a: timestamp missing from FLAMEFeature 🔴

**Problem**: `AudioFeature` uses `timestamp` (float, seconds) for time
alignment. `FLAMEFeature` only has `frame_index` (int). For Stage17 to align
FLAME + Audio, it needs a common time reference.

**Evidence**:
- Runner saves `timestamp` in npz ✅
- `FLAMEFeature` dataclass has **no timestamp field** ❌
- `read_stage12_outputs()` does **not pass timestamp** ❌

**Impact**: Stage17 cannot directly align FLAME features with audio features
without external FPS metadata. While AISDD says FLAME aligns by frame_index
and Audio aligns by timestamp, having both on FLAMEFeature makes alignment
more robust and avoids FPS dependency.

#### Issue 1b: clip_id missing from FLAMEFeature 🔴

**Problem**: `FLAMEFeature` has no `clip_id` field. For multi-clip processing,
Stage17 needs to know which clip each FLAMEFeature belongs to for traceability
and sample_id generation.

**Evidence**:
- Runner saves `clip_id` in npz and metadata ✅
- `FLAMEFeature` dataclass has **no clip_id field** ❌
- `read_stage12_outputs()` does **not pass clip_id** ❌

**Impact**: `TrainingSample.sample_id` generation and traceability chain
(video_id → scene_id → clip_id → frame_index → sample_id) is broken at the
clip_id link.

#### Recommendation

Add `timestamp: float` and `clip_id: str` to `FLAMEFeature` dataclass, and
update `read_stage12_outputs()` to pass them through. The npz already contains
both fields — no changes needed to the runner or inference pipeline.
### 3.3 Stage15/16: Statistics & Annotation

**Stage15 (Quality Statistics)**: Input = Pipeline Metadata (cross-stage).
Output = `StatisticsInfo`. No FLAMEFeature consumption. ✅ No impact.

**Stage16 (Data Annotation)**: Input = Video Metadata. Output =
`AnnotationInfo`. No FLAMEFeature consumption. ✅ No impact.

### 3.4 Stage17: Parameter Alignment & Fusion

#### AISDD Spec

| Input | Source |
|-------|--------|
| FLAMEFeature | Stage 12 |
| GazeFeature | Stage 13 |
| AudioFeature | Stage 14 |
| Metadata | Stage 15/16 |

#### TrainingSample Dataclass (models.py:755-775)

```python
class TrainingSample:
    sample_id: str
    video_clip: Path
    face_sequence: Path
    flame_feature: Tuple[float, ...]      # expcode + posecode
    gaze_feature: Tuple[float, ...]        # 6 dims
    audio_feature: Tuple[float, ...]
    metadata: Tuple[Tuple[str, str], ...]
```

#### Field Mapping

| TrainingSample field | Source | Dims | Status |
|---|---|---|---|
| `flame_feature` | `FLAMEFeature.expcode + posecode` | 100+6=106 | ✅ |
| `gaze_feature` | `GazeFeature.gaze_vector` | 6 | ✅ (via Stage13) |
| `audio_feature` | `AudioFeature.feature` | 768 | ✅ (Stage14) |

**Note**: `eye_pose` goes to `gaze_feature` (via Stage13), NOT to
`flame_feature`. This is correct per AISDD spec.

**Future shapecode**: If MICA is enabled, `shapecode[300]` would need to be
added to `flame_feature`. `Tuple[float, ...]` is flexible enough. ✅ No change.
---
## 4. Issue 2: cam Data Loss 🟡

### Problem

`read_stage12_outputs()` hardcodes `cam=tuple([0.0]*3)` instead of reading
the actual EMICA DecaEncoder camera output.

### Root Cause

The runner.py does NOT save `cam` in the npz. The EMICA DecaEncoder predicts
`cam[3]` (orthographic camera: scale, tx, ty), but the runner result dict
only saves: expression_code, jaw_pose, head_pose, eye_pose, spectre_jawpose,
shapecode, frame_index, timestamp, clip_id.

### Impact

- If DECA is enabled in the future (AISDD 3.18.3: DECA = "optional, default
  off"), the camera parameters would be lost.
- Currently minimal impact since DECA is off and cam is not used in
  TrainingSample.

### Recommendation

If DECA is planned for future use, add `cam` to the runner result dict, npz,
and `read_stage12_outputs()`. Otherwise, document that `cam` is always
`[0,0,0]` when DECA is disabled.

---

## 5. Complete Field Compatibility Matrix

| FLAMEFeature Field | Stage13 | Stage14 | Stage17 | Status |
|---|---|---|---|---|
| `frame_index` | ✅ needed | ✅ needed | ✅ needed | ✅ OK |
| `timestamp` | — | 🔴 needed | ✅ needed | 🔴 MISSING |
| `clip_id` | — | — | 🔴 needed | 🔴 MISSING |
| `expcode[100]` | — | — | ✅ flame_feature | ✅ OK |
| `jawpose[3]` | — | — | ✅ posecode | ✅ OK |
| `globalpose[3]` | — | — | ✅ posecode | ✅ OK |
| `posecode[6]` | — | — | ✅ flame_feature | ✅ OK |
| `eye_pose[6]` | ✅ → gaze | — | — | ✅ OK |
| `shapecode[300]` | — | — | ⚪ future | ✅ OK |
| `cam[3]` | — | — | ⚪ future | 🟡 HARDCODED |
| `spectre_jawpose[3]` | — | — | ⚪ audit | ✅ OK |
| `texcode[50]` | — | — | — | ✅ (None) |
| `lightcode[27]` | — | — | — | ✅ (None) |

---

## 6. Data Flow Diagram

### Current (with issues)

```
Stage12 runner → npz{frame_index, timestamp, clip_id, expcode, ...}
    ↓
read_stage12_outputs() → FLAMEFeature{frame_index, expcode, ...}
                          ❌ timestamp LOST
                          ❌ clip_id LOST
                          ❌ cam HARDCODED [0,0,0]
    ↓
Stage17: needs timestamp & clip_id → ❌ CANNOT ACCESS
```

### Recommended (fixes only)

```
Stage12 runner → npz{frame_index, timestamp, clip_id, expcode, ...}
    ↓
read_stage12_outputs() → FLAMEFeature{frame_index, timestamp, clip_id, ...}
                          ✅ timestamp PRESERVED
                          ✅ clip_id PRESERVED
    ↓
Stage17: has timestamp & clip_id → ✅ CAN ALIGN
```

---

## 7. Recommendations Summary

### 7.1 Required Changes (before Stage13/14/17 implementation)

| # | Change | File(s) | Priority |
|---|--------|---------|----------|
| 1 | Add `timestamp: float` field to `FLAMEFeature` | `models.py` | 🔴 P0 |
| 2 | Add `clip_id: str` field to `FLAMEFeature` | `models.py` | 🔴 P0 |
| 3 | Update `read_stage12_outputs()` to pass timestamp & clip_id | `integration.py` | 🔴 P0 |
| 4 | Document `cam` hardcoding decision | `integration.py` | 🟡 P1 |

### 7.2 Design Decision Required

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Stage13 input | A) `FLAMEFeature` (modify eye_pose) | B ✅ |
| | B) `AlignedFaceInfo` (independent GazeFeature) | Consistent with AISDD |

### 7.3 Changes NOT Required

- **No changes to runner.py**: npz already contains all needed fields
- **No changes to Stage12 inference**: backend output is complete
- **No changes to config**: config structure is adequate
- **No changes to TrainingSample**: flexible enough for current and future use
- **No new fields needed for Stage15/16**: they don't consume FLAMEFeature

---

## 8. Conclusion

Stage12 output is **fundamentally sound** for downstream consumption. The
3 issues identified are all in the **data transfer layer** (integration.py)
and the **data model** (FLAMEFeature dataclass), not in the inference backend.

The runner produces all necessary data; the gap is that `read_stage12_outputs()`
discards `timestamp` and `clip_id` when constructing `FLAMEFeature` objects,
and the dataclass itself has no slots for them.

**Overall assessment**: ⚠️ MINOR FIXES NEEDED — 2 fields to add to dataclass,
1 function to update. No redesign required.
| 3 | Stage13 input spec mismatch: `AlignmentInfo` vs `FLAMEFeature` | 🟡 MEDIUM | ✅ Stage13 → independent GazeFeature |