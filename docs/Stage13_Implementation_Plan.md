# Stage13 Gaze Extraction — Implementation Plan

**Date**: 2026-08-26
**Status**: Step 1 — Plan

---

## 1. Data Flow

```
Stage11 AlignedFaceInfo (256×256 aligned face)
    │
    ├──→ Stage12 FLAMEFeature (existing, unchanged)
    │
    └──→ Stage13 GazeFeature (this implementation)
             │
             └──→ Stage17 TrainingSample (gaze_feature field)
```

**Input**: `List[AlignedFaceInfo]` from Stage11
- `aligned_face_path`: 256×256 aligned face image
- `frame_index`: frame index
- `timestamp`: time in seconds
- `clip_id`: clip identifier

**Output**: `List[GazeFeature]`
- `frame_index`: int
- `timestamp`: float
- `clip_id`: str
- `gaze_vector`: np.ndarray shape [6]

---

## 2. File Changes

### 2.1 Modify: `data/models.py`

**GazeFeature** — add `timestamp` and `clip_id` fields:

```python
@dataclass(frozen=True)
class GazeFeature:
    frame_index: int
    timestamp: float          # NEW
    clip_id: str              # NEW
    left_eye: Tuple[float, float, float]
    right_eye: Tuple[float, float, float]
    gaze_vector: Tuple[float, float, float, float, float, float]
```

### 2.2 Create: `stages/stage_13_gaze/processor.py`

- `get_gaze_vector_from_landmarks()` — transplanted from old pipeline
- `create_mediapipe_detector()` — MediaPipe FaceLandmarker init
- `extract_gaze_parameters()` — Main function: AlignedFaceInfo → GazeFeature

### 2.3 Create: `stages/stage_13_gaze/integration.py`

- `save_gaze_features()` — save GazeFeature list to .npz
- `read_gaze_features()` — read .npz back to GazeFeature list

### 2.4 Create: `stages/stage_13_gaze/runner.py`

- Standalone runner for subprocess mode (optional)

### 2.5 Create: `stages/stage_13_gaze/stage.py`

- `Stage13Gaze(BaseStage)` — Pipeline stage class

### 2.6 Modify: `configs/config.yaml`

- Add `stage13` section

### 2.7 Create: `tests/test_stage13_gaze_datamodel.py`

- GazeFeature field validation
- Default values
- shape=[6] check

### 2.8 Create: `tests/test_stage13_gaze_processor.py`

- Mock MediaPipe output
- Normal iris test
- Fallback (no iris) test
- Vector normalization test

### 2.9 Create: `tests/test_stage13_gaze_real_data.py`

- Real aligned_face images from Stage11 output
- 50-frame sample
- NaN/Inf/norm validation

---

## 3. Input / Output Schema

### Input (from `AlignedFaceInfo`)

| Field | Type | Source |
|-------|------|--------|
| `clip_id` | `str` | `AlignedFaceInfo.clip_id` |
| `frame_index` | `int` | `AlignedFaceInfo.frame_index` |
| `timestamp` | `float` | `AlignedFaceInfo.timestamp` |
| `aligned_face_path` | `Path` | `AlignedFaceInfo.aligned_face_path` |

### Output (`GazeFeature`)

| Field | Type | Source |
|-------|------|--------|
| `frame_index` | `int` | copied from input |
| `timestamp` | `float` | copied from input |
| `clip_id` | `str` | copied from input |
| `left_eye` | `Tuple[float, float, float]` | `gaze_0_x/y/z` |
| `right_eye` | `Tuple[float, float, float]` | `gaze_1_x/y/z` |
| `gaze_vector` | `Tuple[float]*6` | `(gaze_0_x, y, z, gaze_1_x, y, z)` |

---

## 4. Config

```yaml
# configs/config.yaml
stage13:
  enabled: true
  gaze:
    enabled: true
    model_path: models/mediapipe/face_landmarker.task
    delegate: cpu          # cpu | gpu
    batch_size: 8          # for future batching
    image_size: [256, 256] # aligned face size
```

---

## 5. Test Plan

### 5.1 Datamodel Tests
- `GazeFeature` fields exist and are correct types
- `gaze_vector` is tuple of 6 floats
- `left_eye` / `right_eye` are 3-tuples

### 5.2 Processor Tests (mocked MediaPipe)
- With iris landmarks → correct gaze calculation
- Without iris landmarks → fallback to face_normal
- Output vectors are normalized (L2 ≈ 1.0)
- Empty input → empty output

### 5.3 Real Data Test
- Input: `outputs/stage0_11_test_verify/aligned_faces/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001`
- 50 random frames
- Validate: no NaN, no Inf, norm ≈ 1.0