# Stage14_Design_Update_Report.md

# Stage14 Audio Feature Design Update Report

## wav2vec2 → HuBERT Migration

Version: v1.0 | Date: 2026-08-26 | Author: AI Coding Agent

---

## 1. Change Summary

| Aspect | Before | After |
|---|---|---|
| Audio Model | wav2vec2 (`facebook/wav2vec2-base-960h`) | HuBERT (`models/hubert/hubert_base_ls960.pt`) |
| Framework | HuggingFace Transformers | fairseq (original) |
| Loading | `Wav2Vec2Model.from_pretrained()` | `checkpoint_utils.load_model_ensemble_and_task()` |
| Embedding Dim | 768 | 768 (**unchanged**) |
| Sample Rate | 16000 | 16000 (**unchanged**) |
| AudioFeature Interface | `timestamp`, `feature[768]`, `dimension`, `source_audio` | **unchanged** |
| Stage17 Input | `audio_feature[768]` | **unchanged** |

---

## 2. Rationale

1. **HuBERT outperforms wav2vec2** on speech representation tasks (SUPERB benchmark)
2. **Model weights already available** at `models/hubert/hubert_base_ls960.pt` (~1.1GB), confirmed identical to old pipeline
3. **Consistent with downstream** — TalkingHead/Inferno uses HuBERT-based audio features
4. **Same embedding dimension (768)** — zero downstream impact

---

## 3. Architecture Changes

### 3.1 Data Flow (unchanged)

```
Stage 7.1 (AVAlignmentInfo)
    │  corrected_audio_path (16kHz WAV)
    ↓
Stage 14 (HuBERT Feature Extraction)
    │  AudioFeature per frame: timestamp, feature[768], dimension, source_audio
    ↓
Stage 17 (Parameter Alignment & Fusion)
    │  audio_feature = AudioFeature.feature
    ↓
TrainingSample
```

### 3.2 Stage14 Input

| Field | Source | Description |
|---|---|---|
| `corrected_audio_path` | Stage 7.1 `AVAlignmentInfo` | 16kHz WAV, A/V sync corrected |

**Input confirmed**: Stage14 must consume Stage 7.1's corrected audio, NOT raw audio.

### 3.3 HuBERT Feature Extraction Pipeline

```
Corrected Audio (16kHz WAV) → torchaudio.load → waveform [1, T]
 → HuBERT (fairseq) → features [T_h, 768] (50 Hz)
 → resample to 25 FPS → List[AudioFeature] per frame
 → save as NPZ per clip
```
---

## 4. Data Interface — NO CHANGE

### AudioFeature (unchanged)

```python
@dataclass(frozen=True)
class AudioFeature:
    timestamp: float              # corrected audio timestamp (sec)
    feature: Tuple[float, ...]    # [768] HuBERT embedding
    dimension: int                # 768
    source_audio: Path            # corrected audio path
```

### Stage17 Compatibility

```
TrainingSample.audio_feature = AudioFeature.feature  # [768] ndarray
```

Stage17 only reads `audio_feature` as a 768-dim vector. It does NOT care whether the backend is wav2vec2 or HuBERT. **Interface fully compatible.**

---

## 5. Files Modified

### 5.1 Design Documents (AISDD)

| File | Change | Lines |
|---|---|---|
| `docs/00_Project_Overview.md` | `wav2vec2` → `HuBERT` | 634 |
| `docs/02_Data_Model_and_Interface.md` | `wav2vec2/HuBERT输出` → `HuBERT输出` | 910 |
| `docs/03_Pipeline_Stage_Specification.md` | Remove duplicate, keep `HuBERT` | 1684-1685 |
| `docs/04_Config_and_Model_Management.md` | GPU table: `wav2vec2` → `HuBERT` | 864 |
| `docs/04_Config_and_Model_Management.md` | Model path: `wav2vec2 weights` → `HuBERT weights` | 741 |
| `docs/06_Engineering_Guideline.md` | Wrapper: `wav2vec2` → `HuBERT` | 402 |

### 5.2 Configuration

| File | Change | Lines |
|---|---|---|
| `configs/config.yaml` | `model: wav2vec2` → `backend: hubert` + `model_path` | 282-284 |

### 5.3 Data Model

| File | Change | Lines |
|---|---|---|
| `face_pipeline_v2/data/models.py` | Docstring: `wav2vec2` → `HuBERT` | 279, 711 |

### 5.4 New Analysis Documents

| File | Purpose |
|---|---|
| `docs/Stage14_HuBERT_Migration_Analysis.md` | Comprehensive migration analysis |
| `docs/HuBERT_Model_Compatibility_Report.md` | Model file analysis and loading |
| `docs/Stage14_Design_Update_Report.md` | This report |

---

## 6. Files NOT Modified

| File | Reason |
|---|---|
| `docs/01_System_Architecture.md` | No wav2vec2-specific text |
| `docs/05_Data_Flow_and_Output.md` | Stage14 described generically |
| `face_pipeline_v2/data/models.py` (AudioFeature fields) | Interface unchanged |
| `PROJECT_CONTEXT.md` | Not part of AISDD docs |
| `DEVELOPMENT_LOG.md` | Historical log |
| `stages/stage_14_audio/` | **Not yet implemented** |

---

## 7. Stage14 Implementation Plan (Next Steps)

### 7.1 Prerequisites

1. ✅ HuBERT weights confirmed (`models/hubert/hubert_base_ls960.pt`)
2. ❌ `fairseq` must be installed
3. ✅ `torchaudio` available
4. ✅ `transformers` 4.22.2 available

### 7.2 Implementation Order

| Step | Task | Files |
|---|---|---|
| 1 | Create `stages/stage_14_audio/` | `__init__.py`, `processor.py`, `integration.py`, `runner.py`, `stage.py` |
| 2 | Implement `HuBERTProcessor` (fairseq) | `processor.py` |
| 3 | Implement `Stage14Audio` (BaseStage) | `stage.py` |
| 4 | Implement NPZ save/read | `integration.py` |
| 5 | Add `stage14` config | `configs/config.yaml` |
| 6 | Write unit tests | `tests/test_stage14_audio_*.py` |
| 7 | Integrate into pipeline runner | `pipeline/pipeline_runner.py` |

### 7.3 Key Rules

- **Input**: `AVAlignmentInfo.corrected_audio_path` (16kHz WAV, mono)
- **Output**: `List[AudioFeature]` per clip, saved as NPZ
- **Resampling**: 50 Hz → 25 FPS for FLAME alignment
- **No modification to Stage12/13 outputs**
- **AudioFeature.feature is the final 768-dim vector for Stage17**

---

## 8. Stage17 Interface Compatibility (Verified)

| Check | Status |
|---|---|
| AudioFeature.feature dim = 768 | ✅ Unchanged |
| AudioFeature.timestamp exists | ✅ Unchanged |
| AudioFeature.dimension exists | ✅ Unchanged |
| AudioFeature.source_audio exists | ✅ Unchanged |
| Stage17 reads `audio_feature` | ✅ No change needed |
| NPZ serialization format | ✅ Same [N,768] shape |

---

## End of Stage14_Design_Update_Report.md v1.0