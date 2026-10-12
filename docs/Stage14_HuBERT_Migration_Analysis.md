# Stage14_HuBERT_Migration_Analysis.md

# Stage14 Audio Model Migration: wav2vec2 → HuBERT

## Analysis Report

Version: v1.0
Date: 2026-08-26
Author: AI Coding Agent

---

## 1. Current wav2vec2 Design Locations

### 1.1 Design Documents (AISDD 00-06)

| Document | Line(s) | Current wav2vec2 Text | Context |
|---|---|---|---|
| `00_Project_Overview.md` | 634 | `- wav2vec2` | Audio Feature Extraction 模型列表 |
| `03_Pipeline_Stage_Specification.md` | 1684 | `- wav2vec2` | Stage 14 Model 列表 |
| `04_Config_and_Model_Management.md` | 864 | `Stage 14 \| wav2vec2` | GPU 资源并行表 (Feature Extraction Parallel) |
| `04_Config_and_Model_Management.md` | ~740 | `wav2vec2 weights` | Audio 模型路径表 |
| `06_Engineering_Guideline.md` | 402 | `wav2vec2 \| AudioFeatureExtractor` | Model Wrapper 注册表 |

### 1.2 Configuration

| File | Line | Current Text |
|---|---|---|
| `configs/config.yaml` | 282 | `model: wav2vec2` |

### 1.3 Project Context & Dev Log

| File | Line(s) | Current Text |
|---|---|---|
| `PROJECT_CONTEXT.md` | 123 | `HuBERT \| 14 \| ... \| models/wrappers/wav2vec2.py 🔴 empty` |
| `PROJECT_CONTEXT.md` | 513 | `Wrapper stub: face_pipeline_v2/models/wrappers/wav2vec2.py (empty)` |
| `PROJECT_CONTEXT.md` | 540 | `EMICA, SPECTRE, and wav2vec2 wrappers need implementation` |
| `DEVELOPMENT_LOG.md` | 238 | `Wrapper stub: face_pipeline_v2/models/wrappers/wav2vec2.py（空，待实现）` |
| `DEVELOPMENT_LOG.md` | 782 | `Stage 14 — Audio feature (wav2vec2) 🔴 stub` |
| `DEVELOPMENT_LOG.md` | 790 | `Stage 14 Audio Feature (wav2vec2)` |

### 1.4 Data Model (NO CHANGE REQUIRED)

| File | Line | Current Text | Status |
|---|---|---|---|
| `02_Data_Model_and_Interface.md` | 910 | `feature \| 音频Embedding \| array \| wav2vec2/HuBERT输出` | Already mentions HuBERT — no change needed |
| `face_pipeline_v2/data/models.py` | 701-715 | `AudioFeature` dataclass with `timestamp`, `feature`, `dimension`, `source_audio` | Interface-stable — no change |

---

## 2. Locations Requiring Replacement

### 2.1 Must Replace (wav2vec2 → HuBERT)

| Priority | File | Line(s) | Old Text | New Text |
|---|---|---|---|---|
| **P0** | `configs/config.yaml` | 282 | `model: wav2vec2` | `backend: hubert` |
| **P1** | `00_Project_Overview.md` | 634 | `- wav2vec2` | `- HuBERT` |
| **P1** | `03_Pipeline_Stage_Specification.md` | 1684 | `- wav2vec2` | `- HuBERT` |
| **P1** | `04_Config_and_Model_Management.md` | 864 | `wav2vec2` | `HuBERT` |
| **P1** | `04_Config_and_Model_Management.md` | ~740 | `wav2vec2 weights` | `HuBERT weights` |
| **P1** | `06_Engineering_Guideline.md` | 402 | `wav2vec2` | `HuBERT` |

### 2.2 Should Update (wrapper file naming)

| Priority | File | Line(s) | Old Text | New Text |
|---|---|---|---|---|
| **P2** | `PROJECT_CONTEXT.md` | 123 | `models/wrappers/wav2vec2.py` | `models/wrappers/hubert.py` |
| **P2** | `PROJECT_CONTEXT.md` | 513 | `wav2vec2.py` | `hubert.py` |
| **P2** | `PROJECT_CONTEXT.md` | 540 | `wav2vec2 wrappers` | `HuBERT wrapper` |
| **P2** | `DEVELOPMENT_LOG.md` | 238 | `wav2vec2.py` | `hubert.py` |
| **P2** | `DEVELOPMENT_LOG.md` | 782 | `wav2vec2` | `HuBERT` |
| **P2** | `DEVELOPMENT_LOG.md` | 790 | `wav2vec2` | `HuBERT` |

### 2.3 No Change Required

| File | Reason |
|---|---|
| `02_Data_Model_and_Interface.md` | Already documents `wav2vec2/HuBERT` as dual options |
| `05_Data_Flow_and_Output.md` | No model-specific text — Stage 14 data flow described generically |
| `face_pipeline_v2/data/models.py` | `AudioFeature` interface is model-agnostic |
| `01_System_Architecture.md` | No wav2vec2-specific text |

---

## 3. HuBERT Corresponding Design

### 3.1 Model Identity

| Property | wav2vec2 (old) | HuBERT (new) |
|---|---|---|
| Model Name | `facebook/wav2vec2-base-960h` | `hubert_base_ls960` |
| Framework | HuggingFace Transformers | **fairseq** (original) |
| Checkpoint | HuggingFace Hub (auto-download) | `models/hubert/hubert_base_ls960.pt` (~1.1GB) |
| Input | 16kHz raw audio waveform | 16kHz raw audio waveform |
| Output Dim | 768 | 768 |
| Sample Rate | 16000 | 16000 |
| Architecture | Transformer + CNN feature extractor | Transformer + CNN feature extractor |

### 3.2 Loading Strategy

Since the checkpoint is in **fairseq format** and `fairseq` is NOT installed in the host Python environment, the Stage14 processor must:

1. **Install fairseq** as a dependency, OR
2. **Convert** the fairseq checkpoint to HuggingFace Transformers format (requires fairseq for conversion), OR
3. **Use existing av_hubert code** in `third_party/inferno/spectre/external/av_hubert/` (also depends on fairseq)

The current host environment has `transformers==4.22.2` but NOT `fairseq`.

### 3.3 Proposed Inference API

```python
# Stage14 processor will use fairseq to load the HuBERT checkpoint
# and extract 768-dim embeddings from 16kHz audio

def extract_hubert_features(
    audio_path: Path,
    model,
    cfg,
    device: str = "cuda"
) -> np.ndarray:  # [T, 768]
    """Extract HuBERT audio embeddings from corrected audio."""
    ...
```

### 3.4 Stage14 Data Flow (unchanged from original design)

```
Stage 7.1 (AVAlignmentInfo)
    │
    │  corrected_audio_path
    ↓
Stage 14 (HuBERT Feature Extraction)
    │
    │  AudioFeature per frame
    │  - timestamp: float
    │  - feature: Tuple[float, ...]   # [768]
    │  - dimension: 768
    │  - source_audio: Path
    ↓
Stage 17 (Parameter Alignment & Fusion)
    │
    │  audio_feature = AudioFeature.feature
    ↓
TrainingSample
```

---

## 4. Summary

| Metric | Value |
|---|---|
| Documents requiring modification | 6 (00, 03, 04, 06, PROJECT_CONTEXT, DEVELOPMENT_LOG) |
| config.yaml fields to change | 1 (`audio.model`) |
| Data model changes | 0 (AudioFeature unchanged) |
| Downstream interface changes | 0 (Stage17 sees same `audio_feature[768]`) |
| Model file to use | `models/hubert/hubert_base_ls960.pt` |
| Dependency to add | `fairseq` (for checkpoint loading) |

---

## End of Stage14_HuBERT_Migration_Analysis.md v1.0