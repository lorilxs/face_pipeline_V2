# HuBERT_Model_Compatibility_Report.md

# HuBERT 模型兼容性报告

Version: v1.0 | Date: 2026-08-26 | Author: AI Coding Agent

---

## 1. Model File Analysis

| Property | Value |
|---|---|
| Path | `models/hubert/hubert_base_ls960.pt` |
| Size | 1,136,468,879 bytes (~1.1 GB) |
| File Type | `Zip archive data` (PyTorch serialized) |
| Model Type | **fairseq HuBERT Base** (LibriSpeech 960h) |

**NOT** a HuggingFace Transformers checkpoint. Confirmed: `torch.load()` fails with `ModuleNotFoundError: No module named 'fairseq'`.

### Checkpoint Structure (inferred)

```
{'cfg': {'model': {'_name': 'hubert', 'encoder_layers': 12,
  'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072,
  'encoder_attention_heads': 12}}, 'task': {...}, 'model': {...}}
```

---

## 2. Loading Methods

### Method A: fairseq (Recommended)

```python
from fairseq import checkpoint_utils
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    ['models/hubert/hubert_base_ls960.pt'])
model = models[0].eval()
```

- **Dep**: `fairseq` (NOT installed)
- **Pros**: Direct, no conversion
- **Cons**: fairseq heavy (~500MB), may conflict

### Method B: HuggingFace Transformers (Conversion Required)

```python
from transformers import HubertModel
model = HubertModel.from_pretrained('models/hubert_hf/')
```

- **Dep**: `transformers` (v4.22.2 ✅), `fairseq` (for conversion)
- **Cons**: HF v4.22.2 may not support all fairseq features

### Method C: Existing av_hubert (third_party)

```python
from avhubert import AVHubertSeq2Seq
```

- **Dep**: `fairseq`, `avhubert` from third_party
- **Cons**: AV-HuBERT is audio-visual, overkill for audio-only
---

## 3. Model Architecture

| Component | Config |
|---|---|
| Type | HuBERT Base |
| CNN Feature Extractor | 7 conv layers, stride=(5,2,2,2,2,2,2) |
| Encoder Layers | 12 (Transformer) |
| Encoder Embed Dim | **768** |
| FFN Embed Dim | 3072 |
| Attention Heads | 12 |
| Output Rate | 50 Hz (20ms per frame) |

### 3.1 Input Requirements

| Parameter | Value |
|---|---|
| Sample Rate | **16000 Hz** |
| Format | Raw waveform float32 tensor |
| Shape | `[1, T]` or `[T]` (mono) |
| Compatible with Stage 1 | ✅ (already outputs 16kHz) |

### 3.2 Output

| Parameter | Value |
|---|---|
| Embedding Dim | **768** |
| Output Rate | 50 Hz |
| Output Shape | `[T', 768]`, T' ≈ T / 320 |

---

## 4. Inference API (Proposed)

```python
class HuBERTProcessor:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path

    def load_model(self):
        from fairseq import checkpoint_utils
        models, self.cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
            [self.checkpoint_path])
        self.model = models[0].to(self.device).eval()

    def extract_features(self, audio_path: Path, target_fps: int = 25):
        """Extract 768-dim embeddings → resample to target_fps."""
        ...
```

### Feature Extraction Pipeline

```
Corrected Audio (16kHz WAV) → torchaudio.load → [1, T]
 → HuBERT model → [T_h, 768] (50 Hz)
 → resample to 25 FPS → AudioFeature per frame [N, 768]
```

---

## 5. Environment Compatibility

| Package | Status |
|---|---|
| Python 3.10 | ✅ |
| PyTorch + CUDA | ✅ |
| `transformers` 4.22.2 | ✅ installed |
| `torchaudio` | ✅ (audio I/O) |
| `fairseq` | ❌ **NOT installed** |

### Dependency Gap: fairseq

```bash
pip install fairseq
# or from source:
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq && pip install -e .
```

---

## 6. wav2vec2 vs HuBERT

| Property | wav2vec2 (old) | HuBERT (new) |
|---|---|---|
| Framework | HF Transformers | fairseq |
| Embedding Dim | 768 | 768 ✅ same |
| Sample Rate | 16000 | 16000 ✅ same |
| Training Data | LS-960h | LS-960h |
| Objective | Contrastive+Diversity | Masked cluster prediction |
| Loading | `Wav2Vec2Model.from_pretrained()` | `checkpoint_utils.load_...()` |

---

## 7. Conclusion

| Verdict |
|---|
| Model: fairseq HuBERT Base |
| Embedding Dim: 768 → AudioFeature unchanged |
| Sample Rate: 16000 Hz → Stage 1 audio compatible |
| Dependency: Need `fairseq` installed |
| Stage17 Compat: ✅ (same 768-dim output) |
| AudioFeature Compat: ✅ (interface unchanged) |

---

## End of HuBERT_Model_Compatibility_Report.md v1.0