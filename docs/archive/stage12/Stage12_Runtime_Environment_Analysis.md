# Stage12 Runtime Environment Analysis

**Date**: 2026-08-25  
**Status**: Complete

---

## 1. Current Pipeline Launch Method

### Host Machine

- **Hostname**: `304a02301.cloud.a05.am11`
- **Python**: `/opt/hb/bin/python3` (Python 3.10.12)
- **PyTorch**: 2.4.1+cpu
- **Stage0-11**: Run directly on host Python environment

### Docker Container

- **Container**: `face_pipeline_lxs`
- **EMICA Environment**: `/workspace/emica_env/bin/activate`
- **Needed for**: EMICA, SPECTRE, and all their dependencies

### Entry Point

The pipeline runner file is `face_pipeline_v2/pipeline/pipeline_runner.py` (currently empty — pipeline is invoked via test scripts and stage-level scripts).

---

## 2. Python Path

| Environment | Python Path | PyTorch | Can Import EMICA? |
|---|---|---|---|
| Host | `/opt/hb/bin/python3` | 2.4.1+cpu | ❌ No |
| Docker (emica_env) | `/workspace/emica_env/bin/python` | TBD (CUDA) | ✅ Yes |

---

## 3. Stage12 Dependencies

### EMICA Dependencies
- `torch` (CUDA)
- `pytorch-lightning`
- `inferno` (bundled in `third_party/inferno/inferno-master`)
- `omegaconf`
- `yacs`
- `kornia`
- `numpy`, `opencv-python`

### SPECTRE Dependencies
- `torch` (CUDA)
- `inferno` (bundled in `third_party/inferno/spectre`)
- `yacs`
- `numpy`

### Key Finding
- **EMICA/SPECTRE cannot be imported in the host Python environment**
- The `inferno` package is NOT available in host Python (only accessible inside Docker via `emica_env`)
- `omegaconf`, `yacs`, `kornia` are not installed in host Python

---

## 4. Stage0-11 vs Stage12 Environment

| Stage | Environment | Python | PyTorch | Notes |
|---|---|---|---|---|
| Stage 0-5 | Host | `/opt/hb/bin/python3` | 2.4.1+cpu | SCRFD, 6DRepNet, Face Detection |
| Stage 6-11 | Host | `/opt/hb/bin/python3` | 2.4.1+cpu | HyperIQA, SyncNet, Landmark, Alignment |
| Stage 12 | **Docker** | `/workspace/emica_env/bin/python` | CUDA | EMICA + SPECTRE |

---

## 5. Recommendation

**Stage12 must run as a subprocess inside the Docker container.**

Architecture:
```
Stage0-11 (host Python)
    ↓
[AlignedFaceInfo list + metadata JSON]
    ↓
Stage12 Runner (docker exec face_pipeline_lxs)
    ├── source /workspace/emica_env/bin/activate
    └── python stage12_runner.py
    ↓
[FLAMEFeature outputs: .npy + .json]
```

This approach:
- Avoids modifying Stage0-11 environment
- Keeps EMICA/SPECTRE dependencies isolated in Docker
- Provides clean subprocess boundary
- Allows independent testing of Stage12

---

## 6. Files Created

- `docs/Stage12_Runtime_Environment_Analysis.md` (this file)
- `face_pipeline_v2/stages/stage_12_flame/runner.py` (subprocess runner)
- `face_pipeline_v2/stages/stage_12_flame/save_utils.py` (output saving)