# PROJECT_CONTEXT.md — Face_Pipeline_V2

> **For AI coding assistants**. Load at session start. Updated 2026-08-23.

---

## 1. Project Identity

| | |
|---|---|
| **Name** | `Face_Pipeline_V2` |
| **Type** | Multi-modal Talking Face Dataset Generation Pipeline |
| **Input** | Raw video (任意来源、格式) |
| **Output** | Time-aligned training sample: Video Clip + Face Sequence + FLAME Motion (structured) + Gaze (6d) + Audio Embedding (768d) + Metadata |
| **Package** | `face_pipeline_v2` (editable install: `pip install -e .`) |
| **Python** | 3.10 (dev), 3.12.3 (PPU container) |
| **Platform note** | PPU GPU (non-NVIDIA). PyTorch requires `LD_PRELOAD` workaround. ONNX runtime works natively. |

---

## 2. Pipeline Data Flow

```
Raw Video (Path)
 → Stage 0  [FFprobe / ffmpeg]       VideoInfo
 → Stage 0.5 [rule-based]            ResolutionValidationInfo             resolution quality filter
 → Stage 1  [FFmpeg]                 StandardizedVideoInfo + AudioInfo   (25fps / 16kHz mono)
 → Stage 2  [PySceneDetect]          List[SceneInfo]
 → Stage 3  [SCRFD]                  List[SceneInfo] (filtered)          ✅ real model
 → Stage 4  [pure logic]             List[ClipInfo]                      (sliding_window 8s/4s | segment 整段)
 → Stage 5  [SCRFD + bbox expand]    FaceSequence                        (detection + expanded base crop)
 → Stage 5.1 [ByteTrack]             FaceSequence (+ track_id)           (enhancement, optional)
 → Stage 5.2 [rule-based]            FaceSequence (identity_result; rejected DROPPED)   (hard filter, optional)
 → Stage 5.3 [rule-based]            FaceSequence (stability_result; rejected DROPPED)  (hard filter, optional)
 → Stage 6  [6DRepNet PyTorch]  ─┐
 → Stage 7  [SyncNet PyTorch]   ─┤ parallel → Stage 9 [fusion] → QualityMetrics → accepted clips
 → Stage 8  [HyperIQA PyTorch]   ─┘                                      ✅ pure rule logic
 → Stage 7.1 [ffmpeg/rule-based]     AVAlignmentInfo                     (audio timeline correction)
 → Stage 9.1 [rule-based]            QualityRankingInfo                  (enhancement, optional ranking)
 → Stage 10 [InsightFace 2d106det]   LandmarkInfo                        (106-point landmarks)
 → Stage 11 [affine / OpenCV]        AlignedFaceInfo                     (256×256 canonical face)
 → Stage 11.5                        FaceValidityInfo                    🔴 stub
 → Stage 12 [EMICA + SPECTRE]    ─┐
 → Stage 13 [MediaPipe]        ─┤
 → Stage 14 [HuBERT]           ─┤ parallel → Stage 17 [alignment] → TrainingSample
 → Stage 15 [stats]            ─┤
 → Stage 16 [annotation]       ─┘
```

**Audio rule**: Stage 14 MUST consume Stage 7.1's `AVAlignmentInfo.corrected_audio_path`, never raw audio.

**Enhancement modules**: Designed for open-domain video. For professional talking-face data, disable all via `enhancements.*.enabled=false`. Stage 5.2/5.3 are **hard filters** when enabled — rejected sequences are DROPPED and never reach Stage 6/7/8 (Progressive Gating). Stage 9.1 (Quality Ranking) is also an enhancement — provides weighted ranking for data selection, does NOT participate in hard filtering.

---

## 3. Implementation Status

### Stage Details

| Stage | Status | What it does | Model | Input → Output | Config key |
|---|---|---|---|---|---|
| **0** | ✅ complete | Video integrity check | FFprobe (ffmpeg fallback) | `Path` → `VideoInfo` | `video.*` |
| **0.5** | 🔴 planned | Video resolution hard filter | Rule-based (no model) | `VideoInfo` → `VideoInfo` (+ resolution validation) | `stage_00_resolution_filter.*` |
| **1** | ✅ complete | AV standardization to 25fps / 16kHz | FFmpeg | `VideoInfo` → `StandardizedVideoInfo` + `AudioInfo` | `stage_01.*` |
| **2** | ✅ complete | Scene segmentation | PySceneDetect ContentDetector | `StandardizedVideoInfo` → `List[SceneInfo]` | `stage_02.*` |
| **3** | ✅ complete | Coarse face screening 1fps | SCRFD (ONNX, local-first) | `List[SceneInfo]` → `List[SceneInfo]` + metadata | `stage_03.*` |
| **4** | ✅ complete | Sliding window clip gen | Pure logic | `List[SceneInfo]` → `List[ClipInfo]` | `stage_04.*` |
| **5** | ✅ complete | Face detection + expanded base crop | SCRFD (ONNX, local-first) | `List[ClipInfo]` → `List[FaceSequence]` | `stage_05.*`, `stage_05_face_crop.*` |
| **5.1** | ✅ complete | Face tracking | ByteTrack | `List[FaceSequence]` → `List[FaceSequence]` (+ track_id) | `enhancements.*` |
| **5.2** | ✅ complete | Single face identity **hard filter** (rejected dropped) | Rule-based | `List[FaceSequence]` → `List[FaceSequence]` (only passed; identity_result attached) | `enhancements.*` |
| **5.3** | ✅ complete | Temporal stability **hard filter** (rejected dropped) | Rule-based | `List[FaceSequence]` → `List[FaceSequence]` (only passed; stability_result attached) | `enhancements.*` |
| **6** | ✅ complete | Head pose estimation | 6DRepNet (PyTorch) | `List[FaceSequence]` → `List[PoseSequence]` | `stage_06.*` |
| **7** | ✅ complete | A/V sync detection | SyncNet (PyTorch) | `dict{sequences, audio, clips}` → `List[AVSyncInfo]` | `stage_07.*` |
| **7.1** | ✅ complete | A/V sync correction (audio timeline alignment) — 三态 `already_aligned`/`corrected`/`severe_offset_unavailable` | FFmpeg / rule-based | `dict{av_sync_infos, std_video_info, clip_infos?}` → `List[AVAlignmentInfo]` | `stage_07_1_av_alignment.*` |
| **8** | ✅ complete | Face quality assessment | HyperIQA (PyTorch) | `List[FaceSequence]` → `List[FaceQualitySequence]` | `stage_08.*` |
| **9** | ✅ complete | Quality metrics fusion & hard filter | Rule-based (no model) | `dict{pose_sequences, av_sync_infos, quality_sequences}` → `List[QualityMetrics]` | `pose.*`, `sync.*`, `quality.*`, `stage_09.*` |
| **9.1** | ✅ complete | Quality ranking (enhancement) | Rule-based (no model) | `dict{quality_metrics}` → `List[QualityRankingInfo]` | `quality_ranking.*`, `enhancements.*` |
| **9.5** | ✅ complete | Clip export (accepted clips only) | FFmpeg / rule-based | `dict{quality_metrics, clip_infos, std_video_info, av_alignments, av_sync_infos}` → manifest | `stage_09_5_clip_export.*` |
| **10** | ✅ complete | High precision 106-point landmark | InsightFace 2d106det (ONNX) | `dict{face_sequences, accepted_clip_ids}` → `List[LandmarkInfo]` | `landmark.*` |
| **11** | ✅ complete | Face alignment to 256×256 | Affine / OpenCV (no model) | `dict{landmarks, face_sequences}` → `List[AlignedFaceInfo]` | `alignment.*` |
| **11.5** | 🔴 stub | Face validity check | Rule-based | — | — |
| **12** | 🟡 design_frozen | FLAME motion extraction | EMICA + SPECTRE | — | `flame.*` |
| **13** | 🔴 stub | Gaze extraction | MediaPipe | — | `gaze.*` |
| **14** | 🔴 stub | Audio feature extraction | HuBERT / AV-HuBERT | `AVAlignmentInfo.corrected_audio_path` → `AudioFeature` (768d) | `audio.*` |
| **15** | 🔴 stub | Quality statistics | Pandas | — | — |
| **16** | 🔴 stub | Data annotation | Whisper/FairFace/CLIP | — | — |
| **17** | 🔴 stub | Feature alignment & fusion | Custom | — | — |

### Infrastructure Status

| Component | File | Status |
|---|---|---|
| Config loader | `face_pipeline_v2/config/config_loader.py` | ✅ validates 14 top-level keys |
| BaseStage ABC | `face_pipeline_v2/pipeline/base_stage.py` | ✅ 169 lines; validate→process→validate |
| EnhancementStage | `face_pipeline_v2/pipeline/enhancement_stage.py` | ✅ type-consistent passthrough; _get_passthrough_output() hook |
| PipelineContext | `face_pipeline_v2/pipeline/context.py` | ✅ frozen dataclass |
| Data models | `face_pipeline_v2/data/models.py` | ✅ 26 frozen dataclasses |
| PipelineMetadata | `face_pipeline_v2/data/metadata.py` | ✅ provenance tracking |
| Model manager | `face_pipeline_v2/models/model_manager.py` | ✅ registry + **kwargs; 5 models registered |
| Pipeline runner | `face_pipeline_v2/pipeline/pipeline_runner.py` | 🔴 empty |
| Exception hierarchy | `face_pipeline_v2/utils/exceptions.py` | ✅ 8 exception types |
| Logger | `face_pipeline_v2/utils/logger.py` | ✅ PipelineLogger |
| E2E Integration test | `test_scripts/e2e_stage0_11/run_e2e_test.py` | ✅ runs Stage 0-11 on real video |

---

## 4. Models

| Model | Stage | Weights | Wrapper | Interface | Loading |
|---|---|---|---|---|---|
| FFprobe | 0 | system | `stage_00/video_checker.py` | — | system call |
| FFmpeg | 1 | system | `stage_01/ffmpeg_processor.py` | — | system call |
| PySceneDetect | 2 | offline .whl | `stage_02/scene_detector.py` | — | offline pip |
| **SCRFD** | 3, 5 | ✅ `scrfd_10g_bnkps.onnx` (17MB) | `models/wrappers/scrfd.py` | `detect(image) → List[dict]` | ONNX local-first, no auto-download |
| ByteTrack | 5.1 | — | `stage_05_1/tracker_adapter.py` | — | third_party |
| **6DRepNet** | 6 | ✅ `SixDRepNet.onnx` + `6DRepNet_300W_LP_AFLW2000.pth` | `models/wrappers/sixdrepnet.py` | `predict(images) → List[dict{yaw,pitch,roll}]` | PyTorch via third_party/sixdrepnet; device="auto"; PPU GPU support |
| **SyncNet** | 7 | ✅ `syncnet_v2.model` (53MB) | `models/wrappers/syncnet.py` | `predict(video_path) → (offset, score, conf)` | PyTorch via third_party/SyncNet; weights_only=True; device="auto" |
| **HyperIQA** | 8 | ✅ `koniq_pretrained.pkl` + `resnet50-19c8e357.pth` | `models/wrappers/hyperiqa.py` | `predict(images) → List[float]` (0-100) | PyTorch via third_party/HyperIQA; offline resnet50 (patched model_zoo.load_url); device="auto" |
| **InsightFace 2d106det** | 10 | ✅ `2d106det.onnx` | `models/wrappers/insightface_landmark.py` | `predict(image) → (106, 2) landmarks` | ONNX runtime direct |
| MediaPipe | 13 | — | — | — | not started |
| EMICA | 12 | design_frozen | `models/wrappers/emica.py` 🟡 design | — | design_frozen |
| SPECTRE | 12 | design_frozen | `models/wrappers/spectre.py` 🟡 design | — | design_frozen |
| HuBERT | 14 | ✅ `hubert_base_ls960.pt` (1.1GB) | `models/wrappers/wav2vec2.py` 🔴 empty | — | loading verified, wrapper not started |

### ModelManager registry
| Model | Module | Class |
|---|---|---|
| `scrfd` | `models.wrappers.scrfd` | `SCRFDWrapper` |
| `sixdrepnet` | `models.wrappers.sixdrepnet` | `SixDRepNetWrapper` |
| `syncnet` | `models.wrappers.syncnet` | `SyncNetWrapper` |
| `hyperiqa` | `models.wrappers.hyperiqa` | `HyperIQAWrapper` |
| `insightface_landmark` | `models.wrappers.insightface_landmark` | `InsightFaceLandmarkWrapper` |

---

## 5. Configuration

**Single source**: `configs/config.yaml`. 26 top-level sections.

### Per-Stage Config

| Section | Key params |
|---|---|
| `video` | target_fps(25), audio_sample_rate(16000), check_video_stream, check_audio_stream |
| `stage_00_resolution_filter` | enabled(true), min_width(640), min_height(360) — rejects videos below 640×360 |
| `stage_01` | target_fps(25), audio_sample_rate(16000), video_codec(libx264), audio_codec(pcm_s16le), output_dir |
| `stage_02` | detector(ContentDetector), threshold(27.0), min_scene_len(15) |
| `stage_03` | enabled, sampling_fps(1), min_duration(8), min_face_detect_ratio(0.95), min_single_face_ratio(0.95), min_face_confidence(0.6), debug.* |
| `stage_04` | enabled, mode(sliding_window\|segment, 默认 sliding_window), window_size(8), stride(4), target_fps(25) |
| `stage_05` | enabled, face_selection.strategy(largest_area), crop.enabled, crop.size(256), crop.expand_ratio(1.6) |
| `stage_05_1` | tracker.type(bytetrack), track_thresh(0.5), track_buffer(30), match_thresh(0.8) |
| `stage_06` | enabled, model_name(sixdrepnet), batch_size(32), device(cuda) |
| `stage_07` | enabled, model_name(syncnet), checkpoint, device(cpu), sample_rate(16000), batch_size(32), vshift(10) |
| `stage_07_1_av_alignment` | enabled(true), correction_threshold_ms(50), output_dir(aligned_audio)；三态对齐，`aligned_audio_path` 为最终可用音频 |
| `stage_08` | enabled, model_name(hyperiqa), checkpoint, device(cuda), batch_size(32) |
| `stage_09` | enabled, min_pose_ratio(0.85), fusion_threshold(null), fusion_weights{pose:0.4, sync:0.3, quality:0.3} |
| `pose` | yaw_threshold(30), pitch_threshold(20), roll_threshold(30) — Stage 9 pose thresholds |
| `sync` | max_offset_frames(15) — Stage 7.1 correction range; not used by Stage 9 |
| `quality` | quality_threshold(0.32) + sync{max_offset_frames(8), min_lse_c(3.0)} — Stage 9 hard filter（临时值，待 validation set 标定） |
| `quality_ranking` | weights{pose:0.3, sync:0.3, face_quality:0.4}, levels{excellent:0.85, good:0.70, normal:0.50} — Stage 9.1 |
| `landmark` | model_path(models/insightface/2d106det.onnx), device(auto) — Stage 10 config |
| `alignment` | resolution(256), save_aligned_faces(true), output_dir(outputs/aligned_faces) — Stage 11 config |
| `flame` | model(EMICA), modules(mica/deca/emica/spectre), jaw_refinement(lambda=0.6), expression_dim(100), pose_dim(6) |
| `gaze` | model(MediaPipe), output_dim(6) |
| `audio` | model(HuBERT/base), sample_rate(16000), feature_dim(768) |
| `models` | scrfd, sixdrepnet, syncnet, hyperiqa, insightface_landmark (path + device per model) |
| `runtime` | device(auto), backend(ppu), num_workers(4), batch_size(32), enable_parallel |
| `enhancements` | stage_05_1/2/3/9_1 enabled flags + thresholds |
| `storage` | input_dir, output_dir(outputs), model_dir(models), log_dir |
| `validation` | output_dir, save_json, save_report, visualization.* |

---

## 5b. Per-Model Preprocessing Chain

Stage 5 outputs a unified **Base Face Crop** (expanded bbox, native resolution, no resize).
Each downstream stage applies its own preprocessing (expand + resize) via `face_pipeline_v2/utils/face_preprocess.py`.

```
Stage 5:  SCRFD tight bbox → expand_bbox (stage_05_face_crop.expand_ratio=2.0)
          → crop at native resolution → raw face crop
              │
    ┌─────────┼─────────┬──────────────┐
    ▼         ▼         ▼              ▼
  Stage 6   Stage 7   Stage 8        Stage 10
  6DRepNet  SyncNet   HyperIQA       InsightFace
  resize    resize    resize          resize
  256×256   256*      224×224         256×256
```

All preprocess params in config: `stage_0{6,7,8}.preprocess.input_size`.

---

## 6. Runtime / Backend

### 开发环境
- **CPU PyTorch**: 开发机器使用 `torch==2.4.1+cpu`，所有模型 CPU 推理。
- **device="auto"**: 自动检测 `torch.cuda.is_available()`，不可用时 fallback 到 CPU。

### 生产推理环境
- **PPU Docker**: T-Head (平头哥) PPU-ZW810E GPU（CUDA 兼容层，驱动 alixpu v1.5.1，编译器 HGGC）。16 卡、96GB/卡。
- **PyTorch 2.9.0**: 容器 `face_pipeline_lxs`（emodia 镜像）内版本；`torch.cuda.is_available()=True`，device_name=`PPU-ZW810E`，compute capability `sm_80`。
- **PPU SDK**: 宿主机 `/opt/pg1`（CUDA 12.3 + cuDNN 8.9.5 + glibc 2.32）——无法加载 onnxruntime CUDA EP；容器内 `/usr/local/PPU_SDK`（CUDA 12.9 + cuDNN 8.9.5 + glibc 2.39）。
- **LD_PRELOAD workaround**: PPU GPU (非 NVIDIA) 上 PyTorch CUDA 库符号不兼容，需 LD_PRELOAD 解决
- **ONNX runtime**: 无 `PPUExecutionProvider`。PPU 是 CUDA 兼容卡，需 `onnxruntime-gpu` 的 `CUDAExecutionProvider`。**2026-08-18 已通过阿里官方 PPU 适配版 pip 包跑通（见下方「ONNX Runtime / PPU 适配状态」）**。

### Server / Hardware / Runtime Environment（2026-08-18 实测）

| 项 | 值 |
|---|---|
| **GPU** | T-Head PPU-ZW810E（CUDA 兼容，驱动 alixpu v1.5.1，编译器 HGGC） |
| **Python** | 3.12.3 |
| **OS** | Ubuntu 24.04 |
| **CUDA** | 12.9 |
| **PPU SDK** | 2.0（`/usr/local/PPU_SDK`） |
| **环境变量** | `CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK`、`PPU_SDK=/usr/local/PPU_SDK` |
| **ONNX Runtime** | `onnxruntime_gpu==1.21.1+ppu1.7.0.oe`（阿里官方 PPU 适配版） |

### ONNX Runtime / PPU 适配状态

- **✅ 适配完成（2026-08-18）**：无需源码编译 ONNX Runtime。联系阿里 PPU 技术支持确认官方提供 PPU 适配版 ONNX Runtime pip 包。
- **安装**：`pip install onnxruntime_gpu==1.21.1+ppu1.7.0.oe`
- **Provider 验证**：
  ```python
  import onnxruntime as ort
  ort.get_available_providers()
  # ['CUDAExecutionProvider', 'CPUExecutionProvider']
  ```
- **说明**：PPU 通过 `CUDAExecutionProvider` 兼容方式工作，**不显示单独 `PPUExecutionProvider`**。代码侧 `execution_provider=ppu` → `CUDAExecutionProvider` 映射无需改动。
- **结论：当前 SCRFD ONNX 推理已经支持 PPU。**

### Stage 3 SCRFD 状态（PPU）

- Stage 3（粗筛人脸 1fps）使用 `models/SCRFD/scrfd_10g_bnkps.onnx`，走 `runtime/onnx_runtime.py::create_onnx_session()`。
- **PPU 支持已就绪**：配置 `scrfd.execution_provider` / `models.scrfd.execution_provider`（auto/ppu/cuda/cpu），PPU 容器下 `auto` 自动选择 `CUDAExecutionProvider`。
- 2026-08-18 起可在 PPU 容器（onnxruntime_gpu 1.21.1+ppu1.7.0.oe）上 CUDA 加速推理。

### Stage 5 SCRFD 状态（PPU）

- Stage 5（人脸检测 + expanded base crop）同样使用 SCRFD ONNX，共享同一 `execution_provider` 配置与 `create_onnx_session()` 后端。
- **PPU 支持已就绪**：与 Stage 3 一致，PPU 上自动使用 `CUDAExecutionProvider` 加速。
- 帧级检测缓存（`Dict[frame_index, List[FaceInfo]]`）与后端无关，PPU 加速后 Stage 5 整体吞吐显著提升。

### SCRFD PPU 性能实测（2026-08-18）

| 项 | 值 |
|---|---|
| 模型 | `models/SCRFD/scrfd_10g_bnkps.onnx` |
| 输入 | 640×640 |
| PPU（CUDAExecutionProvider） | **约 6.03 ms/帧** |
| CPU（CPUExecutionProvider） | 约 62~182 ms/帧 |
| 加速比 | **约 10~30 倍** |

### 设备选择流程

```
config.yaml
    ↓
PipelineContext
    ↓
ModelManager
    ↓
Wrapper (device="auto" → torch.cuda.is_available() ? "cuda" : "cpu")
```

- **device="auto"**: 自动检测 CUDA 可用性，不可用时 fallback 到 CPU。
- **video_reader**: `runtime/video_reader.py` — 统一视频帧读取抽象层。支持 `ffmpeg`（pipe 顺序读取，减少 MP4 GOP 重复解码）和 `opencv`（seek-based，回退兼容）。配置：`runtime.video_reader`。
- **ONNX runtime**: SCRFD 和 InsightFace Landmark 使用 ONNX（onnxruntime）。SCRFD 支持 `execution_provider`（auto/ppu/cuda/cpu）自动选择后端 + fallback + 打印实际 provider；PPU 需 `onnxruntime-gpu` 的 `CUDAExecutionProvider`（**2026-08-18 已跑通，见上方「ONNX Runtime / PPU 适配状态」**）。

### 各模型设备配置

| Model | Backend | device 支持 |
|---|---|---|
| SCRFD | ONNX | execution_provider (auto/ppu/cuda/cpu) — **PPU 已验证（CUDA EP，~6.03ms/帧）** |
| 6DRepNet | PyTorch | auto / cpu / cuda (PPU) |
| SyncNet | PyTorch | auto / cpu / cuda (PPU) |
| HyperIQA | PyTorch | auto / cpu / cuda (PPU) |
| InsightFace 2d106det | ONNX | N/A (ONNX 自动选择) |

---

## 7. Data Model Chain

26 frozen dataclasses in `face_pipeline_v2/data/models.py`:

```
VideoInfo → ResolutionValidationInfo → StandardizedVideoInfo + AudioInfo → SceneInfo → ClipInfo
  → FaceSequence → FrameInfo / FaceInfo (bbox, 5-point landmark, confidence, crop_path, track_id)
  → SingleFaceIdentityResult / TemporalStabilityResult
  → PoseSequence → PoseInfo (yaw/pitch/roll) / AVSyncInfo (lse_c, lse_d, offset) / FaceQualitySequence → FaceQualityInfo
  → AVAlignmentInfo (corrected_audio_path — for Stage 14)
  → QualityMetrics (fusion_score, is_accepted, per-dimension scores)
  → QualityRankingInfo (weighted_score, quality_level: excellent/good/normal)
  → LandmarkInfo (106-point × 2 = 212 values)
  → AlignedFaceInfo (affine_matrix, original_image_size, aligned_resolution)
  → FLAMEFeature / GazeFeature / AudioFeature
  → StatisticsInfo / AnnotationInfo → TrainingSample
```

**Field notes:**

- `VideoInfo` (Stage 0) — metadata extraction only, **no filtering**: `width`, `height`, `fps`, `duration`, codec, `has_audio` (video/audio stream info).
- `ResolutionValidationInfo` (Stage 0.5, planned) — `width`, `height`, `min_width`, `min_height`, `is_valid`, `reject_reason`.

---

## 8. Data Asset Output (Stage 11 Final)

Stage 11 之后形成最终高质量 clip 数据资产，供下游 Stage 12-17 消费。

### Per-clip assets after Stage 11

| Asset | Source Stage | Format |
|---|---|---|
| video clip | Stage 1 + Stage 4 | MP4 (h264, 25fps) |
| aligned audio | Stage 7.1 | WAV (16kHz, corrected timeline) |
| aligned face sequence | Stage 11 | 256×256 JPG per frame |
| 106 landmark coordinates | Stage 10 | List[LandmarkInfo] (212 floats) |
| affine transformation matrix | Stage 11 | Tuple[float×6] per face |
| quality metrics | Stage 9 + 9.1 | QualityMetrics + QualityRankingInfo |
| metadata manifest | — | JSON per clip |

### Output directory structure (proposed)

```
outputs/
├── clips/           # video clips (Stage 4)
├── audio/           # aligned audio (Stage 7.1)
├── faces/           # face crops (Stage 5)
├── aligned_faces/   # canonical aligned faces (Stage 11)
├── landmarks/       # landmark JSON (Stage 10)
├── alignment/       # affine matrices (Stage 11)
├── metadata/        # quality metrics + ranking (Stage 9 + 9.1)
└── stage_outputs/   # per-stage JSON exports
```

---

## 9. Project Structure

```
Face_Pipeline_V2/
├── configs/config.yaml
├── docs/                                    # 7 AISDD docs v1.0
├── face_pipeline_v2/                        # Main package
│   ├── config/config_loader.py
│   ├── data/models.py, metadata.py
│   ├── models/
│   │   ├── model_manager.py                 # ✅ registry, 5 models
│   │   └── wrappers/scrfd.py, sixdrepnet.py, syncnet.py, hyperiqa.py, insightface_landmark.py
│   ├── pipeline/base_stage.py, enhancement_stage.py, context.py
│   ├── stages/stage_00/ ... stage_11/ (stage 11.5-17 stubs)
│   └── utils/exceptions.py, logger.py, video_utils.py
├── models/                                  # Weight files
│   ├── SCRFD/scrfd_10g_bnkps.onnx           # 17MB
│   ├── 6DRepNet/SixDRepNet.onnx             # ONNX + PyTorch weights
│   ├── syncnet/syncnet_v2.model             # 53MB
│   ├── HyperIQA/
│   │   ├── koniq_pretrained.pkl             # HyperIQA checkpoint
│   │   └── resnet50-19c8e357.pth            # ResNet50 local weights
│   ├── insightface/2d106det.onnx            # 106-point landmark
│   ├── EMICA/EMICA-CVT_flame2020_notexture/  # EMICA weights (MICA+DECA+ExpressionEncoder)
│   ├── FLAME/geometry/                      # FLAME 2020 generic model
│   └── hubert/hubert_base_ls960.pt          # HuBERT Base (1.1GB)
├── third_party/
│   ├── sixdrepnet/                          # Official SixDRepNet (regressor, model, utils)
│   ├── SyncNet/                             # SyncNetModel.py + SyncNetInstance
│   ├── HyperIQA/                            # models.py (HyperNet + TargetNet)
│   ├── ByteTrack/
│   ├── inferno/                             # DECA/EMOCA FaceRecBase (legacy; not used by Stage 12)
│   └── pyscenedetect_offline/
├── tests/                                   # pytest test files
├── test_scripts/
│   ├── e2e_stage0_11/                       # Stage 0-11 E2E integration test
│   ├── stage0_8_validation/                 # Stage 0-8 integration test (legacy)
│   ├── debug_6drepnet/                      # 6DRepNet debug + integration test
│   ├── debug_syncnet/                       # SyncNet debug + interface audit
│   ├── debug_scrfd_output.py                # ONNX output analysis
│   ├── debug_bbox_fix.py                    # SCRFD bbox decode test
│   ├── debug_hyperiqa.py                    # HyperIQA debug test
│   ├── test_stage10_landmark.py             # Stage 10 standalone test
│   ├── test_stage11_alignment.py            # Stage 11 standalone test
│   ├── run_stage_03_e2e.py
│   └── real_model_validation/
├── test_outputs/e2e_test/                   # E2E integration test outputs
├── test_reports/                            # Test reports
└── PROJECT_CONTEXT.md, DEVELOPMENT_LOG.md
```

---

## 10. Design Decisions

1. **SCRFD distance decode** — fixed 2026-08-06: ONNX outputs `(dx1,dy1,dx2,dy2)`, NOT `(cx,cy,w,h)`. Decode uses `grid - dx1` / `grid + dx2` without `exp()`.
2. **SCRFD local-first loading** — fixed 2026-08-11: ONNX model loaded directly, insightface only as fallback if pre-downloaded. No auto-download.
3. **25 FPS** — standard frame rate.
4. **Stage4 双模式 Clip 生成** — `sliding_window`（8s clips, 4s stride，传统样本，默认）；`segment`（整段 1:1 ClipInfo，动态时长，用于 16s segment → T≈400 样本）。Stage4 重新定义为「Clip Generation / Clip Adapter Stage」。
5. **Progressive gating** — cheap stages first, expensive models on accepted data. Stage 9 (Quality Fusion) acts as the final quality gate.
6. **Immutable frozen dataclasses** — all data models are frozen.
7. **EnhancementStage passthrough** — optional modules pass through when disabled. Subclasses with mismatched input/output types override `_get_passthrough_output()`. Stage 9.1 returns `[]` when disabled for type consistency.
8. **ModelManager registry** — dynamic import + **kwargs for wrapper-specific params. 5 models registered.
9. **Stage loading pattern** — `get_model()` first, `load_model()` if not cached. No direct wrapper instantiation (fixed for Stage 7; legacy fallbacks still in Stage 3/5/6/8).
10. **Face crop expand_ratio=1.6** — validated 2026-08-11. Bbox expansion resolved SyncNet input distribution shift (LSE-C 0.62→6.03). Config: `stage_05.crop.expand_ratio=1.6` with `boundary_check.fallback_ratio=1.5`.
11. **Audio timeline correction** — Stage 7.1 applies SyncNet-detected offset via ffmpeg. Corrected audio path stored in AVAlignmentInfo for Stage 14 consumption.
12. **Stage 11 explicit error on save failure** — `cv2.imwrite` failure raises `RuntimeError` instead of silently skipping.
13. **Stage 9 SyncNet hard filter uses raw LSE-C + offset** — `abs(offset) <= max_offset_frames(8)` AND `sync_lse_c >= min_lse_c(5.0)` (typical range 3-10). Normalized sync_score used only for fusion weighting and Stage 9.1 ranking. Reject reasons: "sync_offset_exceeded" / "sync_lse_c_too_low".
14. **Per-model face crop preprocessing** — Stage 5 saves raw bbox crops at native resolution (no resize). Each downstream stage applies its own expand+resize via `face_pipeline_v2/utils/face_preprocess.py`: Stage 6 (expand 1.4→256), Stage 7 (expand 1.6→256, clip-level audio), Stage 8 (expand 1.6→224). Config in `stage_0{6,7,8}.preprocess`.
15. **Stage 7 uses clip-level audio** — SyncNet input is built from face crops + per-clip audio segment (extracted from Stage 1 full audio via ffmpeg `-ss clip_start -t clip_duration`), NOT the full video audio track.
16. **Stage 7.1 produces per-clip corrected audio** — when `clip_infos` is passed in the input dict, corrected audio is trimmed to clip boundaries after offset correction.
17. **Validation output = Stage 9 accepted clips only** — all assets (clips, audio, aligned_faces, landmarks) are filtered to only include clips with `QualityMetrics.is_accepted=True`.
18. **Stage 5 produces expanded base face crop** — SCRFD tight bbox → `expand_ratio=2.0` on original frame → crop at native resolution. No resize in Stage 5. Each downstream stage does its own resize via `face_preprocess`.
19. **Stage 5 frame detection cache** — deduplicates SCRFD inference across overlapping clips. A `Dict[frame_index, List[FaceInfo]]` cache avoids re-running SCRFD on the same frame. Typical hit rate: 46-57%.
20. **Pluggable video reader backend** — `runtime/video_reader.py` with `VideoFrameReader` ABC. FFmpeg pipe (`ffmpeg`) for sequential decode, OpenCV (`opencv`) for fallback. Config: `runtime.video_reader`.
21. **Stage 10/11 coordinate mapping uses `expanded_bbox`** — fixed 2026-08-13. Stage 5 crops with `expanded_bbox` (raw, no resize), so crop↔frame mapping is a pure 1:1 offset: Stage 10 `frame = crop + expanded_bbox.xy1`, Stage 11 `crop = frame − expanded_bbox.xy1`. Previously used `tight_bbox` + scale, which misaligned aligned faces. Legacy fallback to `tight_bbox` retained for old `FaceInfo` without `expanded_bbox`.
22. **Stage 7.1 three-state alignment** — 2026-08-13. `AVAlignmentInfo.alignment_status` ∈ {`already_aligned`, `corrected`, `severe_offset_unavailable`}, and `aligned_audio_path` points to the final usable clip-level audio (or `None` for severe offset). Stage 9.5 reads `aligned_audio_path` (not `corrected_audio_path`), so already-aligned clips still get their trimmed clip audio exported.
23. **InsightFace 2d106det input is raw [0,255]** — fixed 2026-08-13. The ONNX graph applies its own normalization `(x - 127.5) * 0.0078` (Sub + Mul nodes at the input). The wrapper must feed raw [0,255] RGB, NOT `/255`-normalized input, or the landmark output collapses.
24. **2d106det landmark order ≠ standard 106-point order** — fixed 2026-08-13. Verified empirically: contour 0-32, left-eye 33-42, left-brow 43-51, mouth 52-71, nose 72-86, right-eye 87-96, right-brow 97-105. `affine_transform.py` key points are left-eye `33-42`, right-eye `87-96`, nose-tip `83-85`.
25. **Stage 0.5 resolution filter** — video resolution is the cheapest quality signal. Hard-filter on original resolution (default ≥ 640×360) before FFmpeg standardization / scene segmentation / any model inference. Higher-quality datasets can raise the threshold. (Planned, not yet implemented.)
26. **SCRFD execution_provider 后端选择** — 2026-08-16。SCRFD/onnxruntime 统一走 `runtime/onnx_runtime.py::create_onnx_session()`，支持 `execution_provider`（auto/ppu/cuda/cpu）自动选择 + 逐个 fallback + 打印实际 provider。PPU 是 CUDA 兼容卡、无 `PPUExecutionProvider`，`ppu` 映射到 `CUDAExecutionProvider`。配置：`scrfd.execution_provider` / `models.scrfd.execution_provider`（默认 auto）。
27. **PPU ONNX Runtime 官方适配版 pip 包** — 2026-08-18。无需源码编译 ORT。阿里官方提供 `onnxruntime_gpu==1.21.1+ppu1.7.0.oe`，PPU 通过 `CUDAExecutionProvider` 兼容工作（无独立 `PPUExecutionProvider`）。容器：Python 3.12.3 / Ubuntu 24.04 / CUDA 12.9 / PPU SDK 2.0。SCRFD 640×640 推理 PPU ~6.03ms/帧（CPU ~62-182ms/帧），约 10-30× 加速。
28. **Stage 5.2/5.3 as early hard filters (Progressive Gating)** — 2026-08-19. Previously 5.2/5.3 appended every sequence and only attached `identity_result`/`stability_result` (annotation-only); rejected clips still ran through Stage 6/7/8, wasting GPU inference. Now `process()` returns ONLY sequences whose result `passed=True`; rejected sequences are dropped before Stage 6/7/8. Logs `input_count`/`passed_count`/`rejected_count`/`reject_reason_counts`. Disabled (`enabled=false`) still passes through unchanged. Stage 5.3 stays independent of 5.2 (falls back to all-bboxes logic when `identity_result` absent). Docs synced: AISDD 01/02/03/04/05/06.

---

## 10b. FLAME Motion Extractor Design (Stage 12 — Design Frozen, Updated 2026-08-24)

### Architecture

Stage 12 uses **EMICA + SPECTRE** dual-model architecture. EMICA is the primary FLAME encoder; SPECTRE raw jaw is retained for audit only (final scheme: `jaw_pose` = EMICA, no fusion).

### EMICA (Main Encoder)

EMICA inherits **MICA + DECA + ExpressionEncoder + SPECTRE** modules, all controlled via config flags.

| Component | Source | Output | Dims |
|---|---|---|---|
| **expression** | EMICA ExpressionEncoder | `expression_code` | 100 |
| **jaw_pose** | EMICA pose[0:3] | `jaw_pose` | 3 |
| **global_pose** | EMICA pose[3:6] | `global_pose` | 3 |
| **eye_pose** | EMICA (fixed zeros) | `eye_pose` | 6 |
| **shape_code** | EMICA MICA module (optional) | `shape_code` | 300 |

**Key finding (2026-08-24):** EMICA does **NOT** predict `eye_pose`. The encoder's `predicts` list contains `expcode(100), texcode(50), jawpose(3), globalpose(3), cam(3), lightcode(27), shape(300)` — no eye_pose entry. `FaceEncoder._decompose_code()` does not handle any eye parameter. The checkpoint contains only `shape_model.flame.eye_pose` (fixed zeros buffer) and `renderer.uv_face_eye_mask` (rendering mask). **Conclusion**: eye_pose always defaults to `zeros(6)` and must be provided by Stage 13 (Gaze) if required.

### EMICA vs EMOCA Relationship

- **EMICA ExpressionEncoder** produces `expression_code[100]` — this is Stage 12's output.
- **EMOCA** (original DECA-based expression encoder) produces `expcode[50]` — this is NOT Stage 12 output.
- EMICA's ExpressionEncoder is a distinct module from EMOCA's; the two should not be confused.
- EMICA checkpoint (`EMICA-CVT_flame2020_notexture`) contains the full encoder stack (MICA+DECA+ExpressionEncoder).

### EMICA Modules

| Module | Role | Default |
|---|---|---|
| **MICA** | Shape encoder → `shape_code[300]` | disabled |
| **DECA** | Detail/texture encoder backbone | disabled |
| **ExpressionEncoder** | Expression encoder → `expression_code[100]` | enabled |
| **SPECTRE** | Raw jaw (audit) → `spectre_jawpose` | enabled |

### SPECTRE (Auxiliary, Audit Only for jaw)

- SPECTRE processes video frame sequences and outputs raw jaw_pose[3]
- **FINAL scheme: `jaw_pose` = EMICA jaw (no fusion)** — validated by FLAME mesh
  reconstruction (SPECTRE jaw version had mouth-region distortion)
- SPECTRE raw jaw is preserved on the feature as `spectre_jawpose` debug/audit field
- Config: `flame.spectre.enabled`

### Jaw Source (FINAL Adopted)

```
jaw_pose = EMICA_jaw_pose   # pure EMICA, no fusion
```

- `jaw_fusion_lambda` is **deprecated** (no longer participates in computation)
- Config key: `stage12.jaw_source` = `"emica"` (final production scheme)
- 历史记录: 此前曾采用 SPECTRE jaw（Scheme D）与 λ 加权融合
  (`final_jaw_pose = EMICA_jaw_pose + λ × (SPECTRE_jaw_pose − EMICA_jaw_pose)`),
  mesh 重建对比后最终选择 EMICA jaw

### eye_pose

- EMICA encoder does NOT predict eye_pose (confirmed by config/code/checkpoint audit 2026-08-24)
- Stage 12 initializes eye_pose as `zeros(6)` placeholder
- Reserved for Stage 13 gaze estimator (MediaPipe) to replace

### FLAMEFeature Dataclass

- Structured fields (not fixed 56-dim or 112-dim vector): `expression_code[100]`, `jaw_pose[3]`, `global_pose[3]`, `eye_pose[6]`, plus optional `shape_code[300]`, `camera_parameters`
- Flatten operation deferred to Stage 17 fusion
- The old design `expcode[:50] + posecode[:6]` → 56-dim fixed vector is **deprecated**

### Module Switch

- MICA / DECA / ExpressionEncoder / SPECTRE controlled via config flags
- Default: `ExpressionEncoder=true`, `SPECTRE=true`, `MICA=false`, `DECA=false`
- Config: `flame.emica.expression_encoder.enabled`, `flame.spectre.enabled`, etc.

---

## 10c. Audio Feature Extraction Design (Stage 14)

### Model

- HuBERT Base (`hubert_base_ls960.pt`, ~1.1GB)
- 未来可选 AV-HuBERT 用于 audio-visual 联合特征

### Output

- Audio embedding: **768 dims** per frame
- Time-aligned with FLAME frame sequence (25 FPS)
- Input: Stage 7.1 `AVAlignmentInfo.corrected_audio_path`

### Status

- Weights ready: `models/hubert/hubert_base_ls960.pt`
- Wrapper stub: `face_pipeline_v2/models/wrappers/wav2vec2.py` (empty)
- 待实现

---

## 11. Known Issues

### Critical
- `pipeline_runner.py` empty — no multi-stage orchestration (E2E test script serves as de facto orchestrator).
- ~~**Stage 5.2/5.3 were annotators, not hard filters**~~ → **✅ 已解决（2026-08-19）**：5.2/5.3 `process()` 现在只返回 `passed=True` 的 sequence，被拒绝的 sequence 不再进入 Stage 6/7/8，Progressive Gating 生效。见 Design Decision #28。
- Stage 0.5 Resolution Filter implementation pending (design added 2026-08-14, not yet implemented).
- Stages 11.5, 12-17 待实现。
- **PPU GPU 上 PyTorch 模型无法加载**（RD_Radio2_000, PPU-ZW810E + PyTorch 2.9.0）：Stage 6 报 `No module named 'sixdrepnet'`（PYTHONPATH 缺 third_party/sixdrepnet）；Stage 8 报 `weights_only=True` 与 legacy `.tar` 不兼容（PyTorch 2.6+ 默认变更，`koniq_pretrained.pkl` 需 `weights_only=False`）；Stage 7 SyncNet fallback 全零。ONNX 模型（SCRFD/InsightFace）不受影响。
- ~~**SCRFD ONNX 无法在 PPU 上跑 CUDA EP**~~ → **✅ 已解决（2026-08-18）**：原因为 PyPI 的 `onnxruntime-gpu` 无「CUDA 12 + cuDNN 8.9」组合（≤1.18.x = CUDA 11 + cuDNN 8；≥1.19.x = CUDA 12 + cuDNN 9），源码编译又因网络卡死。**解决**：阿里官方 PPU 适配版 pip 包 `onnxruntime_gpu==1.21.1+ppu1.7.0.oe`，PPU 走 `CUDAExecutionProvider`。历史数据：容器 CPU ~113ms/张、宿主机 ~2.5s/张 → **PPU ~6.03ms/张**。

### Medium
- `requirements.txt` empty.
- Stage 1 returns plain `dict` instead of typed dataclass (`{"video_info": ..., "audio_info": ...}`).
- Orphan directories: `stage_04_clip_generation/`, `stage_06_pose_filter/`, `stage_07_av_sync/`, `stage_08_quality_filter/`.
- `test_config.py`, `test_pipeline.py`, `filesystem.py` are empty.
- Stages 3/5/6/8 still have direct wrapper loading fallbacks (bypass ModelManager). Only Stage 7 was cleaned.
- PyTorch requires `LD_PRELOAD` workaround on PPU machine.
- Stage 9 thresholds are provisional — 2026-08-13 lowered to `quality_threshold=0.32`, `min_lse_c=3.0`, `min_pose_ratio=0.85` based on HDTF reference video (RD_Radio1_000: HyperIQA mean 46.95→norm 0.47, LSE-C 4.9–6.5, pose all-frontal). Still need a real validation-set calibration.
- SyncNet LSE-C values on 640×360 test video: 0.27/0.44 (below 0.5 default). May indicate model sensitivity to video resolution.
- **Stage 11 affine alignment**: some aligned face crops missing top of head / chin. Root cause: current crop expansion + alignment mapping causes chin to exceed 256 canvas on extreme poses. Pure geometry analysis verified. Need alignment strategy optimization.

### Minor
- EMICA, SPECTRE, and wav2vec2 wrappers need implementation (design frozen). EMICA wrapper must support module enable/disable for MICA, DECA, ExpressionEncoder, SPECTRE; output `expression_code[100]`, `pose_code[6]`, `eye_pose[6]`, optional `shape_code[300]`, optional `camera_parameters`.
- `hyperiqa.py:126` uses `weights_only=False` — checkpoint contains full model objects, not just state_dict.
- stage_03 uses config.yaml thresholds (0.95) which are stricter than E2E test thresholds (0.5).

---

## 12. Latest Development Status (2026-08-23)

### 2026-08-23 — Stage12 Design Finalized, HuBERT & PPU Environment Confirmed

#### DONE
- **Stage 12 FLAME Motion Design Frozen (Updated 2026-08-24)** — EMICA + SPECTRE jaw refinement architecture finalized. EMICA is the primary FLAME encoder (MICA + DECA + ExpressionEncoder + SPECTRE modules, config-controlled). Expression from EMICA ExpressionEncoder `expression_code[100]`, jaw_pose = EMICA initial + SPECTRE refinement. **Confirmed: EMICA does NOT predict eye_pose** — always `zeros(6)`, reserved for Stage 13 gaze. EMOCA produces `expcode[50]` and is NOT Stage 12 output. FLAMEFeature structured dataclass design (not fixed-dim vector). 8 docs updated + new `Stage12_Design_Finalization_Report.md`.
- **HuBERT audio feature** — `hubert_base_ls960.pt` (1.1GB) ready at `models/hubert/`. Target output: 768-dim audio embedding per frame, time-aligned with FLAME.
- **PPU environment confirmed** — Container `emodia`: Python 3.12.3, PPU torch 2.9.0, `onnxruntime_gpu==1.21.1+ppu1.7.0.oe`. SCRFD ONNX verified: `CUDAExecutionProvider` + `CPUExecutionProvider`.
- **Affine alignment issue analyzed** — Some aligned faces missing top of head / chin on extreme poses. Root cause: crop expansion + alignment mapping causes chin to exceed 256 canvas. Pure geometry analysis verified. Need alignment strategy optimization.

#### NEXT
1. **Stage 12 implementation** — EMICA inference wrapper (module enable/disable: MICA, DECA, ExpressionEncoder, SPECTRE), SPECTRE jaw refinement integration, FLAMEFeature dataclass.
2. **HuBERT feature extraction** — wrapper implementation.
3. **Stage 13 gaze** — MediaPipe integration to replace eye_pose placeholder.
4. **FLAME / audio / gaze timestamp fusion** — Stage 17 alignment.
5. **Fix PPU PyTorch model loading** — sixdrepnet / HyperIQA / SyncNet.

### 2026-08-18 — PPU ONNX Runtime 适配完成，SCRFD GPU 加速验证通过

#### DONE
- **ONNX Runtime PPU 适配完成**：无需源码编译 ORT。阿里 PPU 技术支持确认官方提供 PPU 适配版 ONNX Runtime pip 包。
- **安装**：`onnxruntime_gpu==1.21.1+ppu1.7.0.oe`（容器：Python 3.12.3 / Ubuntu 24.04 / CUDA 12.9 / PPU SDK 2.0；`CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK`、`PPU_SDK=/usr/local/PPU_SDK`）。
- **Provider 验证**：`ort.get_available_providers()` → `['CUDAExecutionProvider', 'CPUExecutionProvider']`。PPU 通过 CUDA EP 兼容方式工作，不显示单独 `PPUExecutionProvider`。
- **SCRFD 已成功接入 PPU 加速**（`models/SCRFD/scrfd_10g_bnkps.onnx`，640×640 输入）：PPU（CUDA EP）**~6.03 ms/帧** vs CPU **~62~182 ms/帧**，**约 10~30 倍加速**。
- **影响范围**：Stage 3 / Stage 5 的 SCRFD ONNX 推理均支持 PPU（共享 `create_onnx_session()` 后端 + `execution_provider` 配置）。

#### To-Do
- **下一步 Stage 12 EMICA + SPECTRE（FLAME Motion）** —— 设计已冻结，进入代码实现。
- 修复 PPU 容器内 PyTorch 模型加载（sixdrepnet / HyperIQA / SyncNet）。
- Stage 9 阈值用更大 validation set 标定。

### DONE（历史 2026-08-16）
- **SCRFD 后端选择改造**（保持 ONNX 模型 / onnxruntime wrapper / FaceInfo 输出不变，只加后端选择）：
  - `runtime/backend.py` 修复 `AzureExecutionProvider`（CPU EP）被误判为 PPU 的 bug。
  - `runtime/onnx_runtime.py`：`create_onnx_session()` 新增 `execution_provider`（auto/ppu/cuda/cpu）+ 解析 + fallback + 打印实际 provider。
  - `models/wrappers/scrfd.py`：新增 `execution_provider` 参数 + `active_providers` 属性 + 加载打印（未动 detect/decode/FaceInfo）。
  - `stage_03` / `stage_05`：config 读取并透传 `execution_provider`。
  - `configs/config.yaml`：新增 `scrfd.execution_provider` / `models.scrfd.execution_provider`（auto/ppu/cpu）。
  - `scripts/test_scrfd_backend.py`：新增验证脚本。
- **环境结论**：PPU-ZW810E 是 T-Head CUDA 兼容卡，无 `PPUExecutionProvider`，正确后端是 `onnxruntime-gpu` 的 `CUDAExecutionProvider`，ONNX 无需转换。

### BLOCKED
- ~~**SCRFD 在 PPU 上跑通**~~ → **✅ 已解决（2026-08-18）**：onnxruntime-gpu 版本与 SDK cuDNN 错配 → 阿里官方 PPU 适配版 pip 包解决，**不再需要源码编译 ORT**（详见上方 2026-08-18 DONE）。

### NEXT
1. **Stage 12 EMICA + SPECTRE（FLAME Motion）** —— 设计已冻结，进入代码实现。
2. 修复 PPU 容器内 PyTorch 模型加载（sixdrepnet / HyperIQA / SyncNet）。
3. Stage 9 阈值用更大 validation set 标定。

### 2026-08-13 记录（历史）

### 今日完成事项
- 修复「expanded_bbox 裁剪，但使用 tight_bbox 坐标映射导致 aligned face 错位」——Stage 10 `_map_to_frame` 与 Stage 11 `align_faces` 统一改用 `expanded_bbox` offset（1:1 无缩放），保留 tight_bbox legacy fallback。
- 修复 Stage 11 aligned face「半张脸」根因（两处）——`insightface_landmark.py` 预处理 `/255` → 改为 [0,255]（模型图内已有 `(x-127.5)*0.0078` 归一化）；`affine_transform.py` 关键点索引改为左眼 33-42 / 右眼 87-96 / 鼻尖 83-85。
- Stage 7.1 重构为三态对齐（`already_aligned` / `corrected` / `severe_offset_unavailable`），`AVAlignmentInfo` 新增 `aligned_audio_path` + `alignment_status`。
- Stage 9.5 改为读取 `aligned_audio_path`，处理 severe offset 的 audio_unavailable。
- Stage 9 阈值基于 HDTF 参考视频标定：`quality_threshold=0.32`、`min_lse_c=3.0`、`min_pose_ratio=0.85`（临时）。

### 今日修改文件
| File | Change |
|---|---|
| `face_pipeline_v2/data/models.py` | `AVAlignmentInfo` 新增 `aligned_audio_path`/`alignment_status`；`FaceInfo.expanded_bbox` 文档化 |
| `face_pipeline_v2/stages/stage_10_landmark/processor.py` | `_map_to_frame` 改用 expanded_bbox offset |
| `face_pipeline_v2/models/wrappers/insightface_landmark.py` | 预处理 `/255` → [0,255]（模型内部 Sub/Mul 归一化） |
| `face_pipeline_v2/stages/stage_11_alignment/affine_transform.py` | 关键点索引改左眼 33-42 / 右眼 87-96 / 鼻尖 83-85 |
| `face_pipeline_v2/stages/stage_11_alignment/processor.py` | 帧→crop 反向映射改用 expanded_bbox offset + coord-check 日志 |
| `face_pipeline_v2/stages/stage_11_alignment/stage.py` | （同步 processor 改动） |
| `face_pipeline_v2/stages/stage_07_1_av_alignment/processor.py` | 三态分类 + `extract_clip_audio` + `should_correct` |
| `face_pipeline_v2/stages/stage_07_1_av_alignment/stage.py` | 三态对齐流程，写 `aligned_audio_path` |
| `face_pipeline_v2/stages/stage_09_5_clip_export/exporter.py` | 读取 `aligned_audio_path`，audio_unavailable 处理 |
| `configs/config.yaml` | 阈值 0.32 / 3.0 / 0.85（临时） |
| `test_scripts/{run_threshold_reference,run_stage8_11_reuse,run_full_pipeline,test_stage71_audio}.py` | 测试脚本新增/更新 |

### 当前 pipeline 状态
- Stage 0–11 + 9.5 闭环可用；Stage 11.5、12–17 仍为 stub。
- **PPU GPU**：ONNX 模型（SCRFD/InsightFace）正常；PyTorch 模型（6DRepNet/SyncNet/HyperIQA）加载失败（见 Known Issues）。
- **FLAME Motion (Stage 12)** 设计已冻结——EMICA + SPECTRE jaw refinement architecture，待代码实现。

### 下一步工作
1. 修复 PPU GPU 容器内 PyTorch 模型加载（sixdrepnet PYTHONPATH、HyperIQA `weights_only=False`、SyncNet 依赖）。
2. 实现 EMICA + SPECTRE（Stage 12 FLAME Motion）。
3. Stage 9 阈值用更大 validation set 标定。

---

## 13. How to Continue

### Priority order

1. **PPU GPU 修复** — ~~SCRFD ONNX PPU 适配~~ **✅ 已完成（2026-08-18）**；剩余 GPU 容器内三个 PyTorch 模型加载（sixdrepnet 路径、HyperIQA `weights_only=False`、SyncNet 依赖），使 Stage 6/7/8 在 PPU 上跑通。
2. **Stage 12 implementation** — EMICA inference wrapper (module enable/disable: MICA, DECA, ExpressionEncoder, SPECTRE), SPECTRE jaw refinement integration, FLAMEFeature dataclass, config module switch.
3. **Stage 14 HuBERT** — audio feature extraction wrapper, 768-dim per-frame embedding.
4. **Stage 13 gaze** — MediaPipe integration for eye_pose replacement.
5. **Stage 17 fusion** — FLAME / audio / gaze timestamp alignment.
6. **Pipeline Runner** — wire Stage 0→17 end-to-end orchestration (currently E2E test script serves this role).
7. **Stage 9 threshold calibration** — run on validation set to determine data-driven thresholds.
8. **Stage 11 alignment fix** — optimize crop expansion + alignment mapping for extreme poses.
9. **requirements.txt** — document all Python dependencies.

### Bootstrap files
`configs/config.yaml` → `data/models.py` → `pipeline/base_stage.py` → `stages/stage_03_face_screening/stage.py` (mature SCRFD reference) → `models/model_manager.py` (registry pattern) → `test_scripts/e2e_stage0_11/run_e2e_test.py` (E2E integration harness).
