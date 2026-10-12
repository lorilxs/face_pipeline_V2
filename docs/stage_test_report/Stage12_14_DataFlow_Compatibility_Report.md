# Stage12_14_DataFlow_Compatibility_Report.md

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


# Stage12 / Stage13 / Stage14 跨阶段数据接口兼容性审计报告

Version: v1.0 | Date: 2026-08-26 | 审计方式: 只读审查（未修改任何代码）

---

## 0. 审计范围与结论摘要

对 Stage 12 (FLAME) / Stage 13 (Gaze) / Stage 14 (Audio) 三个模块的数据接口进行跨阶段兼容性审查，
验证其是否完全满足 Stage 17（Parameter Alignment & Fusion → TrainingSample）最终训练数据生成需求。

**总体结论：三个 Stage 的数据模型与持久化格式基本兼容，时间基准一致，可直接进入 Stage 17 开发。
发现 4 项必须修复的问题（其中 2 项为环境/测试层面，1 项为 Stage12 NPZ 字段缺失，1 项为编排入口缺失），
以及若干建议性改进项。详见第 5 章。**

---

## 1. 当前架构图

```
Stage 5  (face detection, absolute frame_idx) ─ timestamp = frame_idx / target_fps (绝对时间)
    │
    ▼
Stage 10 (landmark) → Stage 11 (alignment) → List[AlignedFaceInfo] (timestamp/clip_id 已携带)
    │                                        │
    │ Stage 12 (Docker subprocess)           │ Stage 13 (in-process MediaPipe)
    │   EMICA + SPECTRE                      │   face_landmarker.task
    ▼                                        ▼
List[FLAMEFeature>                      List[GazeFeature>
    │                                        │
    └── flame_features.npz             gaze_features.npz
            │                                │
            └───────────┬────────────────────┘
                        ▼
Stage 7.1 (AVAlignmentInfo: aligned_audio_path + clip_start)
    │
    ▼
Stage 14 (in-process transformers+HuBERT, 50Hz → 25FPS)
    │
    └── audio_features.npz
            │
            ▼
Stage 17 (Parameter Alignment & Fusion) → TrainingSample
    (nearest-timestamp merge, 见 §3.4)
```

**关键点：**
- Stage 12 / 13 共用同一输入 `List[AlignedFaceInfo]`（Stage 11 输出），因此 `frame_index` / `timestamp` / `clip_id` 逐帧一一对应。
- Stage 14 输入为 `AVAlignmentInfo`（Stage 7.1 校正音频），仅含 `timestamp`（无 `frame_index`），与视频帧的时间对齐必须通过 timestamp 完成。
- 三个 Stage 均各自持久化 NPZ 到 `outputs/stageXX/{clip_id}/`，Stage 17 以 NPZ 为事实数据源（dataclass 可往返恢复）。

---

## 2. 数据字段表

### 2.1 Data Model 字段核对（face_pipeline_v2/data/models.py）

| Feature | shape | required | current status |
|---|---|---|---|
| FLAMEFeature.expression_code (`expcode`) | [T,100] | ✅ 是 | ✅ 满足（EMICA ExpressionEncoder 100 维） |
| FLAMEFeature.pose (`posecode` property = `globalpose`[3] + `jawpose`[3]) | [T,6] | ✅ 是 | ✅ 满足（jaw 融合 + EMICA global） |
| FLAMEFeature.eye_pose | [T,6] | ✅ 是 | ⚠️ 满足但为固定 `zeros(6)` 占位（Stage13 不写回，Stage17 负责替换） |
| FLAMEFeature.timestamp | [T] float | ✅ 是 | ✅ 满足（默认 0.0，实际写入 absolute 时间） |
| FLAMEFeature.clip_id | [T] str | ✅ 是 | ✅ 满足（默认 ""，实际写入 clip 标识） |
| GazeFeature.gaze_vector | [T,6] | ✅ 是 | ✅ 满足（gaze_0_x/y/z + gaze_1_x/y/z） |
| GazeFeature.timestamp | [T] float | ✅ 是 | ✅ 满足（与 Stage12 同源 AlignedFaceInfo） |
| GazeFeature.clip_id | [T] str | ✅ 是 | ✅ 满足 |
| AudioFeature.feature | [T,768] | ✅ 是 | ✅ 满足（HuBERT hidden_size=768 实测确认） |
| AudioFeature.timestamp | [T] float | ✅ 是 | ✅ 满足（clip_start + i/fps） |
| AudioFeature.source_audio | Path | ✅ 是 | ✅ 满足（记录校正音频路径） |
| AudioFeature.dimension | int | ✅ 是 | ✅ 满足（=768） |

> FLAMEFeature 还含 `shapecode[300]`（MICA 未加载，zeros）、`cam[3]`（DECA 未开启，zeros）、`spectre_jawpose[3]`（审计轨迹）等可选字段，均不影响 Stage 17 必需接口。

### 2.2 Stage 12 输出检查（stage_12_flame → flame_features.npz）

| 期望字段 | 实际 NPZ key（runner.save_outputs） | 是否写入 | 恢复途径 |
|---|---|---|---|
| expression_code | `expression_code` [T,100] | ✅ 写入 | read_stage12_outputs → `expcode` |
| jaw_pose | `jaw_pose` [T,3] | ✅ 写入 | → `jawpose` |
| head_pose | `head_pose` [T,3] | ✅ 写入 | → `globalpose` |
| pose_code | （不直接存，由 head+jaw 拼接） | ✅ 可推导 | `posecode` property = globalpose[3]+jawpose[3] |
| eye_pose | `eye_pose` [T,6] | ✅ 写入（zeros） | → `eye_pose` |
| timestamp | `timestamp` [T] (float64) | ✅ 写入 | → `timestamp` |
| clip_id | `clip_id` [N] | ❌ **未写入 NPZ** | ⚠️ 仅靠 read_stage12_outputs 的 clip_id 参数回退 |
| shapecode | `shapecode` [T,300] | ✅ 写入（zeros） | → `shapecode` |
| spectre_jawpose | `spectre_jawpose` [T,3] | ✅ 写入 | → `spectre_jawpose` |
| cam | （未写入） | ❌ 未写入 | read_stage12_outputs 回退 zeros(3) |

**read_stage12_outputs() 可完整恢复 FLAMEFeature**（含 timestamp/clip_id/cam 回退逻辑），
已由 `tests/test_stage12_output_metadata.py`（含 clip_id 回退、cam 回退、camera_params key 回退等用例）验证通过。

### 2.3 Stage 13 输出检查（stage_13_gaze → gaze_features.npz）

| 期望字段 | 实际 NPZ key | 是否写入 | 恢复途径 |
|---|---|---|---|
| gaze_vector | `gaze_vector` [T,6] | ✅ 写入 | read_gaze_features → `gaze_vector` |
| timestamp | `timestamp` [T] (float32) | ✅ 写入 | → `timestamp` |
| clip_id | `clip_id` [T] str | ✅ 写入 | → `clip_id` |
| frame_index | `frame_index` [T] | ✅ 写入 | → `frame_index` |
| left_eye / right_eye | （未单独存） | ⚠️ 可推导 | read 时按 gv[0:3] / gv[3:6] 拆分 |

**时间轴与 Stage12 一致性：** 两者消费同一个 `List[AlignedFaceInfo]`，`timestamp` 同源 → 逐帧一致。
注意：Stage13 会因「图片缺失 / 无检测结果」而 **跳过个别帧**（`continue`），故 `T_gaze` 可能略小于 `T_flame`；
Stage12 也会因图片缺失跳过 → 两序列长度可能不同，**Stage17 必须按 frame_index（而非行号）对齐 Stage12/13**。

### 2.4 Stage 14 输出检查（stage_14_audio_feature → audio_features.npz）

| 期望字段 | 实际 NPZ key | 是否写入 | 恢复途径 |
|---|---|---|---|
| feature | `feature` [N,768] | ✅ 写入 | read_audio_features → `feature` |
| timestamp | `timestamp` [N] (float32) | ✅ 写入 | → `timestamp` |
| dimension | `dimension` (scalar) | ✅ 写入 | → `dimension` |
| source_audio | `source_audio` (str) | ✅ 写入 | → `source_audio` |
| frame_index | （无此字段） | — | AudioFeature 无 frame_index，按 timestamp 对齐 |

**audio_metadata.json** 含 clip_id / num_frames / feature_dim / source_audio / fields；
**未含** clip_start、fps、原始 HuBERT 帧数 T_h（见 §5 建议）。

---
## 3. 时间同步分析

### 3.1 统一时间基准（已确认一致 ✅）

| Stage | timestamp 来源 | 时间基准 |
|---|---|---|
| Stage 12 (FLAME) | `AlignedFaceInfo.timestamp`（Stage 11 ← Stage 10 ← Stage 5） | 绝对时间 = 标准化视频时间轴 |
| Stage 13 (Gaze) | `AlignedFaceInfo.timestamp`（同源） | 绝对时间（与 Stage 12 一致） |
| Stage 14 (Audio) | `AVAlignmentInfo.clip_start + i/fps` | 绝对时间（clip_start = start_frame/fps） |

证据链：
- Stage 5 `stage.py:269`：`timestamp = frame_idx / target_fps`，其中 `frame_idx ∈ range(clip.start_frame, clip.end_frame)` 为**绝对帧号**（非 clip 相对帧号）→ 绝对时间基准。
- Stage 7.1 `stage.py:205-206, 296`：`timestamp = start_frame / fps` → `clip_start=timestamp` → 同样为绝对时间。
- Stage 14 `processor.py:141`：`timestamp = start_time + i / fps`，`start_time = alignment.clip_start`。

**结论：三个 Stage 的时间戳处于同一时间轴（标准化视频绝对秒数），可直接进行 timestamp 级 merge，无需额外偏移换算。**

⚠️ 注意：若 `AVAlignmentInfo.clip_start` 为 `None`，Stage 14 `stage.py:100` 会静默回退 `start_time=0.0`（变为 clip 相对），
与 Stage 12/13 的绝对时间产生偏移。正常流程下 clip_start 由 Stage 7.1 填充，此风险仅在「clip 边界无法解析」时出现，需在 Stage 17 做一致性校验。

### 3.2 HuBERT 帧率与 timestamp 正确性

- 实测 `config.json`：`conv_stride = [5,2,2,2,2,2,2]` → 总 stride = 5×2⁶ = 320；16 kHz 输入 → **50 Hz** 原始帧率；`hidden_size = 768`。
- Stage 14 `resample_to_fps(50 → 25)`：`n_out = round(T_h × 25/50) = round(T_h/2)`，`positions = arange(n_out)×(T_h/n_out)`，对偶数 T_h 即**每隔 1 帧抽取**（取第 0,2,4,… 帧）。
- timestamp = `clip_start + i/25`，与视频帧网格一致。

**两个需要注意的边界/语义问题（均为近似，非阻断）：**
1. **帧数近似**：HuBERT conv 栈对音频长度取 floor，8 s 音频的 `T_h` 可能为 399/400，导致 `N_audio ≈ T_video ± 1`（clip 边界处差 1 帧）。已在实现层面接受「不要求严格 25 fps」，但 Stage 17 合并时需容忍 ±1 帧差。
2. **半帧滞后约定**：AudioFeature[i]（t = start + i×40ms）取的是 HuBERT 帧 2i，其覆盖音频窗口为 `[t, t+20ms)` —— 特征实际代表「视频帧时刻之后 20ms 的音频」。这是通用 lip-sync 管线的常见约定，不影响接口兼容，但建议在 Stage 17 文档中明确记录，避免下游误读。

### 3.3 8 秒 Clip Stage 17 输入模拟（video fps=25）

```
8 s clip, 25 FPS
├─ 视频帧:  T_video = 200        (frame 0..199, t = 0, 0.04, …, 7.96 s)
├─ FLAME:   T_flame = 200        (timestamp 与视频帧一致, absolute)
├─ Gaze:    T_gaze  ≈ 200        (逐帧；若有检测失败则 < 200)
└─ Audio:   T_h = 8s × 50Hz = 400  (floor 后可能 399)
            N_audio = round(400×25/50) = 200
            (若 T_h=399 → N_audio=200，若 T_h=398 → N_audio=199)
```

结论：三种模态帧数 **≈ 200**，可按 timestamp 合并；Audio 在 clip 边界存在 ±1 帧近似。

### 3.4 推荐 Alignment 策略（供 Stage 17）

| 模态 | 对齐键 | 策略 | 理由 |
|---|---|---|---|
| FLAME | `frame_index`（或 timestamp） | **逐帧索引**（frame_index 直接对应视频帧） | FLAME 与 Gaze 共享同一 AlignedFaceInfo 输入，frame_index 即视频帧号 |
| Gaze | `frame_index` | **按 frame_index 对齐 FLAME**；缺失帧用最近可用帧填充或标记 dropout | 避免按行号错位 |
| Audio | `timestamp` | **nearest-timestamp**：对每个视频帧时刻 t_j = start + j/25，取 argmin_i \|t_j − t_audio_i\| | Audio 无 frame_index，且帧数存在 ±1 近似；nearest 天然容忍 |

- **推荐 nearest timestamp，而非插值**：FLAME/Gaze/Audio 均为「每帧一个特征向量」的表示，插值会引入虚假的中间特征；
  talking-head 类训练数据的主流做法是 nearest 对齐（1 帧容差内精确命中 25 fps 网格）。
- 若需严格 T 一致：对 Audio 在尾部做「超出容忍 → 丢弃」或「补最后一帧」的确定性策略，并在 metadata 记录。

---

## 4. 已发现问题

| # | 严重度 | 位置 | 问题 |
|---|---|---|---|
| F1 | 🔴 高 | `tests/test_datamodel.py:159` | `test_all_models_instantiable` 用 4 参构造 `GazeFeature(0, left, right, gv)`，而当前模型要求 6 字段（frame_index, timestamp, clip_id, left_eye, right_eye, gaze_vector）→ TypeError，导致该测试文件全量失败（预存在，非本工作引入） |
| F2 | 🔴 高 | `stage_12_flame/runner.py:307-316` | `flame_features.npz` **未写入 `clip_id`** 数组，与 `docs/03_3.18.10`（期望 `clip_id:[N]`）不一致；`read_stage12_outputs` 靠函数参数回退才恢复正确 clip_id |
| F3 | 🟠 中 | 环境（host Python） | **Stage 13 依赖 `mediapipe`，当前 host python3（3.10.12）、emica_env、venv_myenv 均未安装**（实测 ModuleNotFoundError）；`models/mediapipe/face_landmarker.task` 资产存在。Stage 13 以 in-process 方式运行，无 Docker 桥接 → 当前环境无法真实跑通 |
| F4 | 🟠 中 | `pipeline/pipeline_runner.py` | 文件为 **0 字节（空）**，无编排入口。Stage12/13/14 目前只能通过 manual 脚本 / 各 Stage runner 单独调用，Stage 17 缺少端到端触发与数据装配的宿主 |
| F5 | 🟡 低 | `stage_14` | `audio_metadata.json` 未记录 `clip_start` / `fps` / 原始 `T_h` / 实际 N_audio，Stage 17 难以做「帧数偏差校验」与溯源 |
| F6 | 🟡 低 | 时间戳 dtype | Stage12 timestamp 为 float64，Stage13/14 为 float32（微秒级舍入差，nearest 匹配无影响） |
| F7 | 🟡 低 | `stage_14/stage.py:107` | `process()` 每个 clip 都重新 `load_hubert_wrapper()`（加载 ~1.1GB 模型），批量处理性能差；建议在 stage/context 缓存 wrapper |

---

## 5. Stage17 开发前必须修复的问题

1. **F3：为 Stage 13 安装 mediapipe 或提供容器桥接。**（阻断）
   - 方案 A：在运行管线的主机 Python 安装 `mediapipe`（与 `transformers/soundfile/torch` 同环境）。
   - 方案 B：仿照 Stage 12 增加 Docker/subprocess 桥接（若 host 不便安装 mediapipe）。
   - 修复后需跑通 `tests/test_stage13_gaze_real_data.py`（真实数据，当前 4 个跳过中包含此项）。

2. **F2：Stage 12 runner 的 `save_outputs()` 写入 `clip_id` 数组。**
   - 一行修复：`flame_data["clip_id"] = np.array([r["clip_id"] for r in results])`。
   - 若 Stage 17 选择「直接读 NPZ 而非 read_stage12_outputs」，缺少 clip_id 将导致样本溯源断裂。
   - 同时补充 `read_stage12_outputs` 对 `has_clip_ids` 分支的回归测试（已有用例覆盖回退路径，需覆盖新写入路径）。

3. **F1：修复 `tests/test_datamodel.py::test_all_models_instantiable` 的 GazeFeature 构造。**
   - 改为 6 参：`GazeFeature(0, 0.0, "c", (0,0,0), (0,0,0), (0,0,0,0,0,0))`。
   - 否则全量 CI 被该测试阻断。

4. **F4：Stage 17 落地时建立编排/装配入口（可复用现有 per-clip 模式）。**
   - Stage 17 需要一个宿主脚本：按 clip 读 `flame_features.npz` / `gaze_features.npz` / `audio_features.npz` → 按 §3.4 策略 merge → 输出 `TrainingSample`。
   - 在写 Stage 17 前至少明确 NPZ 为唯一事实数据源（与三个 Stage 的 integration 模块保持一致）。

---

## 6. 不需要修改的问题

- **时间基准统一**（§3.1）：三个 Stage 均使用绝对时间（标准化视频时间轴），无需换算。
- **FLAME 维度接口**：expression[100] / pose[6] / eye[6] 均满足；`shapecode=zeros(300)`、`cam=zeros(3)`、`eye_pose=zeros(6)` 为明确的占位/审计设计，非缺陷。
- **Gaze 接口**：gaze_vector[6] + timestamp + clip_id 齐全；left/right_eye 可由 gaze_vector 拆分还原。
- **Audio 接口**：feature[768] + timestamp + source_audio + dimension 齐全；输入严格为 Stage 7.1 校正音频（非 raw audio），符合 AISDD 3.20.3。
- **HuBERT 帧率**：50 Hz 原始 → 25 FPS 抽取；「不要求严格 25 fps、允许 ±1 帧近似」是既定设计。
- **read_* 往返**：`read_stage12_outputs` / `read_gaze_features` / `read_audio_features` 均可完整恢复 dataclass（已有测试验证）。
- **模型资产**：`models/hubert/chinese-hubert-base/`（config.json + pytorch_model.bin + preprocessor_config.json，hidden 768）与 `models/mediapipe/face_landmarker.task` 均已就位。
- **配置**：`config.yaml` 的 `stage12` / `stage13` / `stage_14` / `audio` 段与各 stage.py 的 `config.get(...)` 键一致。
- **测试状态**：Stage12/13/14 相关 78 项测试通过、4 项跳过（跳过项为需要真实 mediapipe 数据的用例，对应 F3）。

---

## 7. 环境隔离检查（实测）

| 阶段 | 设计/实现 | 实测环境 | Python | 关键依赖 | 状态 |
|---|---|---|---|---|---|
| Stage 0-11 | host Python | `/opt/hb/bin/python3` | 3.10.12 | torch 2.4.1+cpu, onnxruntime, etc. | ✅ |
| Stage 12 | Docker subprocess（`docker exec face_pipeline_lxs` + `source /workspace/emica_env/bin/activate`） | 容器 `face_pipeline_lxs` / `emica_env`（`/workspace/emica_env/bin/python`） | 3.10 | torch(CUDA), pytorch-lightning, inferno, omegaconf | ✅ 设计中；docker CLI 在当前审查会话不可用，未复测 |
| Stage 13 | **in-process**（stage.py 直接调 mediapipe） | host python3（测量） | 3.10.12 | **mediapipe 缺失**（transformers/soundfile 亦缺失） | 🔴 阻断（F3） |
| Stage 14 | **in-process**（stage.py 直接调 transformers+soundfile） | host python3（测量） | 3.10.12 | transformers 4.22.2 ✅, soundfile 0.14.0 ✅, torch 2.4.1+cpu ✅ | ✅ 可运行 |

补充说明：
- `emica_env`（py3.10）与 `venv_myenv`（py3.10）均未安装 mediapipe / transformers / soundfile；host python3 已具备 Stage 14 全部依赖。
- HuBERTWrapper 注释中「容器 Python 3.12 的 torchaudio ABI 损坏」指向生产 PPU 容器；本审查环境为 3.10.12 dev host，Stage 14 的 WAV 读取统一走 `soundfile`，不依赖 torchaudio，故无影响。
- 生产 PPU 容器为 Python 3.12.3（`face_pipeline_lxs` / emodia 镜像），`fairseq` 在 3.12 不可用 → 已确认 Stage 14 固定使用 transformers 后端（`models/hubert/chinese-hubert-base`），fairseq `.pt` 显式拒绝。

---

## 附录：审查依据（只读访问的文件）

- `face_pipeline_v2/data/models.py`（AVAlignmentInfo / FLAMEFeature / GazeFeature / AudioFeature / TrainingSample）
- `face_pipeline_v2/stages/stage_12_flame/{stage,processor,integration,runner}.py`
- `face_pipeline_v2/stages/stage_13_gaze/{stage,processor,integration,runner}.py`
- `face_pipeline_v2/stages/stage_14_audio_feature/{stage,processor,integration,runner}.py`
- `face_pipeline_v2/models/wrappers/hubert.py`
- `face_pipeline_v2/pipeline/pipeline_runner.py`（0 字节）、`face_pipeline_v2/stages/stage_05_face_detection/stage.py`、`stage_07_1_av_alignment/stage.py`、`stage_11_alignment/processor.py`
- `configs/config.yaml`、`docs/02_Data_Model_and_Interface.md`、`docs/03_Pipeline_Stage_Specification.md`、`docs/04_Config_and_Model_Management.md`、`docs/05_Data_Flow_and_Output.md`、`docs/Stage14_Design_Update_Report.md`
- `tests/test_stage12_output_metadata.py`、`tests/test_datamodel.py` 等（并运行 Stage12/13/14 相关测试：78 passed, 4 skipped）
- 运行环境实测（python 版本、mediapipe/transformers/soundfile/torch 可用性、模型资产清单）

---

## End of Stage12_14_DataFlow_Compatibility_Report.md v1.0

