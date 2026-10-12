# Stage4 Removal 全管线影响审计报告（只读）

> 审计日期：2026-08-29
> 审计范围：Stage5–Stage17（只读检查，未修改任何代码 / config / pipeline）
> 审计目标：确认取消 Stage4（固定 8s 滑动窗口切片）后，哪些 stage 可直接工作、哪些存在隐藏 8s 依赖、哪些需要修改及修改复杂度。
> 结论先行：**生产代码中不存在硬编码的 8 秒 / 200 帧依赖**；真正的隐藏依赖是 **`ClipInfo` + `clip_id` 数据结构契约**（贯穿 Stage5/7/7.1/9.5 及 Stage12/13/14/17 的 per-clip 文件布局）。推荐 **方案 C（Stage4 改为 1:1 动态长度 clip 的薄适配层）**，详见 §5。

---

# 1. Current Pipeline

当前 Face_Pipeline_V2 完整数据流（0–17）：

```
Stage0  Video Integrity Check           → VideoInfo
Stage0.5 Resolution Filter              → ResolutionInfo（过滤过小分辨率）
Stage1  AV Standardization              → StandardizedVideoInfo（25fps / 16k 音频 / 标准 mp4）
Stage2  Scene Segmentation              → List[SceneInfo]（镜头切分，任意长度）
Stage3  Coarse Face Screening (SCRFD)   → List[SceneInfo]（带 filter_result；min_duration=8s 只是下限，16s 通过）
Stage4  Temporal Clip Generation        → List[ClipInfo]  ← 8s 滑动窗口 / 4s stride【本次审计对象】
Stage5  Full Face Detection & Crop      → List[FaceSequence]（每个 Clip 一个序列，逐帧 FaceInfo）
Stage5.1 Face Tracking (ByteTrack)      → 增强（默认关闭）
Stage5.2 Single-Face Filter             → 增强（默认关闭）
Stage5.3 Temporal Stability             → 增强（默认关闭）
Stage6  Pose Estimation (6DRepNet)      → List[PoseSequence]
Stage7  A/V Sync (SyncNet)              → List[AVSyncInfo]（per-clip offset / LSE-C）
Stage7.1 A/V Alignment                  → List[AVAlignmentInfo]（per-clip 对齐音频，offset 校正）
Stage8  Face Quality (HyperIQA)         → List[FaceQualitySequence]（per-frame score）
Stage9  Quality Fusion & Hard Filter    → List[QualityMetrics]（per-clip 接受/拒绝）
Stage9.1 Quality Ranking                → 增强（默认关闭）
Stage9.5 Clip Export                    → clips/{clip_id}.mp4 + audio/{clip_id}.wav + manifest.json
Stage10 Landmark (InsightFace 106)      → per-frame LandmarkInfo
Stage11 Alignment (affine)              → per-frame AlignedFaceInfo（256×256 对齐人脸图）
Stage12 FLAME (EMICA+SPECTRE)           → outputs/stage12_flame/{clip_id}/flame_features.npz
Stage13 Gaze (MediaPipe)                → outputs/stage13_gaze/{clip_id}/gaze_features.npz
Stage14 Audio (HuBERT)                  → outputs/stage14_audio/{clip_id}/audio_features.npz
Stage15 Statistics                      → 【空 stub，未实现】
Stage16 Annotation                      → 【空 stub，未实现】
Stage17 Fusion & Export                 → TrainingSample (T×112 / T×768) → stage17_multimodal_export/
```

**管线实际接线**（`scripts/stage0_17_e2e_validate.py`）：

| 接线点 | 代码位置 | 说明 |
|---|---|---|
| Stage4 → Stage5 | `scripts/stage0_17_e2e_validate.py:249` 写入 `env["clips"]`；`:261` Stage5 消费 `env["clips"]` | Stage4 输出是 Stage5 的唯一输入 |
| Stage5/7/7.1/9.5 | `:345-352`（Stage7 传 `clip_infos`）、`:368-372`（Stage7.1 传 `clip_infos`）、`:430-435`（Stage9.5 传 `clip_infos`） | 均依赖 `List[ClipInfo]` |
| Stage12/13/14 独立 runner | `:483-505` 导出 `{clip_id}_aligned_faces.json` 与 `{clip_id}_av_alignment.json` | per-clip 输入契约 |
| Stage17 | 独立 runner，读取 Stage12/13/14 的 per-clip NPZ | `sample_id := clip_id` |

---

# 2. Stage4 Responsibility

Stage4（`stage_04_temporal_clip_generation/`）当前负责：

1. **时间切割 / clip 生成**：对每个 `SceneInfo` 用滑动窗口切成固定长度 `ClipInfo`。
   - `clip_generator.py:61` `while current_start + window_size <= scene_end` → 只保留完整窗口，尾部丢弃；
   - `clip_generator.py:66-68` 由 `start_time/duration × target_fps` 计算 `start_frame/end_frame/frame_count`。
2. **clip 元数据管理**：生成 `clip_id = f"{scene.scene_id}_clip_{index:04d}"`（`clip_generator.py:70`）、`start_time/end_time/duration/frame_count`。
3. **数据组织**：`ClipInfo` 是后续所有 per-clip 资产（`clips/{clip_id}.mp4`、`audio/{clip_id}.wav`、`outputs/stage*_*/{clip_id}/`）的**样本主键来源**。

Stage4 **本身不裁切视频、不检测人脸、不做音频** —— 纯逻辑切片（`clip_generator.py:1-8`）。8s 窗口完全来自配置：

- `configs/config.yaml:50-54`：`stage_04.enabled=true, window_size=8, stride=4, target_fps=25`
- `stage.py:99-101`：代码默认值同样 `window_size=8, stride=4`

因此"8 秒"是一个**配置值**，不是代码常量。

---

# 3. Stage5-17 Impact Table

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| Stage | 阶段 | string | stage编号 |
| Current Input | 当前输入 | string | 当前输入数据 |
| Current Output | 当前输出 | string | 当前输出数据 |
| Fixed Window Dependency | 固定窗口依赖 | bool | 是否依赖8秒窗口 |
| Fixed Frame Dependency | 固定帧依赖 | bool | 是否依赖200帧 |
| Impact Level | 影响等级 | enum | None/Low/Medium/High |
| Evidence | 代码证据 | string | 文件和行号 |
| Required Change | 修改建议 | string | 是否需要修改 |

| Stage | Current Input | Current Output | Fixed Window | Fixed Frame | Impact | Evidence | Required Change |
|---|---|---|---|---|---|---|---|
| 5 | `List[ClipInfo]` | `List[FaceSequence]` | 否 | 否 | **Low** | `stage_05_face_detection/stage.py:80`（validate 全为 ClipInfo）；`:172-176`（min/max_frame 并集）；`:252-276`（per-clip 组装 FaceSequence） | 需适配：若完全删 Stage4，需提供"整段 1 个 ClipInfo"或改收 `SceneInfo`（逻辑本身长度无关） |
| 5.1 | `List[FaceSequence]` | `List[FaceSequence]`(track) | 否 | 否 | **None** | `stage_05_1_face_tracking/tracker.py:88,96`（per-frame/per-seq 追踪） | 否 |
| 5.2 | `List[FaceSequence]` | `List[FaceSequence]`(filtered) | 否 | 否 | **None** | `stage_05_2_single_face_filter/single_face_filter.py:112-116`（anomaly_ratio 比值过滤） | 否 |
| 5.3 | `List[FaceSequence]` | `List[FaceSequence]`(stabilized) | 否 | 否 | **None** | `stage_05_3_temporal_stability/temporal_stability.py:135-158`（逐帧中心位移/尺度） | 否 |
| 6 | `List[FaceSequence]` | `List[PoseSequence]` | 否 | 否 | **None** | `stage_06_pose/stage.py:77`（batch_size=32 仅推理批）；`pose_estimator.py:86-124`（逐帧批推理） | 否 |
| 7 | `{sequences, audio_info, clip_infos}` | `List[AVSyncInfo]` | 否（SyncNet 仅要求**最小**时长，无上限） | 否 | **Low** | `sync_estimator.py:73`（fps=frame_count/duration）；`:84`（按 `clip.start_time/duration` 切音频）；`stage.py:87-91`（clip_by_id 索引） | 需适配：依赖 `ClipInfo`（时长/起止）；16s 视频 SyncNet 可直接跑 |
| 7.1 | `{av_sync_infos, std_video_info, clip_infos(可选)}` | `List[AVAlignmentInfo]` | 否 | 否 | **Low** | `stage.py:23-35`（start_time/duration 从 sync 或 clip_info 派生）；`processor.py:181-278`（按 clip_start/clip_duration 裁切+shift 对齐音频） | 需适配：依赖 `ClipInfo`（时长），duration 动态 |
| 8 | `List[FaceSequence]` | `List[FaceQualitySequence]` | 否 | 否 | **None** | `stage_08_quality/stage.py:81-88`（逐序列 per-frame quality） | 否 |
| 9 | `{pose_sequences, av_sync_infos, quality_sequences, video_id}` | `List[QualityMetrics]` | 否（`max_offset_frames=8` 是 **SyncNet offset 阈值**，非窗口） | 否 | **None** | `fusion.py:43`（max_offset_frames=8 注释 "hard filter: \|offset\|<=N frames"）；`stage.py:189-237`（per-clip 融合） | 否 |
| 9.1 | `List[QualityMetrics]` | ranking/tier | 否 | 否 | **None** | `stage_09_1_quality_ranking/processor.py:137`（per-clip 加权分） | 否 |
| 9.5 | `{quality_metrics, clip_infos, std_video_info, av_alignments}` | `clips/{id}.mp4` + `audio/{id}.wav` + manifest | 否（按 `clip.duration` 动态裁切） | 否 | **Medium** | `exporter.py:97-166`（`clip=clip_by_id[qm.clip_id]`；`:107` 按 `clip.start_time/duration` ffmpeg 裁视频；`:117-122` audio/{id}.wav）；`stage.py:50-84`（输入含 clip_infos） | 需适配：依赖 `ClipInfo` + `clip_id` 资产键；16s 可裁但整段成为单一资产 |
| 10 | `{face_sequences, accepted_clip_ids}` | `List[LandmarkInfo]` | 否 | 否 | **None** | `processor.py:25-115`（逐帧 106 点 landmark） | 否 |
| 11 | `{landmarks, face_sequences}` | `List[AlignedFaceInfo]` | 否 | 否 | **None** | `processor.py:123-338`（逐帧 affine 对齐，键=(clip_id,frame_idx,face_id)） | 否 |
| 12 | `{clip_id}_aligned_faces.json` | `{clip_id}/flame_features.npz` | 否（batch_size=8 仅推理批） | 否（T=len(faces) 动态） | **None** | `runner.py:138-139`（对全部 faces 分批）；`integration.py:45,138`（per-clip 目录）；`stage.py:217`（clip_id 取自 aligned_faces[0]） | 否（T=400 直接支持） |
| 13 | `{clip_id}_aligned_faces.json` | `{clip_id}/gaze_features.npz` | 否 | 否（`T=len(features)`） | **None** | `integration.py:37`（T=len(features)） | 否（T=400 直接支持） |
| 14 | `{clip_id}_av_alignment.json` | `{clip_id}/audio_features.npz` | 否 | 否（fps=25 重采样，T 动态） | **None** | `processor.py:116-174`（50Hz→25fps 重采样，T 随 clip 时长）；`runner.py:62-67`（clip_start/clip_duration 动态） | 否（16s→约 400 特征，±1 帧容差已在 Stage17 处理） |
| 15 | — | — | 否 | 否 | **None** | `stage_15_statistics/__init__.py`（0 字节空 stub） | 否（未实现） |
| 16 | — | — | 否 | 否 | **None** | `stage_16_annotation/__init__.py`（0 字节空 stub） | 否（未实现） |
---

# 4. Detailed Analysis

> 通用结论：Stage5–17 的推理/逻辑全部是 **per-frame 或 per-sequence**，T 动态；没有一处要求固定 8s / 200 帧。下面逐阶段给出输入输出、时长依赖、结构依赖与修改点。

## Stage5（Face Detection & Crop）

- **输入/输出**：`List[ClipInfo]` → `List[FaceSequence]`（每个 Clip 一个序列，每帧 `FrameInfo`+`FaceInfo`）。
- **时长依赖**：无固定 8s/200 帧。它只对 `clips` 的帧**并集**做 SCRFD 检测：
  - `stage.py:172-176`：`min_frame=min(c.start_frame)`、`max_frame=max(c.end_frame)`、`unique_frames_needed` 并集；
  - `stage.py:180-247`：ffmpeg 顺序解码 `[min_frame,max_frame)`，命中并集的帧才检测并缓存；
  - `stage.py:252-276`：Phase 2 按每个 clip 的 `[start_frame,end_frame)` 从缓存组装 `FaceSequence`。
- **结构依赖**：**强依赖 `ClipInfo`**（`stage.py:80` 校验输入全为 ClipInfo；`clip_id` 用于 face_id/裁剪目录）。若 16s 整段作为 1 个 ClipInfo 传入，`min==该段 start_frame`、`max==end_frame`，去重缓存自然退化为单窗口，逻辑零改动即可工作。
- **修改**：仅"谁提供 ClipInfo"。**Low**。

## Stage5.1 / 5.2 / 5.3（增强模块，默认关闭）

- 输入：`List[FaceSequence]`；输出：tracked / filtered / stabilized 序列。
- 5.1 ByteTrack：per-frame 追踪（`tracker.py:88,96`），任意长度。
- 5.2 单人脸过滤：`anomaly_ratio = anomaly_frames / total_frames`（`single_face_filter.py:112-113`），**比值**与长度无关。
- 5.3 时间稳定性：逐帧 center motion / scale change（`temporal_stability.py:135-158`），同样比值化。
- **修改**：无。**None**。

## Stage6（Pose Estimation, 6DRepNet）

- 输入：`List[FaceSequence]`；输出：`List[PoseSequence]`。
- 逐帧批推理，`batch_size` 只是 GPU 推理批大小（`stage.py:77` / `pose_estimator.py:86-124`），与 clip 长度无关。
- **修改**：无。**None**。

## Stage7（A/V Sync, SyncNet）

- 输入：`{"sequences", "audio_info", "clip_infos"}`；输出：`List[AVSyncInfo]`。
- **时长依赖**：不固定 8s。`estimate_sync`（`sync_estimator.py`）：
  - `:73` `fps = clip.frame_count / clip.duration`（动态）；
  - `:84` 按 `clip.start_time / clip.duration` 从全片音频裁 clip 音频；
  - `:87-97` 用人脸 crop 拼视频 + 混流 clip 音频，交给 SyncNet。
  - SyncNet 官方实现只要求**最小**时长（需足够帧计算 MFCC），16s 无上限问题；反而更长窗口平均掉单帧噪声，LSE-C 更稳。
- **结构依赖**：**依赖 `ClipInfo`**（`stage.py:87-91` `clip_by_id[seq.clip_id]` 取起止/时长）。若 16s 单 Clip 传入，正常。
- **修改**：仅"谁提供 ClipInfo"。**Low**。

## Stage7.1（A/V Alignment）

- 输入：`{"av_sync_infos", "std_video_info", "clip_infos"(可选)}`；输出：`List[AVAlignmentInfo]`。
- 对每个 clip 先 `extract_clip_audio`（`processor.py:181-212`，按 `clip_start/clip_duration`），再 `shift_clip_audio`（`:214-278`，offset 校正且 duration 严格保持）。duration 全部动态。
- 附注：`stage.py` 中 "duration 保持 8s" 是**过时注释**（`:80` 附近），代码实际用 `duration` 变量，无 8s 常量。
- **修改**：仅"谁提供 ClipInfo"。**Low**。

## Stage8（Face Quality, HyperIQA）

- 输入：`List[FaceSequence]`；输出：`List[FaceQualitySequence]`（per-frame score）。
- 逐帧质量评分（`stage.py:81-88`），与 clip 长度无关。
- **修改**：无。**None**。

## Stage9（Quality Fusion & Hard Filter）

- 输入：`{"pose_sequences", "av_sync_infos", "quality_sequences", "video_id"}`；输出：`List[QualityMetrics]`。
- 所有硬过滤维度都是**比值/阈值**（pose_ratio、LSE-C、offset 帧数、quality 均值），与 clip 长度无关。
- ⚠️ 注意：`fusion.py:43` 的 `max_offset_frames: int = 8` 是 **SyncNet offset 绝对值阈值**（\|offset\|≤8 帧），**不是窗口长度**。同理 `config.yaml` `sync.max_offset_frames: 8`。审计时勿混淆。
- **修改**：无。**None**。
- 数据影响提示：8s→16s 后，一个整段内任一坏窗口（pose 超限 / offset 超限 / 低质量帧比例超限）会**整段拒绝**，而不再只是拒绝其中 8s 子窗。需要训练同事确认 16s 全段接受率可接受。

## Stage9.1（Quality Ranking，默认关闭）

- 输入：`List[QualityMetrics]`；输出：加权分 + 档位（excellent/good/normal）。
- per-clip 加权分排序（`processor.py:137`），比值化，与 clip 长度无关。
- **修改**：无。**None**。

## Stage9.5（Clip Export — 训练资产产出）

- 输入：`{"quality_metrics", "clip_infos", "std_video_info", "av_alignments"}`；输出：`clips/{clip_id}.mp4`、`audio/{clip_id}.wav`、`manifest.json`。
- 对每个 accepted clip（`exporter.py:97-166`）：
  - `:97-101` `clip = clip_by_id[qm.clip_id]`；
  - `:104-110` 按 `clip.start_time / clip.duration` 用 ffmpeg 从标准化视频裁 `clips/{clip_id}.mp4`；
  - `:113-122` 复制 Stage7.1 对齐音频 → `audio/{clip_id}.wav`；
  - `:143-166` 写 manifest 条目（含 `duration=round(clip.duration,2)`）。
- **时长依赖**：动态（`clip.duration`），16s 可直接裁。
- **结构依赖**：**强依赖 `ClipInfo` + `clip_id` 资产键**——这是把"16s 整段"物化成物理视频/音频资产的唯一关卡。整段 1 个 clip_id → 产出 1 个 16s mp4 + 1 个 16s wav，正是 Stage12/13/14 的输入。
- **修改**：仅"谁提供 ClipInfo"。但因涉及最终资产格式/命名/清单，且是所有后续 runner 的源头，评级 **Medium**（风险主要在于改动面广，而非逻辑复杂度）。
- 附注：`manifest.json`（Stage9.5 产物）与 Stage17 交付 `manifest.csv`（`export.py:83-88` 字段 `clip_id/num_frames/fps`）都以 `clip_id` 为主键，格式兼容任意 T。

## Stage10（Landmark, InsightFace 106）

- 输入：`{"face_sequences", "accepted_clip_ids"}`；输出：`List[LandmarkInfo]`。
- 逐帧 landmark 提取（`processor.py:25-115`），`accepted_clip_ids` 只是 per-clip 过滤开关，长度无关。
- **修改**：无。**None**。

## Stage11（Alignment, affine）

- 输入：`{"landmarks", "face_sequences"}`；输出：`List[AlignedFaceInfo]`。
- 逐帧 affine 对齐，键 `(clip_id, frame_index, face_id)`（`processor.py:123-338`），输出 256×256 对齐图，长度无关。
- **修改**：无。**None**。

## Stage12（FLAME, EMICA + SPECTRE）

- 输入：`{clip_id}_aligned_faces.json`（Stage11 产出的 per-frame 对齐人脸记录）；输出：`outputs/stage12_flame/{clip_id}/flame_features.npz`。
- 逐帧编码，`batch_size=8` 仅是**推理批大小**（`runner.py:138-139` 对全部 faces 分批；`stage.py:77`），不是时间窗口。
- `T = len(faces)` 动态 → **T=400（16s @25fps）直接支持**。输出数组 expression/jaw/head 长度跟随帧数。
- **结构依赖**：per-clip 目录 `output_dir/{clip_id}/`（`integration.py:138`），`clip_id` 取自 `aligned_faces[0].clip_id`（`stage.py:217`）。
- **修改**：无。**None**。

## Stage13（Gaze, MediaPipe）

- 输入：`{clip_id}_aligned_faces.json`；输出：`outputs/stage13_gaze/{clip_id}/gaze_features.npz`。
- `T = len(features)`（`integration.py:37`）→ **T 动态，T=400 直接支持**；`frame_index/timestamp/clip_id` 逐帧记录。
- **修改**：无。**None**。

## Stage15 / Stage16（Statistics / Annotation）

- 均为**空 stub**（`stage_15_statistics/__init__.py`、`stage_16_annotation/__init__.py`，0 字节），未实现、未被 e2e 调用。
- **修改**：无。**None**。不影响 Stage4 移除。

## Stage17（Fusion & Multimodal Export）

- 输入：Stage12/13/14 的 per-clip NPZ；输出：`TrainingSample`（`visual_feature[T,112]` / `audio_feature[T,768]` / `frame_index` / `timestamps` / `audio_match_idx` / `dropout_mask` / `valid_length`）+ `outputs/stage17_multimodal_export/`（`flame/{clip_id}.npy`、`audio/{clip_id}.npy`、`timestamp/{clip_id}.npz`、`metadata/{clip_id}.json`、`manifest.csv`）。
- **时长依赖**：完全动态。`processor.py:217-218` `T = len(flame_sorted)`（主时间轴=FLAME）；Gaze 按 frame_index hash join + forward-fill；Audio 按 nearest timestamp 匹配（容差 0.04s）。→ **T=400 直接支持**（FLAME 400 / Gaze 400 / Audio 400 对齐，`valid_length≈400`）。
- **结构依赖**：`sample_id := clip_id`（`models.py:777-793` TrainingSample 注释与实现），是唯一 clip 级标识来源。
- **修改**：无。**None**。唯一提醒：Stage17 设计文档（`docs/Stage17_Design_Review.md:77,106,184`）目前以"8s≈200 帧"描述样本窗口语义——这是**文档/语义层**，需同步更新为 16s≈400。

---

## 隐藏 8s / 200 帧依赖汇总（全仓库核查结果）

| 类别 | 位置 | 性质 | 是否构成阻塞 |
|---|---|---|---|
| Stage4 配置 | `configs/config.yaml:50-54`；`stage_04_temporal_clip_generation/stage.py:99-101` | `window_size=8, stride=4` 纯配置 | 否（改配置/加模式即可） |
| Stage3 min_duration | `configs/config.yaml:40` `min_duration: 8`；`stage_03_face_screening/stage.py:130` | **下限**过滤，16s 通过 | 否（16s > 8s） |
| Stage9 offset 阈值 | `fusion.py:43`、`config.yaml` `sync.max_offset_frames: 8` | SyncNet **offset 帧数阈值**，非窗口 | 否（语义不同） |
| 文档 | `docs/03_Pipeline_Stage_Specification.md:622-629`（Window Length 8s / Step 4s）；`docs/Stage17_Design_Review.md:77,106,184`（FULL_WINDOW=200、8s≈200帧）；`docs/Stage12_14_DataFlow_Compatibility_Report.md:156-161`（T_video=200、T_h=400） | 设计/规范文档 | 否（需随方案更新文档） |
| 测试夹具 | `tests/test_datamodel.py:101`（`ci.frame_count == 200`）；`tests/test_stage_03_coarse_face_screening.py:450`（8s 场景夹具）；`scripts/stage0_17_e2e_stage12.py:98,111-112`（`--num_frames 200` 默认 0=全部） | 8s/200 仅作测试 fixture / CLI 可选参数 | 否（实现无硬约束；测试需按新语义调整） |
| 代码硬约束 | 全仓库 grep `window/stride/8s/200帧/frame_count==200` | **无**（Stage5–17 全部 per-frame / 动态 T） | 否 |

---

# 5. Final Recommendation

按代码实际情况，评估三个方案：

## 方案 A：完全删除 Stage4

- **改动面**：Stage5（`stage.py:80` 校验 + `:172-176` 并集逻辑）、Stage7（`stage.py:87-91` 输入契约）、Stage7.1（`stage.py:23-35`）、Stage9.5（`exporter.py:97-110`）、e2e 接线（`scripts/stage0_17_e2e_validate.py:249,261,345-372,430-435`）、Stage12/13/14 的 per-clip JSON 导出（`:483-505`）、per-clip 目录布局（`outputs/stage*_*/{clip_id}/`）、Stage17 `sample_id`。还要重命名 `clip_id` 语义 → `segment_id`，波及 models、manifest、报告脚本。
- **复杂度**：**High**。需要把"clip"数据契约替换为"segment"契约并回归验证 13 个 stage；收益仅是"少一个目录/少一段代码"。
- **风险**：破坏大量已验证的单测（`tests/test_datamodel.py:101` 等）与端到端报告脚本。

## 方案 B：Stage4 保留为可选切片模式（config 开关）

- 做法：`stage_04.enabled=false` 时，Stage4 透传（或 e2e 跳过 Stage4 并构造 1 个 ClipInfo/SceneInfo）。
- 复杂度：**Low-Medium**。但"禁用后仍要造 ClipInfo"与方案 A 同源——只要下游还吃 `List[ClipInfo]`，就绕不开"谁造整段 ClipInfo"。适合需要"随时切回 8s/4s"的过渡期。

## 方案 C：Stage4 改为动态长度 clip（1:1 段→clip，推荐）

- 做法：给 Stage4 增加一个模式（例如 `stage_04.mode: segment` 或 `window_size: 0` 表示"整段"），使 `generate_clips` 对每个 passed SceneInfo 产出 **1 个覆盖全段的 ClipInfo**（`clip_id = scene.scene_id`，`start/end = scene` 全范围，`duration = scene.duration`，`frame_count = duration×fps`）。8s/4s 滑动窗口逻辑保留为 `mode: sliding`。
- **改动量**：仅 `clip_generator.py` + `stage.py`（新增分支，约 20-30 行）+ `config.yaml`（加 mode 字段）。**下游 13 个 stage 零改动**——因为它们本来就 per-frame/动态 T。
- **效果**：数据流在架构图上仍是 "Stage3 → Stage4 → Stage5…"，但行为完全等价于"取消 8s 切片"：16s 段 → 1 clip → Stage9.5 产 16s mp4/wav → Stage12/13/14 产 T≈400 → Stage17 `T=400` 样本，**完全符合训练同事目标**。
- **附加工作（无论 A/B/C 都要做）**：
  1. 更新文档：`docs/03_Pipeline_Stage_Specification.md`（3.7 节 Window 8s→动态）、`docs/Stage17_Design_Review.md`（8s≈200→16s≈400 样本语义）；
  2. 更新测试夹具：`tests/test_datamodel.py:101`（frame_count=200→400 或动态）、Stage3 8s 夹具；
  3. 与训练同事确认 **Stage9 整段接受策略**：16s 单样本下，任一坏子窗口会整段拒绝，样本量从 16s→3 条 8s 变为 16s→1 条 16s；
  4. 跑 `scripts/stage0_17_e2e_validate.py` + Stage12/13/14/17 回归验证 T=400 全链路。

## 推荐

> **推荐 方案 C**（Stage4 保留为动态长度 clip 的薄适配层）：以约 1/10 的改动量获得与"完全删除"完全一致的行为，下游 Stage5–17 **全部无需修改**（审计确认均为 per-frame / 动态 T），同时保留随时切回 8s/4s 的灵活性。若团队坚持"架构上抹掉 Stage4"，则退化为方案 A（High 复杂度，需重构 ClipInfo→segment 契约），在完成方案 C 验证后再做纯代码层面的搬迁，风险最低。




