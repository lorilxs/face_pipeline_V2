# Stage6-9 可选执行模式审计报告（Mode A / Mode B）

> 审计性质：**纯代码审计，不修改任何代码**。
> 审计范围：Stage 6 / 7 / 7.1 / 8 / 9，以及它们对 Stage 10 / 11 / 12 / 13 / 14 / 17 的影响。
> 目标：评估"跳过质量筛选、直接处理已有高质量数据集"（Mode B：Stage0-5 → 10 → 11 → 12 → 13 → 14 → 17）是否可行，并给出所需改动清单。
> 证据基准：`face_pipeline_v2/` 源码、`scripts/stage0_17_e2e_validate.py`（当前唯一真实的 Stage0-17 接线驱动）、`configs/config.yaml`。

---

## 0. 结论先行（Executive Summary）

1. **Mode B（dataset 模式）在架构上可行**，但存在 **两个硬性断点** 必须在接线层（wiring layer）解决，而不是靠逐个把 `enabled: false`：
   - **断点① Stage10 的 `accepted_clip_ids` 目前只能由 Stage9 产出**。跳过 Stage6-9 后，需要接线层把"全量 clip"注入该集合（Stage10 本身对任意集合已兼容，见 §2.5）。
   - **断点② Stage14 硬依赖 Stage7.1 的 `aligned_audio_path`**（文件必须存在，缺失时 `validate_input` 失败、官方 runner `SystemExit(1)`）。跳过 Stage7/7.1 后，需要回退到"从 Stage1 原始音频按 clip 窗口裁剪"的原始 wav（可复用 Stage7.1 处理器中已验证的 `extract_clip_audio`，offset=0）。
2. **不建议** 依赖各质量阶段现有的 `enabled` 标志做"单阶段跳过"：Stage6/7/8/9/7.1 禁用时 `process()` 统一返回 `[]`，而这会直接打爆下游 `validate_input`（Stage9 要求 pose/sync/quality 三表非空、Stage7.1 要求 av_sync_infos 非空），`BaseStage.run()` 会抛 `InputValidationError`。
3. **推荐方案**：新增顶层配置 `pipeline_mode: open_domain | dataset`，在接线层（e2e 驱动 / 尚未实现的 `pipeline_runner.py`）按模式分派链路；**不删除任何 stage、不改 Stage6-9 内部语义**。Mode A 路径保持零改动。
4. **Stage10-17 对 Stage6/7/8 的产物零直接引用**（已 grep 验证）：Stage10/11/12/13/17 目录中无 `PoseSequence` / `AVSyncInfo` / `FaceQualitySequence` / `QualityMetrics` 引用；Stage14 是唯一直接消费质量段产物的下游（消费 Stage7.1 的 `AVAlignmentInfo`）。Stage12 的 head_pose 由 FLAME 网格自身计算，与 Stage6 无关。

---

## 1. Stage 依赖关系图（Dependency Graph）

### 1.1 审计范围内各 Stage（Stage6 / 7 / 7.1 / 8 / 9）

| stage | 输入 (input) | 输出 (output) | 下游依赖 (downstream_dependency) | 是否可关闭 (optional) | 影响 (impact) |
|---|---|---|---|---|---|
| **Stage6 Pose**（`stage_06_pose/stage.py`） | `List[FaceSequence]`（来自 Stage5.x） | `List[PoseSequence]`（逐帧 yaw/pitch/roll） | Stage9（`pose_sequences`）；Stage9.1 / 9.5（间接，读 `QualityMetrics.pose_score`） | 有 `enabled` 标志（`stage_06.enabled`），但禁用返回 `[]` → **非安全跳过** | 单独禁用会让 Stage9 `validate_input` 抛 `InputValidationError`（要求 pose 表非空）；对 Stage10/11/12 无直接数据依赖 |
| **Stage7 Sync**（`stage_07_sync/stage.py`） | `dict{sequences, audio_info, clip_infos}` | `List[AVSyncInfo]`（offset / LSE-C / LSE-D + clip 边界） | Stage7.1（`av_sync_infos`）；Stage9（`av_sync_infos`）；Stage9.1/9.5（间接） | 有 `enabled` 标志（`stage_07.enabled`），禁用返回 `[]` → **非安全跳过** | 单独禁用会让 Stage7.1 与 Stage9 的 `validate_input` 同时崩；Stage14/17 音频链路断 |
| **Stage7.1 AV Alignment**（`stage_07_1_av_alignment/stage.py`） | `dict{av_sync_infos, std_video_info, clip_infos?}` | `List[AVAlignmentInfo]`（`aligned_audio_path`、`raw_audio_path`、offset、status） | **Stage14（硬依赖 `aligned_audio_path`）**；Stage9.5（canonical audio 导出）；Stage17（经 Stage14 间接） | 有 `enabled` 标志（`stage_07_1_av_alignment.enabled`），禁用返回 `[]` → **非安全跳过** | 禁用后 Stage14 无输入（`validate_input` 失败 / runner `SystemExit(1)`）；Stage9.5 可优雅降级为 video-only（`audio_unavailable`） |
| **Stage8 Quality**（`stage_08_quality/stage.py`） | `List[FaceSequence]` | `List[FaceQualitySequence]`（逐帧 HyperIQA 分） | Stage9（`quality_sequences`）；Stage9.1/9.5（间接） | 有 `enabled` 标志（`stage_08.enabled`），禁用返回 `[]` → **非安全跳过** | 单独禁用会让 Stage9 `validate_input` 崩；对 Stage10-17 无直接数据依赖 |
| **Stage9 Fusion**（`stage_09_quality_fusion/stage.py`） | `dict{pose_sequences, av_sync_infos, quality_sequences, video_id}` | `List[QualityMetrics]`（`is_accepted` 门控 + fusion 分） | **Stage10（`accepted_clip_ids` 门控）**；Stage9.1（`quality_metrics`）；Stage9.5（accepted → 导出）；Stage11/12/13/14/17（经 Stage10/11 metadata 间接） | 有 `enabled` 标志（`stage_09.enabled`），禁用返回 `[]` → **非安全跳过** | 禁用后 `accepted_ids` 为空 → e2e 提前返回 PARTIAL / Stage10 输出空 → 全下游 10-17 无声量 |

### 1.2 下游 Stage10-17 消费图（证明质量段产物的流向边界）

| 下游 Stage | 输入 | 是否消费 Stage6/7/8/9 产物 |
|---|---|---|
| **Stage10 Landmark**（`stage_10_landmark/stage.py`） | `dict{face_sequences(Stage5), accepted_clip_ids(Stage9)}` | 只消费 Stage9 的"接受集合"；**不消费** Pose/Sync/Quality 数据对象 |
| **Stage11 Alignment**（`stage_11_alignment/stage.py`） | `dict{landmarks(Stage10), face_sequences(Stage5)}` | **不消费** |
| **Stage12 FLAME**（`stage_12_flame/runner.py` + processor） | `<clip_id>_aligned_faces.json`（含 `aligned_face_path` 图片） | **不消费**；head_pose 由 FLAME 网格计算，与 Stage6 无关（processor 无 Stage6 引用） |
| **Stage13 Gaze**（`stage_13_gaze/runner.py`） | `<clip_id>_aligned_faces.json` | **不消费** |
| **Stage14 Audio**（`stage_14_audio_feature/runner.py` + `stage.py`） | `<clip_id>_av_alignment.json` → `AVAlignmentInfo`（**`aligned_audio_path` 必须存在**） | **唯一直接消费质量段产物的下游**（消费 Stage7.1） |
| **Stage17 Fusion**（`stage_17_fusion/runner.py`） | Stage12/13/14 的 NPZ 输出目录 | **不消费**（只读 12/13/14 的 NPZ） |

> 证据：`grep -rn 'PoseInfo|PoseSequence|AVSyncInfo|FaceQualitySequence|QualityMetrics'` 在 `stage_10_landmark/`、`stage_11_alignment/`、`stage_12_flame/`、`stage_13_gaze/`、`stage_17_fusion/` 目录 **零命中**；仅在 `stage_14_audio_feature/` 命中（且只涉及 `AVAlignmentInfo`）。
---

## 2. 跳过可行性逐项验证（Verify Skip Possibility）

> 通用结论（先看）：Stage6/7/8/9/7.1 的 `enabled: false` 在各自 `process()` 里只是 `return []`（`stage_06_pose/stage.py:70-75`、`stage_07_sync/stage.py:84-86`、`stage_08_quality/stage.py:72-74`、`stage_09_quality_fusion/stage.py:163-167`、`stage_07_1_av_alignment/stage.py:152-155`）。而下游 `validate_input` 要求非空列表（Stage9：`stage_09_quality_fusion/stage.py:87-126`；Stage7.1：`stage_07_1_av_alignment/stage.py:83-93`），`BaseStage.run()` 在 `validate_input` 返回 False 时抛 `InputValidationError`（`pipeline/base_stage.py:132-137`）。因此"逐个置 false" 会 FAIL，**安全跳过必须发生在接线层**（不调用该 stage 并向下游注入替代输入）。

### 2.1 Stage6 disabled → 是否影响 Stage10 / Stage11 / Stage12？

| 下游 | 结论 | 证据 |
|---|---|---|
| Stage10 | **无直接数据依赖**；仅间接受影响：Stage6 禁用 → `pose_sequences=[]` → Stage9 `validate_input` 崩 → e2e FAIL，Stage10 不会被调用。若在接线层同时跳过 Stage9 并注入 `accepted_ids=全量`，则 Stage10 完全不受影响 | Stage10 输入仅 `{face_sequences, accepted_clip_ids}`；Stage10 目录无 `PoseSequence` 引用 |
| Stage11 | 同 Stage10（间接，经 Stage9 门控），无直接依赖 | Stage11 输入仅 `{landmarks, face_sequences}` |
| Stage12 | **完全不受影响**。Stage12 的 head_pose（`head_pose.npy`）由 FLAME/SPECTRE 网格自身计算，与 Stage6 的 6DRepNet 姿态无关；processor 中无任何 Stage6/PoseInfo 引用 | `stage_12_flame/processor.py` 仅读 `aligned_face_path` 图片 |

**结论：Stage6 可跳；对 Stage10/11/12 无直接数据影响，唯需在接线层解决 Stage9 门控的替代。**

### 2.2 Stage7 disabled → Stage14 音频输入是否需要修改？

**需要修改（或等价地在接线层合成回退输入）。** 原因链：
1. Stage14 的输入是 `AVAlignmentInfo`（来自 Stage7.1），且 **`aligned_audio_path` 必须存在**：
   - `stage_14_audio_feature/stage.py` 的 `validate_input`：`aligned_audio_path is None` → 返回 False；文件不存在 → 返回 False；
   - 官方 runner `stage_14_audio_feature/runner.py:97-102`：`aligned_audio_path is None or not exists()` → `logger.error(...)` + `raise SystemExit(1)`。
2. Stage7.1 的输入是 `av_sync_infos`（来自 Stage7），`validate_input` 要求非空（`stage_07_1_av_alignment/stage.py:83-93`）。Stage7 禁用 → `av_sync_infos=[]` → Stage7.1 崩（或自身也禁用 → 无 `AVAlignmentInfo` 产出）。
3. 因此 Stage7 禁用后，Stage14 拿不到任何有效 `aligned_audio_path`，**必须回退到原始音频**：
   - 推荐回退源：Stage1 的标准化音频 `StandardizedVideoInfo.audio_path`，按 clip 窗口 `[clip_start, +clip_duration]`（来自 `ClipInfo` / `AVSyncInfo` 边界字段）用 ffmpeg 裁剪成 clip 级 wav（offset=0、status=`already_aligned`），合成一个 `AVAlignmentInfo` 交给 Stage14。
   - 该裁剪逻辑已存在且经过测试：`stage_07_1_av_alignment/processor.py:181 extract_clip_audio(...)`，可直接复用（offset=0 即纯裁剪、不 shift）。

**结论：Stage7 可跳，但 Stage14 音频输入需要"原始 wav 回退"的接线层改动（或 Stage14 自身增加 fallback）。**

### 2.3 Stage7.1 disabled → 是否需要 fallback original wav？

**是，需要。** 当前不存在自动回退：
- Stage7.1 禁用 → `av_alignments=[]` → 接线层导出的 `<clip_id>_av_alignment.json` 缺失 → Stage14 无输入（同上 runner `SystemExit(1)`）。
- Stage9.5 对缺失 align 是**优雅降级**的（`exporter.py:120-131`：`aligned_audio_path` 缺失 → `audio_unavailable += 1`，仅导出视频、无 canonical 音频）——但 Stage14/17 没有这种降级。
- 因此 dataset 模式下，若跳过 Stage7.1，必须为每个 clip 准备"原始 wav 回退"（Stage1 音频按 clip 窗口裁剪，offset=0、status=`already_aligned`），再生成 `AVAlignmentInfo` 元数据。

**结论：需要 fallback original wav；推荐复用 `extract_clip_audio`（offset=0）在接线层合成 `already_aligned` 的 `AVAlignmentInfo`。**

### 2.4 Stage8 disabled → 是否影响 Stage9？

**是，直接影响且会崩。** Stage9 `validate_input` 要求 `quality_sequences` 为非空 `FaceQualitySequence` 列表（`stage_09_quality_fusion/stage.py:117-126`）；Stage8 禁用 → `quality_sequences=[]` → `validate_input` 返回 False → `BaseStage.run()` 抛 `InputValidationError` → e2e 该 Stage 记为 FAIL。

**结论：Stage8 不能单独禁用而不处理 Stage9**（要么接线层同时跳过 Stage9，要么让 Stage9 在缺失 quality 维度时跳过该 gate——后者会改变 Stage9 硬过滤语义，不推荐）。

### 2.5 Stage9 disabled → 是否影响 Stage10？

**是，功能性影响（无声量，而非崩溃）。**
- Stage10 `validate_input` 对空 `accepted_clip_ids` 的处理是"非硬错误"：`stage_10_landmark/stage.py:99-105` 明确注释 `# Not a hard error — just nothing to do.` 并返回 True；`process()` 只处理 accepted 集合内的 clip → 空集合输出 `[]`。
- 但在 e2e 接线层有提前短路：`scripts/stage0_17_e2e_validate.py`（Stage9 后）`if not accepted_ids: return _finalize(..., "PARTIAL")` → Stage10 根本不会被调用。
- 即使绕过短路，Stage10 → `[]` → Stage11 `[]` → 无 aligned_faces → Stage12/13 无 metadata、Stage14/17 无声量。

**结论：dataset 模式必须在接线层为 Stage10 注入 `accepted_ids = 全部 clip_id`（或配置指定的子集）；Stage10 本身无需改动。** 这也是"跳过 Stage6-9 后整链 10-17 仍产出样本"的**关键接线改动**。

---

## 3. 推荐架构（Recommend Architecture）

### 3.1 原则

- **不删除任何 Stage**（Stage6/7/7.1/8/9 在 Mode A 下完整保留、零改动）。
- 质量筛选是一个**整体可选模块**（6→7→7.1→8→9），不是 5 个可独立开关的 stage；因此"跳过"应在**接线层**做整块分支，而不是逐个 `enabled=false`。
- 复用现有已测试的纯函数（`extract_clip_audio`、`fuse_quality` 等）合成 dataset 模式的替代输入，避免为 Mode B 复制实现。

### 3.2 推荐：顶层 `pipeline_mode` 配置

新增顶层键（ConfigLoader 只校验 required 键存在、不拒绝未知键，见 `config/config_loader.py:79-101`，低风险）：

```yaml
pipeline_mode: open_domain    # open_domain | dataset
```

| pipeline_mode | 链路 | 语义 |
|---|---|---|
| `open_domain`（默认，= 现 Mode A） | Stage0-17 全链 | open-domain 视频挖掘，完整质量筛选 |
| `dataset`（= 现 Mode B） | Stage0-5 → (跳过 6/7/7.1/8/9/9.5) → Stage10 → 11 → 12/13/14 → 17 | 已有高质量数据集加工，跳过质量验证 |

`dataset` 模式接线层的两个替代输入：
1. **Stage10 `accepted_clip_ids`** ← `{c.clip_id for c in clips}`（全量接受；也可扩展为配置指定子集）。
2. **Stage14 `AVAlignmentInfo`** ← 对每个 clip 用 `extract_clip_audio(原始Stage1音频, 窗口=clip_start/clip_duration, offset=0)` 生成 clip 级 wav，合成 `alignment_status="already_aligned"` 的 `AVAlignmentInfo`（`aligned_audio_path` 指向该 wav），供 Stage14 与 Stage12/13/14 metadata 导出使用。

### 3.3 备选方案与取舍（不推荐）

- **备选 A：把 Stage6/7/8/9/7.1 改造成 EnhancementStage 语义**（禁用时 passthrough 中性输出，参考 `pipeline/enhancement_stage.py`）。缺点：需要改 Stage9 让它容忍缺失维度（跳过对应 gate），会**污染硬过滤语义**，且改动面大于接线层分支。
- **备选 B：让 Stage14 自身支持 missing-align 回退到原始音频**。可行（改动集中在 `stage_14_audio_feature/stage.py + runner.py`），但会把"音频裁剪"职责下沉到 Stage14，不如在接线层统一合成、保持 Stage14 契约不变。
- 结论：**接线层 `pipeline_mode` 分支 + 复用 `extract_clip_audio`** 是改动最小、风险最低、语义最清晰的方案。

### 3.4 现状提示

- `face_pipeline_v2/pipeline/pipeline_runner.py` 目前是 **0 字节空文件**（`wc -c` = 0），真实接线只存在于 `scripts/stage0_17_e2e_validate.py`。`pipeline_mode` 分支的归属地有两种选择：
  - 短期：直接在 e2e 驱动脚本中按 `config["pipeline_mode"]` 分支；
  - 长期：实现 `pipeline_runner.py` 作为两模式共享入口，e2e 脚本降级为"验证驱动"。
- `configs/config.yaml:165` 的 `sync.correction_enable: true` 目前是**死配置**（全仓库仅 config 出现，代码零读取），与本次审计无关，仅提示可后续清理。

---

## 4. 所需代码改动清单（Identify Required Code Changes）

> 仅列清单，**禁止在本任务中修改**。所有改动均为 Mode B 新增路径；Mode A 路径保持零改动。

| # | 文件 | 函数/位置 | 修改原因 |
|---|---|---|---|
| 1 | `configs/config.yaml` | 新增顶层键 `pipeline_mode: open_domain` | 声明式选择运行模式（默认保持 Mode A）；ConfigLoader 校验不拒绝未知键，无破坏性 |
| 2 | `scripts/stage0_17_e2e_validate.py` | `main()`：Stage6-9 整段分支 | 在 `pipeline_mode=dataset` 时：跳过 Stage6/7/7.1/8/9/9.5 调用；`accepted_ids = 全部 clip_id` 注入 Stage10；Stage14 metadata 改用合成 `AVAlignmentInfo` |
| 3 | （可选，见 §3.4）`face_pipeline_v2/pipeline/pipeline_runner.py` | 实现 `run(config)` | 当前为空文件；作为两模式共享入口，避免逻辑只藏在 e2e 脚本 |
| 4 | `scripts/stage0_17_e2e_validate.py` | 新增辅助函数（如 `_make_dataset_alignment(clip, std_info, ...)`） | 为 Stage14 合成 `already_aligned` 的 `AVAlignmentInfo`（复用 `stage_07_1_av_alignment/processor.py:extract_clip_audio`，offset=0） |
| 5 | （可选，若不想复用 processor）`face_pipeline_v2/stages/stage_14_audio_feature/runner.py` + `stage.py` | `_parse_alignment()` / `validate_input()` | 若选择"Stage14 自身回退原始音频"方案：当 `aligned_audio_path` 缺失时回退 `original_audio_path` + clip 窗口裁剪。**备选方案，非必需**（推荐走 #4） |
| 6 | （可选）`face_pipeline_v2/stages/stage_09_quality_fusion/stage.py` | `validate_input()` | 若希望"仅关闭 6/7/8 但仍保留 Stage9 门控"：将空维度视为"该维度 gate 关闭"。**不推荐**（污染硬过滤语义）；dataset 模式直接跳过 Stage9 更干净 |
| 7 | `docs/02_Data_Model_and_Interface.md` 等 | 文档同步 | 记录 `pipeline_mode` 与 dataset 模式的 Stage14 音频回退约定 |

> 明确**不需要**修改：`stage_06/07/08/09/07_1` 各 stage 内部逻辑（Mode A 语义不动）；`stage_10_landmark`（对空/任意 accepted 集合已兼容）；`stage_11/12/13`（与质量段无数据耦合）；`stage_17_fusion`（只读 Stage12/13/14 NPZ）。

---

## 5. 总结（Summary）

### 5.1 是否支持 dataset 模式（Mode B：0-5 → 10 → 11 → 12 → 13 → 14 → 17）

**架构上支持，但当前代码不能直接支持**，原因是两个"接线断点"：
- 断点①：Stage10 的 `accepted_clip_ids` 由 Stage9 产出，跳过 Stage9 后必须由接线层注入"全量 clip"（Stage10 本身零改动）。
- 断点②：Stage14 硬依赖 Stage7.1 的 `aligned_audio_path`，跳过 Stage7/7.1 后必须回退到"原始 wav 按 clip 窗口裁剪"（可复用 `extract_clip_audio`）。
- 除此之外，Stage10/11/12/13/17 对 Stage6/7/8 产物**零直接依赖**（grep 验证），因此跳过质量段不会破坏下游数据契约。

### 5.2 需要修改哪些地方

最小改动集 = `configs/config.yaml`（加 `pipeline_mode`）+ `scripts/stage0_17_e2e_validate.py`（模式分支 + 两处替代输入注入）+ 一份 Stage14 音频回退辅助函数（复用 `extract_clip_audio`）。可选：实现空文件 `pipeline_runner.py` 作为共享入口；不建议改 Stage9/Stage14 内部语义。

### 5.3 修改风险

| 风险项 | 等级 | 说明 |
|---|---|---|
| Mode A 回归 | **低** | `pipeline_mode` 默认 `open_domain`，分支不触碰原路径；Stage6-9 内部零改动 |
| 新增接线逻辑正确性 | **中** | dataset 分支是全新代码路径，首次运行前需单独验证（可先用已有 e2e 视频跑通再上量） |
| Stage14 音频回退 | **中** | 需保证裁剪出的 clip wav 为 16kHz mono、窗口与 `ClipInfo` 边界一致（segment 模式下 duration 动态，务必按实际 `clip_start/clip_duration` 裁剪）；`extract_clip_audio` 已验证可复用 |
| 配置兼容性 | **低** | ConfigLoader.validate 只查 required 键存在，新增 `pipeline_mode` 无破坏；旧配置缺该键时按 `open_domain` 默认处理即可 |
| Stage12/13 输入 | **低** | dataset 模式下 aligned_faces 由 Stage10/11 正常产出，与 Mode A 相同 |

### 5.4 建议实施顺序

1. **Step 1**：`configs/config.yaml` 新增 `pipeline_mode`（默认 `open_domain`），并在 ConfigLoader 文档/注释中说明取值。风险最低，先行落地。
2. **Step 2**：在 `scripts/stage0_17_e2e_validate.py` 增加 dataset 分支骨架：跳过 Stage6-9/9.5、注入 `accepted_ids=全量`、导出 Stage14 metadata（先用"空/占位"对齐，验证 Stage10/11 在 dataset 模式下正常产出 aligned_faces）。
3. **Step 3**：实现 Stage14 音频回退辅助（复用 `extract_clip_audio` 合成 `already_aligned` 的 `AVAlignmentInfo`），接通 Stage14 → Stage17，用 1 个真实视频跑通 dataset 模式全链。
4. **Step 4**：回归验证 Mode A（全链 Stage0-17 与改动前逐字段一致）+ 新增 dataset 模式冒烟测试；再按需同步文档。
5. **Step 5（可选）**：将共享入口迁入 `pipeline_runner.py`，e2e 脚本退化为验证驱动。

---

*报告生成方式：纯代码审计（读取源码 + 定向 grep 验证），未修改任何代码。证据文件：`stage_06_pose/stage.py`、`stage_07_sync/stage.py`、`stage_07_1_av_alignment/{stage,processor}.py`、`stage_08_quality/stage.py`、`stage_09_quality_fusion/{stage,fusion}.py`、`stage_09_5_clip_export/exporter.py`、`stage_10_landmark/stage.py`、`stage_11_alignment/stage.py`、`stage_12/13/14/17` 的 `runner.py` 与 `stage.py`、`pipeline/base_stage.py`、`pipeline/enhancement_stage.py`、`pipeline/pipeline_runner.py`（空）、`config/config_loader.py`、`configs/config.yaml`、`scripts/stage0_17_e2e_validate.py`。*

