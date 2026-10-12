# Stage17_Design_Review.md

# Stage17 Training Sample Assembly 设计审查报告

Version: v1.0 | Date: 2026-08-27 | 审查方式: 只读审查（未修改任何 Python 文件，未实现 Stage17）

---

## 0. 结论摘要

1. **Stage17 当前完全未落地**：`face_pipeline_v2/stages/stage_17_fusion/` 仅有空 `__init__.py`；
   `face_pipeline_v2/pipeline/pipeline_runner.py` 为 0 字节；`configs/config.yaml` 无 `stage_16/stage_17` 配置段。
2. **旧 Stage5.1 融合逻辑可迁移 4 项**：① 逐帧 FLAME+Gaze 特征拼接（本质与提案 visual_feature 一致）；
   ② 缺失帧「末帧重复」补齐策略；③ 批量预加载 + 索引构建模式（从"逐帧文件/CSV 查找"升级为"按 clip 预载 NPZ"）；
   ④ 样本打包 + manifest 导出。**需删除**：expcode 截断 50 维、逐帧 codes.npz/gaze CSV 文件布局、
   FLAME QC(mahalanobis) 帧过滤、wav2vec2 窗口级音频、固定 200 帧 FULL_WINDOW 语义、float16 音频。
3. **Stage17 三问**：✅ 合并 visual_feature（FLAME+Gaze 单张量）| ✅ 保留 audio 独立 | ✅ 输出 training sample package。
4. **数据格式**：采纳用户提案 `visual_feature[T,112] + audio_feature[T,768] + metadata`，并补充
   `frame_index / timestamps / valid_length / dropout_mask / audio_match_idx / 溯源字段`；
   **明确删除 `eye_pose[6]` 占位块，不把 gaze 写入 FLAMEFeature.eye_pose**（语义不兼容，详见 §4.1）。
5. **时间同步**：FLAME/Gaze 以 `frame_index` 精确 join；Audio 以 `timestamp` 做 nearest 匹配（容差 ±20ms，
   clip 边界容忍 ±40ms）；需防御 `clip_start=None` 导致的音频相对时间偏移。
6. **接口支持度**：FLAMEFeature / GazeFeature / AudioFeature 字段齐全，Stage13/14 NPZ 可直接消费；
   **必须修改 1 处数据模型**（`TrainingSample` 从单帧重构为时间序列/样本包）；Stage12 NPZ 缺 `clip_id`（需以参数回退）；
   Stage14 metadata 缺 `clip_start/fps`（需配置侧补齐）；需新增 `stage_17` 配置段与独立 runner。

---

## 1. 审查范围与现状盘点

### 1.1 当前 Stage 12/13/14 输出现状（事实数据源 = NPZ）

| Stage | 输出 NPZ | 数组 keys | 时间轴 | 备注 |
|---|---|---|---|---|
| 12 FLAME | `{clip_id}/flame_features.npz` | `expression_code[T,100]`, `jaw_pose[T,3]`, `head_pose[T,3]`, `eye_pose[T,6](zeros)`, `shapecode[T,300](zeros)`, `spectre_jawpose[T,3]`, `timestamp[T]`, `frame_index[T]` | 绝对秒（=frame/25） | **未写 `clip_id` 数组**（F2，`read_stage12_outputs` 靠参数回退）；timestamp dtype float64 |
| 13 Gaze | `{clip_id}/gaze_features.npz` | `frame_index[T]`, `timestamp[T]`, `clip_id[T]`, `gaze_vector[T,6]` | 绝对秒 | 最完整；`T` 可能因图片缺失/无检测而 < FLAME |
| 14 Audio | `{clip_id}/audio_features.npz` | `timestamp[N]`, `feature[N,768]`, `dimension`, `source_audio` | 绝对秒（=clip_start+i/25） | **无 `frame_index`/`clip_id`**；metadata 缺 `clip_start/fps`（F5）；HuBERT 50Hz→25FPS，clip 边界 ±1 帧 |

- 三模态 timestamp 同处「标准化视频绝对秒」时间轴（Stage12/13 同源于 `AlignedFaceInfo.timestamp`，Stage14 源于
  `AVAlignmentInfo.clip_start`），可直接跨模态 merge（既有报告 §3.1 已确认）。

### 1.2 旧管线参考（/mnt/home/lxs/complete_single_face_package_20260624/scripts/）

| 脚本 | 功能 | 关键逻辑 |
|---|---|---|
| `handle_stage5_1_flame.py` | FLAME+Gaze 融合 | `load_flame_fast`：codes.npz → `expcode[:50] + posecode[:6]`=56 维；`load_gaze_fast`：CSV → `gaze_0/1 xyz`=6 维；逐帧 `np.concatenate([flame, gaze])`=62 维；按 `history_start_frame..future_end_frame` 窗口取帧；QC 剔除坏帧；**缺失帧末帧重复补齐到 200**；输出 `{video}/{idx}.npy [200,62]` float32 |
| `handle_stage5_2_audio.py` | wav2vec2 音频 | 16kHz mono wav；按 `history_start_seconds..future_end_seconds` 切片；`last_hidden_state [T',768]` float16 存 `{video}/{idx}.npy`（窗口级，非逐帧） |
| `export_final_samples.py` | 最终打包 | 收集 flame/audio .npy → 统一目录 `{video}_{idx}.npy`；生成 `manifest.csv`（sample_name/video_id/窗口边界） |

### 1.3 既有设计文档约定（02 / 03）与差异

- `docs/02 §2.10.3`：`flame_feature = expcode(100) + posecode(6) = 106`，`gaze_feature = 6`，`audio_feature = 768`（**FLAME 与 Gaze 分离**）。
- `docs/02 §2.10.1`：`eye_pose[6]` 为 placeholder，**「由 Stage13 gaze 输出替换」**；`stage_13/stage.py` docstring 亦声明「不修改 eye_pose，那是 Stage17 职责」。
- `docs/03 §3.23`：Stage17 = Parameter Alignment & Fusion，输入 FLAMEFeature/GazeFeature/AudioFeature/Metadata，输出 `TrainingSample`，方法 `Custom Alignment`。
- **与本提案的核心差异**：提案把 gaze 并入 visual_feature（112 维单张量），与旧融合哲学一致；而 doc02 的 106+6 分离也可接受——**两种分组总计都是 112**，仅下游张量组织不同，见 §4.1。
---

## 2. 旧 Stage5.1 FLAME+Gaze 融合逻辑迁移评估

### 2.1 ✅ 可迁移到 Stage17 的逻辑

| # | 旧逻辑 | 迁移方式 | 说明 |
|---|---|---|---|
| M1 | 逐帧 FLAME+Gaze 特征拼接（`concat([flame, gaze])`） | 保留为 `visual_feature = concat([expcode, posecode, gaze_vector])` | 与提案 [T,112] 一致；**expcode 不再截断 50**，用满 100 维 |
| M2 | 缺失帧「末帧重复」补齐（`np.tile(last, pad_count)`） | 保留为缺失策略：前向填充（forward-fill）+ `dropout_mask`/`valid_length` | 旧法只补尾部；新法需对**稀疏缺失**（个别帧检测失败）做逐帧前向填充 |
| M3 | 批量预加载 + 索引构建（`build_flame_index` / `build_gaze_index` / 全窗口预载内存） | 升级为「按 clip 预载 3 个 NPZ + 按 frame_index/timestamp 建 dict 索引」 | 避免每帧 IO；结构与旧法同构 |
| M4 | 样本打包 + `manifest.csv`（sample_name/video_id/边界） | 保留为「每 clip 一个 sample 包 + 全量 manifest」 | 命名从 `{video}_{idx}` 改为 `{clip_id}`（见 §4.4） |

### 2.2 ❌ 必须删除 / 不复用的逻辑

| # | 旧逻辑 | 删除原因 | 新管线替代 |
|---|---|---|---|
| D1 | `expcode[:50]` 截断 | 旧用 EMOCA 50 维；新 EMICA 输出完整 100 维（Stage12 processor 已实测） | 直接用 100 维 |
| D2 | 逐帧 `codes.npz` / 逐帧 gaze CSV 文件布局与查找（`frame_name_candidates` 容错） | 新管线已统一为 per-clip NPZ | `read_stage12_outputs` / `read_gaze_features` / `read_audio_features` |
| D3 | FLAME QC 坏帧剔除（`quality_report.json` + mahalanobis/union/shape/pose_jump/expression） | 质量职责已由 Stage9/15/16 承接；Stage17 应保溯源而非再过滤 | Stage15 StatisticsInfo / Stage16 AnnotationInfo（可选并入 metadata） |
| D4 | wav2vec2 窗口级音频提取（Stage5.2 全部） | 已由 Stage14 HuBERT 逐帧 768 维取代 | `audio_features.npz [N,768]` |
| D5 | 固定 `FULL_WINDOW=200`（history 100 + future 100）语义 | 旧法为「预测任务」的样本构造；新管线 Stage4 clip=8s≈200 帧天然等长，且 history/future 切分属于训练 dataloader 职责 | clip 全窗口 |
| D6 | 音频 float16 | 新管线三模态统一 float32（训练友好） | float32 |
| D7 | `export_final_samples.py` 的拷贝式二次打包 | 新管线 Stage17 直接装配，无需再拷贝散落 .npy | Stage17 样本包生成器 |

---

## 3. Stage17 职责决策（三问）

### 3.1 ✅ 是否合并 visual feature？

**建议：合并为单张量 `visual_feature [T,112]`。**

- 与旧 Stage5.1 的「FLAME+Gaze 单张量」哲学一致，下游编码器可一次性消费视觉模态；
- 三子块来源互斥、维度互补：`expcode[100] + posecode[6] + gaze_vector[6]`，不存在信息重复；
- 与 doc02 的 106+6 分离方案相比，仅多一步 Stage17 内 concat；若后续需要分离（例如 gaze 单独 loss），
  可保留子块边界索引 `visual_layout = {"expression":(0,100), "pose":(100,106), "gaze":(106,112)}` 供拆分，不影响存储。

### 3.2 ✅ 是否保留 audio 独立？

**建议：保留独立 `audio_feature [T,768]`。**

- 与旧管线 audio 独立 .npy 一致；音频与视觉为两类特征空间（HuBERT 768 vs 参数 112），无理由硬拼；
- 音频无 `frame_index`，对齐方式（nearest timestamp）与视觉（frame_index join）不同，独立更利于溯源与校验；
- 下游模型可自由选择「visual-only / audio-only / concat」三种输入组织。

### 3.3 ✅ 是否输出 training sample package？

**建议：输出「每 clip 一个样本包 + 全量 manifest」**：

- 一个 Stage4 clip = 一个训练样本窗口（8s≈200 帧），`sample_id := clip_id` 即天然样本主键；
---

## 4. 最终数据格式设计

### 4.1 visual_feature [T,112] 评估（含 eye_pose 取舍）

**维度构成（采纳提案）：**

```
visual_feature[T,112] =
    expression[100]   ← FLAMEFeature.expcode            (Stage12 EMICA)
  + pose[6]           ← FLAMEFeature.posecode           = globalpose[3] + jawpose[3]
  + gaze[6]           ← GazeFeature.gaze_vector          (Stage13 MediaPipe, gaze_0/1 xyz)
```

**必须删除 `eye_pose[6]` 占位块，且不要把 gaze 写入 `FLAMEFeature.eye_pose`：**

- **语义不兼容**：`eye_pose` 是 FLAME 的眼球旋转参数（blendshape 系），`gaze_vector` 是 2×3D 视线单位方向向量，
  二者不同特征空间，直接替换会污染两个语义块。
- **信息重复**：若把 gaze「替换」进 eye_pose 又单独保留 gaze 列，则 gaze 在 visual 里出现两次。
- **零信息占位**：当前 `eye_pose` 恒为 `zeros(6)`（EMICA/SPECTRE 均不预测），拼入只会引入恒定零块，
  对训练无增益且可能污染归一化。
- **结论**：`eye_pose` 仅作为 Stage12 原始 NPZ 的审计字段保留（`spectre_jawpose` 同理），**不进入 visual_feature**。
  若下游确实需要 6D FLAME 眼球参数，应属于 Stage12 模型增强（另行立项），与 Stage17 无关。

> 与 `docs/02 §2.10.1`「eye_pose 由 gaze 替换」的旧约定冲突——**本报告建议修订该约定**（文档层修改，
> 不涉及任何 Python 文件）。

### 4.2 audio_feature [T,768]（保留独立）

- 直接取 `AudioFeature.feature`（HuBERT hidden 768，float32）；
- 对齐后 `T_audio_out == T`（与视觉帧数一致）；原始 `N_audio` 与匹配索引写入 metadata 供溯源。

### 4.3 metadata 字段定义

| 字段 | 类型 | 来源 | 说明 |
|---|---|---|---|
| sample_id | str | 生成 | `= clip_id`（每 clip 一个样本） |
| clip_id | str | Stage4 | 溯源主键 |
| video_id / scene_id | str | ClipInfo | 溯源 |
| fps | float | config `video.target_fps`=25 | 时间网格 |
| start_frame / end_frame | int | ClipInfo | 绝对帧号 |
| clip_start / clip_duration | float | AVAlignmentInfo / ClipInfo | 绝对时间窗口 |
| timestamps | [T] float | 生成 | = start_frame/fps + i/fps |
| frame_index | [T] int | Stage12/13 | 绝对帧号 |
| valid_length | int | 生成 | 三模态齐全的帧数（不含 dropout/audio 失配） |
| dropout_mask | [T] bool | 生成 | True=该帧缺失/前向填充（可选） |
| audio_match_idx | [T] int | 生成 | 每帧命中的音频特征索引（nearest 溯源） |
| source_audio | str | Stage14/7.1 | 校正音频路径 |
| T_flame / T_gaze / T_audio | int | 输入 | 原始长度，用于偏差审计 |
| audio_tail_policy | str | config | drop / last_repeat（边界 ±1 帧处理） |
| visual_layout | dict | 常量 | {"expression":(0,100),"pose":(100,106),"gaze":(106,112)} |

### 4.4 样本包结构与输出布局

```
outputs/stage17_samples/
├── manifest.csv                     # 数据集清单（每条 = 一个样本）
│    sample_id, clip_id, video_id, fps, T, valid_length, clip_start, source_audio
└── {sample_id}/                     # sample_id := clip_id
     ├── training_sample.npz         # 事实数据（推荐单文件，便于 dataloader 一次加载）
     │    visual_feature [T,112] float32
     │    audio_feature  [T,768] float32
     │    frame_index    [T]   int32
     │    timestamps     [T]   float32
     │    dropout_mask   [T]   bool      (可选)
     │    audio_match_idx[T]   int32
     │    valid_length   scalar int32
     │    fps            scalar float32
     │    clip_id        scalar str
     └── metadata.json               # §4.3 全字段（人类可读 + 审计）
```

### 4.5 备选方案对比

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| **A（推荐）** 单 `training_sample.npz`（visual+audio+meta 同包） | dataloader 一次加载、对齐关系内聚、与 Stage12/13/14 的 per-clip NPZ 模式一致 | 包较大（8s≈200×880 float32≈704KB，可压缩） | ✅ |
| B 分离 `visual_feature.npz` + `audio_feature.npz` | 镜像旧管线 layout，单模态替换不影响他模态 | 需二次对齐（重新读 timestamps 匹配），易错 | ⚠️ 不推荐 |
| C 直接复用 dataclass `TrainingSample` 列表 | 与 Stage 间标准 Data Object 通信（Rule1）一致 | 现有 `TrainingSample` 是单帧标量结构，无法表达 [T,D] | ❌ 需先重构 |

---

- 样本包内同时放**对齐后张量**与**溯源元数据**，满足 `docs/02 §2.12`「所有数据可追踪」原则；
- 同时生成数据集级 `manifest.csv`，承接旧 `export_final_samples.py` 的目录清单职能。

---

## 5. 时间同步策略

### 5.1 FLAME / Gaze：按 `frame_index` 精确对齐（推荐）

- 两者消费同一个 `List[AlignedFaceInfo]`（Stage11 输出），`frame_index` 即视频绝对帧号，**语义上逐帧一一对应**；
- **必须按 frame_index（而非行号/顺序位置）做 hash join**：Stage12/13 都会因图片缺失/无检测而 `continue` 跳过个别帧，
  两序列长度可能不同（既有报告 §2.3 已确认）；
- 锚点选择：以 **FLAME 的 frame_index 为主键**（Stage12 缺失概率更低），Gaze 缺失帧 → 前向填充 + `dropout_mask=True`；
- 校验：`np.all(flame_frame_index == gaze_frame_index)`（对存在的交叠帧）应全等，否则视为数据异常并告警。

### 5.2 Audio：按 `timestamp` nearest 匹配（推荐，沿用既有报告 §3.4）

- AudioFeature **无 frame_index 字段**，且 HuBERT 帧数在 clip 边界存在 ±1 帧近似 → 无法逐帧索引；
- 对每个目标视频帧时刻 `t_j = clip_start + j/fps`，取 `argmin_i |t_j - t_audio_i|`（nearest，不插值）；
  插值会制造虚假中间特征，talking-head 数据主流做法是 nearest 对齐；
- 预期误差：Audio 已重采样到 25FPS 网格 → 正常帧 `|Δt| ≤ 20ms`；clip 边界 ±1 帧 → `|Δt| ≤ 40ms`；
- **确定性尾帧策略**：超出容差的边界帧 → `audio_tail_policy`：`drop`（删该帧，缩短 T）或
  `last_repeat`（末帧重复，保持 T 固定），并在 metadata 记录。

### 5.3 风险与防御

| 风险 | 来源 | 防御 |
|---|---|---|
| 音频整体偏移 | `AVAlignmentInfo.clip_start=None` 时 Stage14 `stage.py:100` 静默回退 `start_time=0.0` → 音频变 clip 相对时间，与 FLAME/Gaze 绝对时间系统性错位 | Stage17 对齐前先校验 `max_j|t_j - t_audio[argmin]|` 的**整体偏差**：若中位数误差 > 100ms → 判定时间基准不一致，跳过该 clip 并记录（不静默对齐） |
| 音频帧数近似 | HuBERT conv stride floor（8s 音频 T_h 可能 399/400）→ N_audio = T_video ± 1 | nearest 天然容忍；`audio_match_idx` 全程记录 |
| Gaze 缺失 | 检测失败/图片缺失 | forward-fill + `dropout_mask`；`valid_length` 反映真实有效帧数 |
| FLAME 缺失 | 图片缺失 | 同上；若 FLAME 缺失导致连续大段空洞 → 该 clip 建议整段标记低质 |

### 5.4 缺失/无效帧处理策略汇总

```
策略（按优先级）：
1. frame_index 精确 join（FLAME 锚点）
2. Gaze 缺失 → forward-fill（沿用旧 M2 末帧重复思想的逐帧版） + dropout_mask=True
3. Audio 每帧 nearest 匹配；|Δt|≤20ms 通过，≤40ms（边界）按 tail_policy，>40ms 标记无效
4. valid_length = 三模态齐全且 |Δt| 合格的帧数
5. 时间基准整体校验失败 → 跳过 clip（不进 manifest）
```

---

## 6. 当前数据接口支持度检查（task 6）

### 6.1 `face_pipeline_v2/data/models.py`

| 模型 | 字段 | Stage17 需要 | 状态 |
|---|---|---|---|
| FLAMEFeature | `frame_index`/`timestamp`/`clip_id`/`expcode[100]`/`posecode`(property=global[3]+jaw[3])/`eye_pose[6](zeros)` | exp+pose+gaze 源 | ✅ 支持（eye_pose 不进融合，§4.1） |
| GazeFeature | `frame_index`/`timestamp`/`clip_id`/`gaze_vector[6]` | gaze 源 | ✅ 支持 |
| AudioFeature | `timestamp`/`feature[768]`/`dimension`/`source_audio` | audio 源 | ✅ 支持（无 frame_index，nearest 对齐） |
| **TrainingSample** | `sample_id`/`video_clip`/`face_sequence`/`flame_feature`/`gaze_feature`/`audio_feature`/`metadata` | 输出模型 | ❌ **单帧标量结构，无法表达 [T,D]** → 需重构（见 §7） |

> `TrainingSample` 现仅被 `tests/test_datamodel.py` 引用（该测试另有 F1 的 GazeFeature 4 参构造缺陷，预存在）。
> 重构 TrainingSample 的影响面很小。

### 6.2 Stage12 输出（`flame_features.npz`）

- ✅ `read_stage12_outputs(output_dir, clip_id)` 可完整恢复 FLAMEFeature（含 timestamp/clip_id 回退）；
- ❌ NPZ **未写 `clip_id` 数组**（F2，既有报告 §5）→ Stage17 必须显式传 `clip_id` 参数读入，**不要裸读 NPZ 断言字段**；
  （如需根治，属 Stage12 一行修改 + 回归测试，与 Stage17 实现解耦。）

### 6.3 Stage13 输出（`gaze_features.npz`）

- ✅ 字段最完整：`frame_index`/`timestamp`/`clip_id`/`gaze_vector[T,6]` 齐备；`read_gaze_features` 往返已验证；
- ✅ 本次已完成 host 环境 mediapipe 0.10.8 安装与 50 帧真实数据验证（`docs/Stage13_Runtime_Environment_Report.md`）——F3 已解除。

### 6.4 Stage14 输出（`audio_features.npz`）

- ✅ `read_audio_features` 恢复 AudioFeature（timestamp/feature/dimension/source_audio）；
- ❌ metadata 缺 `clip_start`/`fps`/`N_audio`（F5）→ Stage17 需从 `AVAlignmentInfo.clip_start`（或 config fps）
  计算目标时间网格；`source_audio` 已记录可溯源。可选：后续给 Stage14 metadata 补 `clip_start/fps`（非阻断）。

### 6.5 配置与编排

- ❌ `configs/config.yaml` 无 `stage_17` 段 → 需新增（§7.3 给键位清单）；
- ❌ `pipeline/pipeline_runner.py` 为 0 字节 → 无端到端宿主；Stage17 应仿照 stage12/13/14 提供独立 `runner.py`
  （按 clip 装配），后续再决定是否补齐全局编排器（F4，与 Stage17 解耦）。

---

## 7. 需要修改点汇总（只列设计方案，不实现）

### 7.1 必改（Stage17 落地前提）

| # | 位置 | 修改内容 | 类型 |
|---|---|---|---|
| R1 | `face_pipeline_v2/data/models.py` | 将 `TrainingSample` 从「单帧标量」重构为「时间序列样本包」：`visual_feature[T,112]` / `audio_feature[T,768]` / `timestamps[T]` / `frame_index[T]` / `valid_length` / `dropout_mask` / `audio_match_idx` / `metadata`（或新增 `TrainingSamplePackage` 并保留旧类兼容） | 数据模型（Python 修改，本次未做） |
| R2 | `face_pipeline_v2/stages/stage_17_fusion/` | 新增 `processor.py`（三模态 merge：frame_index join + nearest audio + 缺失填充）、`integration.py`（样本包 NPZ/JSON/manifest 读写）、`stage.py`（BaseStage 接口 + config 驱动）、`runner.py`（独立 CLI，仿 stage12/13/14） | Stage17 实现（本次未做） |
| R3 | `configs/config.yaml` | 新增 `stage_17` 段：`enabled` / `fps:25` / `output_dir:outputs/stage17_samples` / `visual_dim:112` / `audio_dim:768` / `audio_nearest_tolerance_ms:20` / `boundary_tolerance_ms:40` / `audio_tail_policy: last_repeat|drop` / `dropout_policy: forward_fill` / `package_format: npz` / `write_manifest:true` / `overall_offset_check_ms:100` | 配置（本次未做） |
| R4 | 数据源接入 | Stage17 以「按 clip 读 3 个 NPZ（经各自 read_* 函数）+ AVAlignmentInfo.clip_start」为唯一事实数据源；Stage12 必须传 `clip_id` 参数（F2 未修期间） | 实现约定（本次未做） |

### 7.2 建议改（低风险、提升健壮性）

| # | 位置 | 内容 |
|---|---|---|
| S1 | `docs/02 §2.10.1` | 修订「eye_pose 由 gaze 替换」表述 → 「eye_pose 保持 zeros 审计字段，gaze 独立进入 visual_feature」（文档层，消除歧义） |
| S2 | `stage_14_audio_feature/integration.py` metadata | 补记 `clip_start` / `fps` / `N_audio`（缓解 F5；非阻断） |
| S3 | `stage_12_flame/runner.py` | `flame_features.npz` 补写 `clip_id` 数组（修 F2，一行 + 回归测试；与 Stage17 解耦，可后置） |
| S4 | `tests/test_datamodel.py` | 修 F1 的 `GazeFeature` 4 参构造（预存在缺陷，阻塞全量 CI，建议顺手修） |

### 7.3 明确不改 / 保持

- **Stage12/13/14 的架构与输出格式**（本次审查不动）；`eye_pose`/`cam`/`shapecode` zeros 作为审计字段保留。
- **不把 Stage17 并入 emica_env / 不改 Docker**：Stage17 为纯 NumPy 装配（无模型推理），host Python 即可运行。
- **不做逐帧插值对齐**（nearest 即可，插值会引入虚假特征）。
- **不在 Stage17 做质量过滤**（质量职责在 Stage9/15/16）；Stage17 只保留 `valid_length`/`dropout_mask` 供训练侧丢弃。

---

## 8. 附录：审查依据（只读访问）

- `face_pipeline_v2/data/models.py`（TrainingSample / FLAMEFeature / GazeFeature / AudioFeature / AlignedFaceInfo / ClipInfo / AVAlignmentInfo）
- `face_pipeline_v2/stages/stage_12_flame/{integration,processor,stage}.py`、`stage_13_gaze/{integration,stage}.py`、`stage_14_audio_feature/{integration,processor,stage}.py`
- `face_pipeline_v2/stages/stage_17_fusion/__init__.py`（空）、`face_pipeline_v2/pipeline/pipeline_runner.py`（0 字节）
- `configs/config.yaml`（无 stage_16/17 段）、`docs/02_Data_Model_and_Interface.md`（§2.9.1/§2.10）、`docs/03_Pipeline_Stage_Specification.md`（§3.23）、`docs/Stage12_14_DataFlow_Compatibility_Report.md`
- 旧管线：`/mnt/home/lxs/complete_single_face_package_20260624/scripts/handle_stage5_1_flame.py`、`handle_stage5_2_audio.py`、`export_final_samples.py`
- 环境：Stage13 host mediapipe 0.10.8 已装并 50 帧验证通过（`docs/Stage13_Runtime_Environment_Report.md`）

---

## End of Stage17_Design_Review.md v1.0

