# Stage12_17 全链路回归报告 (真实视频)

> **⚠️ 历史记录** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh
> 重建验证）。本报告验证的是此前 AISDD 批准的 SPECTRE-jaw 方案，保留作为历史记录；
> 最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。

- **日期**: 2026-08-29
- **测试视频**: `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4` (1467 帧, 25 fps)
- **评测 clip**: `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001` (帧 100–299, 共 200 帧, 时间戳 4.000–11.960 s, 25 fps)
- **目标**: 验证 AISDD 批准的 Stage12 姿态源变更 (`jaw_pose = SPECTRE` 纯取, `head_pose`/`expression` 保持 EMICA) 在**真实视频完整链路 Stage0→Stage17** 上全流程 PASS。
- **范围**: 只读验证 — **未修改任何 pipeline 代码** (唯一新增/删除为本次临时校验脚本, 已清理)。

---

## 1. 完整链路执行概况 (全部 PASS)

| 阶段 | 环境 | 输入/范围 | 结果 | 说明 |
|---|---|---|---|---|
| Stage 0–11 | docker python3.12 | 全视频 (7 clip, 1400 张对齐人脸) | **PASS** (共 170.4 s) | 全部 18 个子阶段 PASS; 重新导出 `_aligned_faces.json` / `_av_alignment.json` |
| Stage 12 (FLAME) | emica_env | clip_0001 全 200 帧 | **PASS** | Scheme D: expression [200,100] std=0.449, pose std=0.0730 (弧度) |
| Stage 13 (gaze) | emica_env | clip_0001 全 200 帧 | **PASS** | Extracted 200/200 |
| Stage 14 (audio) | docker python3.12 | clip_0001 全 200 帧 (HuBERT) | **PASS** | Extracted 200 audio features (device=cuda) |
| Stage 17 (fusion) | docker python3.12 | Stage12/13/14 输出装配 | **PASS** | `RESULT: PASS`, round-trip 一致, valid_length=200 |

**Stage0-11 各子阶段耗时**: Stage0=1.4s · 0.5=0.0s · 1=7.0s · 2=4.8s · 3=9.4s · 4=0.0s · 5=24.7s · 5.1/5.2/5.3=0.0s(禁用) · 6=24.1s · 7=30.4s · 7.1=1.2s · 8=53.7s · 9=0.0s · 9.5=5.7s · 10=4.3s · 11=3.6s (本次重跑日志同量级, Stage3≈9.8s / Stage5≈65s / Stage8≈84.5s / Stage10≈11.7s / Stage11≈3.7s)。

---

## 2. 报告字段表 (PASS/FAIL)

| 字段 | 值 | 校验 | 判定 |
|---|---|---|---|
| `stage12_expression_shape` | `(200, 100)` | expression 100 维, std=0.449 (无异常离散) | ✅ PASS |
| `stage12_head_pose_deg` | `[-7.84°, +12.87°]` (mean_abs 4.92°, std 5.77°) | 正常头部运动范围, 最大单轴 12.87° — 无异常大姿态 | ✅ PASS |
| `stage12_jaw_pose_deg` | `[-2.30°, +4.18°]` (mean_abs 0.93°, std 1.12°) | 正常下颌小幅运动, 最大单轴 4.18° — 无异常大姿态 | ✅ PASS |
| `stage12_jaw_source_check` | `True` | `jaw_pose == spectre_jawpose` **max_abs_diff=0.0, exact_equal=True**; 与 EMICA 原始 jaw 最大差 **13.33°** (Scheme D 实质生效) | ✅ PASS |
| `stage13_gaze_norm` | `left≈1.0000, right≈1.0000` (200/200 有效) | 每眼 3 维单位向量, norm≈1 | ✅ PASS |
| `stage14_audio_shape` | `(200, 768)` | Chinese HuBERT, std=0.2468, nonzero_ratio=1.0 | ✅ PASS |
| `stage17_visual_shape` | `(200, 112)` | = expression[100]+pose[6]+gaze[6] | ✅ PASS |
| `stage17_audio_shape` | `(200, 768)` | 与 Stage14 对齐 | ✅ PASS |
| `stage17_alignment` | `T_visual=200 == T_audio=200; matched=200, dropout=0, valid_length=200` | 时间戳 dt=0.04s 严格单调; audio_match=nearest_timestamp | ✅ PASS |

**最终结论: PASS (9/9)**

---

## 3. 关键校验明细

### 3.1 FLAME 姿态 (Stage12, Scheme D)
- `flame_metadata.json` 明确记录: `jaw_source=spectre` · `head_source=emica` · `expression_source=emica` · `backend=EMICA+SPECTRE` · `jaw_fusion_lambda=1.0`。
- 帧索引 100–299 共 200 帧; 时间戳 4.000→11.960 s, `dt_mean=dt_min=dt_max=0.0400` (严格 25 fps 单调递增)。
- **jaw 保持 SPECTRE 纯取**: `jaw_pose` 与 `spectre_jawpose` 逐帧恒等 (max diff 0.0), 与 EMICA 原始 jaw 差异最大 13.33° — 证实 Scheme D 消除融合并改用 SPECTRE 后确实改变了 jaw 运动学。
- 无异常大 FLAME 姿态: head 最大 |值|=12.87°, jaw 最大 |值|=4.18° (均远低于异常阈值)。

### 3.2 NaN/Inf 扫描 (全链路)
| 文件 | NaN | Inf |
|---|---|---|
| `stage12_flame/<clip>/flame_features.npz` (expression/jaw/head/eye/spectre_jaw/emica_jaw/shapecode/timestamp) | 0 | 0 |
| `stage13_gaze/<clip>/gaze_features.npz` (gaze_vector) | 0 | 0 |
| `stage14_audio/<clip>/audio_features.npz` | 0 | 0 |
| `stage17_fusion/<clip>/training_sample.npz` (visual/audio/对齐数组) | 0 | 0 |

### 3.3 Stage17 输出格式 (`flame` / `audio` / `timestamp` 布局)
`outputs/stage17_fusion/<clip>/training_sample.npz`:
- `visual_feature [200,112] float32` (flame 布局: expression[0:100]+pose[100:106]+gaze[106:112])
- `audio_feature  [200,768] float32` (Chinese HuBERT)
- `timestamps [200] float32`, `frame_index [200] int32`, `audio_match_idx [200] int32`, `dropout_mask [200] bool`, `valid_length=200`
- `metadata.json`: visual_dim=112, audio_dim=768, num_frames=200, valid_length=200, fps=25, clip_start=4.0, duration=8.0, `n_input={flame:200, gaze:200, audio:200}`。
- `read_training_sample(save_training_sample(s))` **round-trip 逐数组/字段一致** — PASS。
- 对齐统计: FLAME=200 / Gaze=200 / Audio=200, matched=200, dropout=0 (0.00%), valid_length=200。

---

## 4. 环境与执行备注

- **Stage 0–11 / 14 / 17** → docker `face_pipeline_lxs` python3.12; **Stage 12 / 13** → `/workspace/emica_env` (python3.10)。
- Stage13/14 的 `runner.py` 直接按路径执行时 repo 根不在 `sys.path`, 需 `PYTHONPATH=/workspace/Face_Pipeline_V2` (既有 runner sys.path 假设, 超出本次变更范围, 未修改)。
- Stage14 输入使用 **`_av_alignment_stage14.json`** (指向持久化导出 wav `outputs/stage0_17_e2e/stage0_11/audio/<clip>.wav`): Stage9.5 的 `cleanup_after_export` 会删除 stage7.1 中间 aligned wav, 默认 `_av_alignment.json` 指向的文件在重跑后不存在 — 此为既有 pipeline 行为, 未修改。
- **manifest.csv 遗留行**: Stage17 追加式写 manifest, 现含上一轮 50 帧遗留行 + 本轮 200 帧行; 当前 `training_sample.npz` 为 200 帧 (与 metadata.json / 本次校验一致)。
- Stage12/13 在 docker 内 CPU 回退 ~3–4 分钟/200 帧; Stage14 使用 cuda (HuBERT)。

---

## 5. 结论

Stage12 (`jaw=SPECTRE` 纯取 / `head`/`expression`=EMICA) 的生产变更在真实视频 `CNFACE_00001_..._000000.mp4` 的 **Stage0→Stage17 完整链路**上回归通过: 所有阶段 PASS, 9/9 报告字段 PASS, 全链路无 NaN/Inf, 无异常 FLAME 姿态, jaw 与 SPECTRE 逐帧恒等, Stage17 样本包结构/维度/对齐均符合设计契约。
