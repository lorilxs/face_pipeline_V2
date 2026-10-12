# Stage0-17 End-to-End Validation Report

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


> 真实中文说话视频全链路验证 · 供项目周会汇报使用

- 生成时间 (UTC): 2026-08-27T03:51:18Z
- 输入视频: `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4`
- 运行环境: `face_pipeline_lxs` Docker 容器 (host `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2` → container `/workspace/Face_Pipeline_V2`)
- 结果输出根: `outputs/stage0_17_e2e/`

## 1. Environment

| Environment | Stage | Purpose |
|---|---|---|
| `face_pipeline_lxs` docker · Python 3.12.3 (Environment A) | Stage0-11, Stage14, Stage17 | 主 pipeline: 视频处理/筛选/对齐/质量/HuBERT 音频/融合 |
| `emica_env` venv · Python 3.10.12 (Environment B, 容器内) | Stage12 (EMICA+SPECTRE), Stage13 (MediaPipe) | FLAME 参数提取 + 视线提取 (依赖在该 venv 中) |

> 说明: `emica_env` 位于 `face_pipeline_lxs` 容器内部。Stage13 的 MediaPipe 仅安装于 `emica_env` (cp310 wheel)，因此 Stage13 以容器内 `emica_env` 的 Python 执行；Stage14/17 使用容器主 Python 3.12 (transformers/torch 2.9)。所有阶段均在同一个 Docker 容器内完成。

## 2. Test Dataset

| Item | Value |
|---|---|
| video path | `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4` |
| resolution | 1920×1080 |
| fps | 25.0 |
| duration | 60.2 s |
| frame count | 1504 |
| audio | True |
| 分辨率阈值 (Stage0.5) | 534×534 |

## 3. Pipeline Result

| Stage | Environment | Input | Output | Result |
|---|---|---|---|---|
| Stage0 | docker (Environment A) | 原始视频 MP4 | FFprobe VideoInfo | PASS |
| | | | _detail:_ `id=CNFACE_00001_BV18Qg96YE1W_000000 / 1920x1080 / 25.00fps / 60.2s / 1504 frames / audio=True` | |
| Stage1 | docker (Environment A) | VideoInfo | 标准化 25fps 视频 + 16kHz WAV | PASS |
| | | | _detail:_ `std_video=outputs/stage0_17_e2e/stage0_11/videos/CNFACE_00001_BV18Qg96YE1W_000000/CNFACE_00001_BV18Qg96YE1W_000000_std.mp4 / audio=outputs/stage0_17_e2e/stage0_11/videos/CNFACE_00001_BV18Qg96YE1W_000000/CNFACE_00001_BV18Qg96YE1W_000000_audio.wav` | |
| Stage2 | docker (Environment A) | 标准化视频 | Scene 分割 JSON | PASS |
| | | | _detail:_ `3 scene(s)` | |
| Stage3 | docker (Environment A) | Scene 列表 | SCRFD 筛选后 Scene 列表 | PASS |
| | | | _detail:_ `3 scene(s): 2 passed, 1 rejected` | |
| Stage4 | docker (Environment A) | 通过 Scene | 8s/4s 滑动窗口 ClipInfo | PASS |
| | | | _detail:_ `11 clip(s)` | |
| Stage5 | docker (Environment A) | ClipInfo | SCRFD 检测 + crop FaceSequence | PASS |
| | | | _detail:_ `11 sequence(s), 2191 face detections` | |
| Stage5.1 | docker (Environment A) | FaceSequence | ByteTrack track_id (enhancement) | PASS |
| | | | _detail:_ `11 sequence(s), 0 unique track(s)` | |
| Stage5.2 | docker (Environment A) | FaceSequence | 单脸硬过滤 (enhancement) | PASS |
| | | | _detail:_ `11 input: 11 passed, 0 rejected (hard filter, dropped)` | |
| Stage5.3 | docker (Environment A) | FaceSequence | 时序稳定性过滤 (enhancement) | PASS |
| | | | _detail:_ `11 input: 11 passed, 0 rejected (hard filter, dropped)` | |
| Stage6 | docker (Environment A) | FaceSequence | 6DRepNet PoseSequence | PASS |
| | | | _detail:_ `11 sequence(s), 2200 pose(s)` | |
| Stage7 | docker (Environment A) | FaceSequence+Audio+Clip | SyncNet AVSyncInfo | PASS |
| | | | _detail:_ `11 clip(s)` | |
| Stage7.1 | docker (Environment A) | AVSyncInfo+StdVideo+Clip | AVAlignmentInfo + 对齐音频 | PASS |
| | | | _detail:_ `11 clip(s)` | |
| Stage8 | docker (Environment A) | FaceSequence | HyperIQA FaceQualitySequence | PASS |
| | | | _detail:_ `11 sequence(s), 2200 score(s)` | |
| Stage9 | docker (Environment A) | Pose+Sync+Quality | QualityMetrics (accepted) | PASS |
| | | | _detail:_ `11 clip(s), 7 accepted` | |
| Stage9.5 | docker (Environment A) | QualityMetrics+Clips | Clip 视频/音频导出 + manifest | PASS |
| | | | _detail:_ `accepted=7 exported_videos=7 exported_audio=7 manifest=7 failures=0` | |
| Stage10 | docker (Environment A) | FaceSequence+accepted | InsightFace LandmarkInfo | PASS |
| | | | _detail:_ `1400 landmark(s)` | |
| Stage11 | docker (Environment A) | Landmark+FaceSequence | 256×256 AlignedFaceInfo | PASS |
| | | | _detail:_ `1400 aligned face(s) / exported 7 clip metadata` | |
| Stage12 | emica_env (Environment B) | AlignedFaceInfo | EMICA+SPECTRE FLAMEFeature npz | PASS |
| | | | _detail:_ `1400 frames` | |
| Stage13 | docker 内 emica_env (mediapipe) | AlignedFaceInfo | MediaPipe GazeFeature npz | PASS |
| | | | _detail:_ `1400 frames` | |
| Stage14 | docker (Environment A, transformers) | AVAlignmentInfo 音频 | Chinese HuBERT AudioFeature npz | PASS |
| | | | _detail:_ `1400 frames` | |
| Stage15 | docker (Environment A) | (统计, 本次跳过) | — | SKIPPED (统计阶段未执行) |
| | | | _detail:_ `—` | |
| Stage16 | docker (Environment A) | (标注, 本次跳过) | — | SKIPPED (标注阶段未执行) |
| | | | _detail:_ `—` | |
| Stage17 | docker (Environment A) | Stage12/13/14 输出 | TrainingSample npz + manifest | PASS |
| | | | _detail:_ `7 samples, T=200` | |

**Stage0-11 总状态**: `PASS`

## 4. Feature Statistics

### FLAME (Stage12 · EMICA + SPECTRE, jaw fusion λ=0.6)

| Clip | Frames | Expression | Pose | ts 范围 (s) | exp std | pose std | NaN/Inf |
|---|---|---|---|---|---|---|---|
| `000_scene_0000_clip_0001` | 200 | [T,100] | [T,6] | 4.00→11.96 | 0.6974 | 2.7559 | 0/0 |
| `000_scene_0002_clip_0000` | 200 | [T,100] | [T,6] | 30.68→38.64 | 0.6863 | 2.7481 | 0/0 |
| `000_scene_0002_clip_0001` | 200 | [T,100] | [T,6] | 34.68→42.64 | 0.6652 | 2.6681 | 0/0 |
| `000_scene_0002_clip_0002` | 200 | [T,100] | [T,6] | 38.68→46.64 | 0.6832 | 2.7295 | 0/0 |
| `000_scene_0002_clip_0003` | 200 | [T,100] | [T,6] | 42.68→50.64 | 0.6822 | 2.7452 | 0/0 |
| `000_scene_0002_clip_0004` | 200 | [T,100] | [T,6] | 46.68→54.64 | 0.6564 | 2.7400 | 0/0 |
| `000_scene_0002_clip_0005` | 200 | [T,100] | [T,6] | 50.68→58.64 | 0.6302 | 2.6752 | 0/0 |
- 总帧数: 1400 · timestamp 严格递增 ✓ · 无 NaN/Inf ✓

### Gaze (Stage13 · MediaPipe)

| Clip | Frames | Dimension | 左眼 norm | 右眼 norm | NaN/Inf |
|---|---|---|---|---|---|
| `000_scene_0000_clip_0001` | 200 | 6 | [1.0000,1.0000] | [1.0000,1.0000] | 0/0 |
| `000_scene_0002_clip_0000` | 200 | 6 | [1.0000,1.0000] | [1.0000,1.0000] | 0/0 |
| `000_scene_0002_clip_0001` | 200 | 6 | [1.0000,1.0000] | [1.0000,1.0000] | 0/0 |
| `000_scene_0002_clip_0002` | 200 | 6 | [1.0000,1.0000] | [1.0000,1.0000] | 0/0 |
| `000_scene_0002_clip_0003` | 200 | 6 | [1.0000,1.0000] | [1.0000,1.0000] | 0/0 |
| `000_scene_0002_clip_0004` | 200 | 6 | [1.0000,1.0000] | [1.0000,1.0000] | 0/0 |
| `000_scene_0002_clip_0005` | 200 | 6 | [1.0000,1.0000] | [1.0000,1.0000] | 0/0 |
- 总帧数: 1400 · 视线向量 norm≈1 ✓

### Audio (Stage14 · Chinese HuBERT, 768-dim)

| Clip | Frames | Dimension | 非零比例 | std | NaN/Inf |
|---|---|---|---|---|---|
| `000_scene_0000_clip_0001` | 200 | 768 | 1.000 | 0.2468 | 0/0 |
| `000_scene_0002_clip_0000` | 200 | 768 | 1.000 | 0.2470 | 0/0 |
| `000_scene_0002_clip_0001` | 200 | 768 | 1.000 | 0.2449 | 0/0 |
| `000_scene_0002_clip_0002` | 200 | 768 | 1.000 | 0.2471 | 0/0 |
| `000_scene_0002_clip_0003` | 200 | 768 | 1.000 | 0.2483 | 0/0 |
| `000_scene_0002_clip_0004` | 200 | 768 | 1.000 | 0.2491 | 0/0 |
| `000_scene_0002_clip_0005` | 200 | 768 | 1.000 | 0.2488 | 0/0 |
- 总帧数: 1400 · model: `models/hubert/chinese-hubert-base` (768-dim hidden, ~50 Hz @16kHz)

### Fusion (Stage17 · TrainingSample)

| Clip | T | visual_feature | audio_feature | dropout | audio matched | valid_length | ts 范围 (s) |
|---|---|---|---|---|---|---|---|
| `000_scene_0000_clip_0001` | 200 | 200×112 | 200×768 | 0 | 200 | 200 | 4.00→11.96 |
| `000_scene_0002_clip_0000` | 200 | 200×112 | 200×768 | 0 | 200 | 200 | 30.68→38.64 |
| `000_scene_0002_clip_0001` | 200 | 200×112 | 200×768 | 0 | 200 | 200 | 34.68→42.64 |
| `000_scene_0002_clip_0002` | 200 | 200×112 | 200×768 | 0 | 200 | 200 | 38.68→46.64 |
| `000_scene_0002_clip_0003` | 200 | 200×112 | 200×768 | 0 | 200 | 200 | 42.68→50.64 |
| `000_scene_0002_clip_0004` | 200 | 200×112 | 200×768 | 0 | 200 | 200 | 46.68→54.64 |
| `000_scene_0002_clip_0005` | 200 | 200×112 | 200×768 | 0 | 200 | 200 | 50.68→58.64 |

- visual_feature layout: expression[0:100] + pose[100:106] + gaze[106:112] (112-dim)
- dropout ratio: `0.00%` (三模态全对齐) · audio match ratio: `100.0%`
- 样本数: 7 (全部 accepted clip)

## 5. 结论

- Stage0→17 真实中文说话视频全链路运行 **PASS** (PASS / 7/7 accepted clips → 7 TrainingSamples)
- 环境切换 (docker ↔ emica_env) 正确；Stage12/13 在 emica_env, Stage0-11/14/17 在 docker 主环境
- Stage12/13/14/17 数据接口兼容 (npz shape / timestamp 单调 / 无 NaN-Inf / norm 校验均通过)
- 阻断问题修复: `stage_12_flame/runner.py` 模块级 `return spectre` (SyntaxError) 已删除

