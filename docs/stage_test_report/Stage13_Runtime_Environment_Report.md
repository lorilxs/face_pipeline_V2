# Stage13_Runtime_Environment_Report.md

# Stage13 Gaze 运行环境设计确认与验证报告

Version: v1.0 | Date: 2026-08-27 | 验证方式: host Python 实测 50 帧真实数据

---

## 1. 结论摘要

- **Stage13 调用链确认**：Stage11（Alignment）→ Stage13（Gaze）为**进程内直接调用**（`List[AlignedFaceInfo]` 直传），
  与 Stage12（Docker 子进程）不同，**无需 Docker / 无需迁移到 emica_env**。
- **Host Python（`/opt/hb/bin/python3`）缺少 mediapipe**（`ModuleNotFoundError`）→ 已**仅在 host 环境**安装
  `mediapipe 0.10.8`（含所需依赖 `protobuf 4.25.9`、`absl-py 2.5.0`）。
- **50 帧真实数据验证全部通过**：`GazeFeature` 共 50 条、`gaze_vector` 形状 **[T=50, 6]**、
  **timestamp 正确**（= `frame_index / 25.0`，与 Stage5 绝对时间线一致）、**无 NaN / Inf**、左右眼向量模长 = 1.0。
- **未修改 Stage13 架构，未修改任何 Docker 配置**；Stage13 源文件零改动。
- 项目既有回归测试 `tests/test_stage13_gaze_real_data.py` 由「skip（mediapipe 缺失）」变为 **PASS**。

---

## 2. Stage13 调用链确认（Stage11 → Stage13）

### 2.1 调用方式

| 环节 | 机制 | 说明 |
|---|---|---|
| Stage11 → Stage13 | **进程内直接调用** | `Stage13Gaze.process(input_data, context)`，入参 `List[AlignedFaceInfo]`（`face_pipeline_v2/stages/stage_13_gaze/stage.py:60`） |
| 推理后端 | **host Python 内 MediaPipe** | `processor.py::create_mediapipe_detector()` → `mediapipe.tasks.python.vision.FaceLandmarker` |
| 模型资产 | `models/mediapipe/face_landmarker.task`（3.7 MB，已存在） | 配置 `configs/config.yaml → stage13.gaze.model_path` |
| 数据持久化 | `integration.py::save_gaze_features()` → `{output_dir}/{clip_id}/gaze_features.npz` | 供 Stage17 读取 |

### 2.2 与 Stage12 的运行隔离差异

- Stage12（FLAME / EMICA）依赖冲突严重，走 **Docker**（`face_pipeline_lxs:/workspace/emica_env`）。
- Stage13（MediaPipe FaceLandmarker）依赖轻量，项目原设计即为 **host Python 运行** —— 本次确认并保持该设计，
  **不将 Stage13 迁移到 emica_env**，不修改 Docker 配置。

### 2.3 runner.py 子进程模式（未改动，仅说明）

`runner.py` 提供独立子进程 CLI（`--input_metadata` + `--output_dir`），底层调用同一套
`create_mediapipe_detector / extract_gaze_parameters / save_gaze_features`，逻辑与 `stage.py` 完全一致，
本次验证直接复用这些公共函数，**未改动任何 Stage13 源码**。

---

## 3. Host Python 环境检查（/opt/hb/bin/python3）

### 3.1 安装前状态

```
Python 3.10.12  (pip 24.3.1)
  mediapipe      : 未安装（ModuleNotFoundError: No module named 'mediapipe'）   ← 缺失，本次修复
  protobuf       : 7.35.1          ← 版本过新（mediapipe 0.10.8 要求 <4）
  absl-py        : 未安装          ← 缺失，本次修复
  numpy          : 2.1.3   ✓
  flatbuffers    : 25.12.19 ✓   attrs: 24.2.0 ✓
  matplotlib     : 3.9.2   ✓
  opencv         : opencv-python-headless 5.0.0.93 ✓（cv2 5.0.0，供 Stage0-11 使用）
  onnxruntime    : 1.23.2   ✓   onnx: 1.22.0 ✓
  sounddevice    : 未安装（Gaze 人脸任务无需音频 I/O，mediapipe 导入不受影响）
```

### 3.2 安装后状态

```
mediapipe  : 0.10.8   （cp310 manylinux wheel，项目内已预置）
protobuf   : 4.25.9   （由 7.35.1 降级；项目内已预置 wheel）
absl-py    : 2.5.0    （项目内已预置 wheel）
onnx 1.22.0 / onnxruntime 1.23.2 依旧正常导入运行  ✓（protobuf 4.25.9 同时满足二者）
```

---

## 4. 安装步骤（只在 host 环境）

### 4.1 依赖冲突分析与解决

| 包 | protobuf 约束 | 结论 |
|---|---|---|
| mediapipe 0.10.8 | `protobuf<4,>=3.11`（wheel METADATA） | 需要 protobuf 3.x / 4.25 实测兼容 |
| onnx 1.22.0 | `protobuf>=4.25.1` | 需要 protobuf ≥4.25.1 |
| onnxruntime 1.23.2 | `protobuf`（无上界） | 任意兼容版本 |

- **交集**：`protobuf 4.25.9` 同时满足 onnx（≥4.25.1）且经实测与 mediapipe 0.10.8 运行兼容
  （mediapipe 元数据保守标注 `<4`，其运行时对 4.25.x 兼容）。
- 若直接 `pip install mediapipe`（自动解析），pip 会因 mediapipe 的 `<4` 与 onnx 的 `>=4.25.1`
  判定冲突并可能把 protobuf 降到 3.20.x → **破坏 onnx**。故采用：
  `pip install --no-deps mediapipe-0.10.8.whl`（依赖已手工齐备）。

### 4.2 实际执行命令

```bash
# 1) 降级 protobuf（满足 onnx>=4.25.1），并安装 absl-py（mediapipe 依赖）
/opt/hb/bin/python3 -m pip install --no-input \
    protobuf-4.25.9-cp37-abi3-manylinux2014_x86_64.whl \
    absl_py-2.5.0-py3-none-any.whl

# 2) 安装 mediapipe（--no-deps：避免强制安装 opencv-contrib-python 而降级 cv2 5.0.0）
/opt/hb/bin/python3 -m pip install --no-input --no-deps \
    mediapipe-0.10.8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 3) 验证导入
/opt/hb/bin/python3 -c "import mediapipe; print(mediapipe.__version__)"   # 0.10.8
```

### 4.3 刻意未安装的项（避免破坏 Stage0-11）

- **opencv-contrib-python**：mediapipe 元数据依赖项，但 host 已有 `opencv-python(-headless) 5.0.0.93`。
  安装 contrib 会把 cv2 降到 4.x，影响 Stage0-11。实测 `mediapipe.tasks.python.vision` 在现有 cv2 5.0.0 下正常，
  **Gaze 任务无需安装**。
- **sounddevice**：mediapipe 元数据依赖项，仅用于音频 I/O；Gaze（人脸图像）路径不导入，实测无需安装。

---

## 5. 验证结果（50 帧真实数据）

### 5.1 输入

- 数据源：`outputs/stage0_11_test_verify/aligned_faces/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001/`
  （Stage11 输出，共 200 帧对齐人脸，本次取**前 50 帧连续帧**：frame 100..149）
- 时间基准：`manifest.json → fps=25`，`timestamp = frame_index / 25.0`（与 Stage5 绝对标准时间线一致）
- 模型：`models/mediapipe/face_landmarker.task`，delegate=cpu（实测自动回退 XNNPACK CPU 加速）
- 脚本：`scripts/stage13_verify.py`（**仅调用 Stage13 既有公共函数**：`create_mediapipe_detector` /
  `extract_gaze_parameters` / `save_gaze_features` / `read_gaze_features`）

### 5.2 检查项结果

| # | 检查项 | 结果 | 实测值 |
|---|---|---|---|
| 1 | 提取数量 == 50 | **PASS** | 50 / 50 |
| 2 | gaze_vector 形状 [T,6] | **PASS** | [50, 6] |
| 3 | 无 NaN | **PASS** | nan count = 0 |
| 4 | 无 Inf | **PASS** | inf count = 0 |
| 5 | 左右眼向量模长 ≈ 1.0 | **PASS** | left/right = [1.0000, 1.0000] |
| 6 | timestamp == frame_index/25 | **PASS** | 首 4.0000s → 末 5.9600s |
| 7 | frame_index 严格递增 | **PASS** | 100 → 149 |
| 8 | clip_id 一致 | **PASS** | CNFACE_..._scene_0000_clip_0001 |
| 9 | NPZ 保存/读取回环 | **PASS** | gaze_vector dtype=float32 |

**总判定：PASS**

### 5.3 聚合统计（50 帧）

```
mean: [ 0.01038,  0.13083, -0.99094,  0.00554,  0.13089, -0.99082]
std : [ 0.01121,  0.02608,  0.00382,  0.01458,  0.02958,  0.00450]
range: [-0.9956, 0.2191]
```

z 分量稳定 ≈ -0.99（朝向相机，与 OpenFace 约定的视线坐标系一致），x/y 分量小幅波动，符合正常说话/看镜头场景。

### 5.4 产出文件（供 Stage17 读取）

```
outputs/stage13_gaze/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001/
  gaze_features.npz   (2358 B, np.savez_compressed)
    frame_index  shape=(50,)  dtype=int32    # [100..149]
    timestamp    shape=(50,)  dtype=float32  # [4.00, 4.04, ...] = frame/25
    clip_id      shape=(50,)  dtype=<U53
    gaze_vector  shape=(50,6) dtype=float32  # [T,6] 与 Stage12/14 时间线对齐
  gaze_metadata.json  (num_frames=50, gaze_dim=6)
```

### 5.5 既有回归测试

```bash
/opt/hb/bin/python3 -m pytest tests/test_stage13_gaze_real_data.py -v
# 1 passed（安装 mediapipe 前为 skipped: mediapipe not installed）
```

---

## 6. 发现与注意事项

1. **protobuf 版本必须锁定在 4.25.x**：onnx(≥4.25.1) 与 mediapipe(<4 但实测兼容 4.25.9) 的唯一交集。
   后续**不要**把 host 环境 protobuf 升级到 5.x/7.x，否则 mediapipe 运行时将失效。
2. **既有测试的 timestamp 断言缺口**：`tests/test_stage13_gaze_real_data.py::_build_aligned_face_info`
   对文件名 `aligned_000100_..._face_000.jpg` 执行 `int(img_path.stem)`（stem 非纯数字）→ `frame_index=0`、
   `timestamp=0.0`；该测试只验证形状/无 NaN/模长，**未验证 timestamp 正确性**。
   本次 `scripts/stage13_verify.py` 已补上「解析真实 frame_index → 校验 timestamp == frame_index/25」。
3. **CPU delegate**：`delegate=cpu` 时 MediaPipe 打印 EGL 初始化失败并自动回退 XNNPACK CPU 推理，
   属预期行为，不影响正确性。
4. **pip check 提示**：mediapipe 元数据声明的 `opencv-contrib-python` / `sounddevice` / `protobuf<4`
   未安装/未满足，均为**刻意取舍**（见 §4.3），Gaze 路径实测不受影响。

---

## 7. 变更清单（本次仅涉及）

| 类型 | 路径 / 动作 |
|---|---|
| host Python 包安装 | mediapipe 0.10.8、protobuf 4.25.9（7.35.1 降级）、absl-py 2.5.0 |
| 新增验证脚本 | `scripts/stage13_verify.py`（复用 Stage13 公共 API，非架构代码） |
| 新增产出数据 | `outputs/stage13_gaze/CNFACE_..._clip_0001/{gaze_features.npz, gaze_metadata.json}` |
| 新增报告 | `docs/Stage13_Runtime_Environment_Report.md`（本文件） |
| **未改动** | Stage13 全部源码（`stage.py / processor.py / integration.py / runner.py`） |
| **未改动** | 任何 Docker 配置 / Dockerfile / 运行脚本 |
| **未改动** | emica_env 环境 |

