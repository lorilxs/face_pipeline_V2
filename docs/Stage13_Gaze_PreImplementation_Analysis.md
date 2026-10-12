# Stage13 Gaze Extraction — Pre-Implementation Analysis

**Date**: 2026-08-26
**Status**: Analysis — 未实现
**Source**: 旧管线 `handle_stage3_pre_gazed_mediapipe.py`

---

## 1. 旧管线核心算法分析

### 1.1 MediaPipe FaceLandmarker 初始化

```python
# 旧管线 (handle_stage3_pre_gazed_mediapipe.py:148-159)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

delegate = mp_python.BaseOptions.Delegate.GPU  # 或 CPU
base_options = mp_python.BaseOptions(
    model_asset_path=STAGE3_MEDIAPIPE_MODEL,  # face_landmarker.task
    delegate=delegate,
)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,          # 关闭不需要的输出
    output_facial_transformation_matrixes=False,
    num_faces=1,                            # 仅处理单张人脸
)
detector = mp_vision.FaceLandmarker.create_from_options(options)
```

关键参数：
- 使用 `mp_vision.FaceLandmarker`（非旧版 `mp.solutions.face_mesh`）
- `num_faces=1` — 单张人脸处理
- 关闭 blendshapes 和 transformation matrixes 以减少开销
- 支持 GPU/CPU delegate 切换

### 1.2 Landmark 索引使用

旧管线使用的 MediaPipe FaceMesh 478 点拓扑索引：

| 关键点 | 索引 | 用途 |
|--------|------|------|
| 左眼轮廓 | `[33, 160, 158, 133, 153, 144, 163, 7]` | 左眼中心计算 |
| 右眼轮廓 | `[362, 385, 387, 263, 373, 380, 381, 362]` (最后一个重复) | 右眼中心计算 |
| **左眼瞳孔 (iris)** | **468** | 瞳孔位置 |
| **右眼瞳孔 (iris)** | **473** | 瞳孔位置 |
| 鼻尖 | 1 | 面部参考点 |
| 额头 | 10 | 面部法向量计算 |
| 下巴 | 152 | 面部法向量计算 |
| 左脸颊 | 234 | 面部宽度计算 |
| 右脸颊 | 454 | 面部宽度计算 |

**关键发现**: 468/473 是 MediaPipe FaceMesh 的 **iris landmarks**（瞳孔中心），不是眼球中心。这是 MediaPipe 2023 年新增的 iris tracking 功能，精度高于旧版单纯的眼球轮廓中心。
### 1.3 Gaze Vector 计算算法

```python
def get_gaze_vector_from_landmarks(landmarks, frame_width, frame_height):
```

#### Step 1: 坐标转换 — 归一化 → 像素空间

```python
def lm_to_3d(idx):
    l = landmarks[idx]
    return np.array([l.x * frame_width, l.y * frame_height, l.z * frame_width])
```

- `landmarks[idx].x/y` 是归一化坐标 [0, 1]
- `landmarks[idx].z` 是深度值（相对于头部中心），缩放因子与图像宽度一致
- 输出为 3D 像素空间坐标 (x, y, z)

#### Step 2: 面部法向量计算

```python
v1 = p_right_cheek - p_left_cheek      # 脸颊连线 (左右方向)
v2 = p_chin - p_forehead               # 下巴-额头连线 (上下方向)
face_normal = np.cross(v1, v2)          # 叉积 → 面部法向量
face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)  # 归一化

if face_normal[2] > 0:
    face_normal = -face_normal          # 确保 z 朝摄像头方向为负
```

**物理意义**: face_normal 是面部平面的法向量，指向面部前方。当人脸正对摄像头时，`face_normal ≈ (0, 0, -1)`。

#### Step 3: 瞳孔偏移计算

```python
left_eye_center = np.mean([lm_to_3d(i) for i in left_eye_indices], axis=0)
right_eye_center = np.mean([lm_to_3d(i) for i in right_eye_indices], axis=0)
left_pupil = lm_to_3d(left_pupil_idx)    # 468
right_pupil = lm_to_3d(right_pupil_idx)  # 473

face_width = np.linalg.norm(p_right_cheek - p_left_cheek)
left_offset = (left_pupil - left_eye_center) / (face_width + 1e-8)
right_offset = (right_pupil - right_eye_center) / (face_width + 1e-8)
```

- 瞳孔偏移 = 瞳孔位置 − 眼球中心位置
- 使用面部宽度归一化，使输出与图像分辨率无关

#### Step 4: 视线方向计算

```python
gaze_weight = 3.0
left_gaze = face_normal + gaze_weight * left_offset
right_gaze = face_normal + gaze_weight * right_offset
left_gaze = left_gaze / (np.linalg.norm(left_gaze) + 1e-8)
right_gaze = right_gaze / (np.linalg.norm(right_gaze) + 1e-8)
```

**核心公式**: `gaze_direction = normalize(face_normal + 3.0 × pupil_offset)`

- `face_normal`: 面部朝向（头朝向）
- `pupil_offset`: 瞳孔相对于眼球中心的偏移（眼朝向）
- `gaze_weight = 3.0`: 经验权重，放大瞳孔偏移的影响
- 归一化后输出单位向量

#### Step 5: Fallback — 无瞳孔检测

当 iris landmarks 不可用时，退化为纯面部法向量（假设视线方向=面部朝向），输出 `(face_normal, face_normal)` 作为左右眼 gaze。

### 1.4 输出维度

| 字段 | 维度 | 含义 |
|------|------|------|
| `gaze_0_x` | 1 | 左眼视线 x 分量 |
| `gaze_0_y` | 1 | 左眼视线 y 分量 |
| `gaze_0_z` | 1 | 左眼视线 z 分量 |
| `gaze_1_x` | 1 | 右眼视线 x 分量 |
| `gaze_1_y` | 1 | 右眼视线 y 分量 |
| `gaze_1_z` | 1 | 右眼视线 z 分量 |

**总计 6 维**，与 `GazeFeature.gaze_vector: Tuple[float, float, float, float, float, float]` 一致。

### 1.5 可复用性判断

| 组件 | 可复用 | 说明 |
|------|--------|------|
| `get_gaze_vector_from_landmarks()` 核心算法 | ✅ 完全复用 | 直接移植到 `processor.py` |
| Landmark 索引 | ✅ 完全复用 | 468/473 是 MediaPipe 标准 |
| FaceLandmarker 初始化 | ✅ 复用 | 模型文件已存在 |
| `process_video()` 视频读取逻辑 | ❌ 不适用 | 新管线不读取视频 |
| `find_frame_dir()` 帧目录查找 | ❌ 不适用 | 新管线直接用 AlignedFaceInfo |
| csv 输出格式 | ❌ 不适用 | 新管线输出 GazeFeature dataclass |
---

## 2. 旧管线 vs 新管线差异分析

### 2.1 数据流对比

**旧管线**:
```
video → frames (jpg/png) → MediaPipe FaceLandmarker → gazex.csv (per frame)
```
- 重新读取视频帧图像
- 对每帧进行完整的人脸检测 + landmark 提取
- 输出 per-frame csv 文件

**新管线**:
```
Stage11 AlignedFaceInfo (256×256 aligned face)
    │
    ├──→ Stage12 FLAMEFeature (expression, jaw, pose, eye_pose placeholder)
    │
    └──→ Stage13 GazeFeature (left_eye, right_eye, gaze_vector)
```

### 2.2 输入分析

**问题**: Stage13 应该用什么作为输入？

| 选项 | 数据源 | 优点 | 缺点 |
|------|--------|------|------|
| **A. AlignedFaceInfo** | Stage11 `aligned_face_path` (256×256) | 人脸已对齐、标准化 | 损失原始图像尺寸信息 |
| B. FrameInfo | Stage5 `image_path` (原始帧) | 保留完整图像信息 | 需重新读取视频帧 |
| C. Video 直接读取 | 原始视频 | 完全独立 | 重复 IO，破坏 pipeline 架构 |

**推荐**: 选项 A — 使用 `AlignedFaceInfo.aligned_face_path`

**理由**:
1. AlignedFaceInfo 是 Stage11 的标准输出，已经在 pipeline 中可用
2. 256×256 对齐人脸足以进行 landmark 检测（MediaPipe 内部会自动缩放）
3. 不重新读取视频，遵循 pipeline 设计原则
4. 与 Stage12 消费同一数据源，保持 pipeline 一致性

**注意**: 使用对齐人脸时，MediaPipe 的 `frame_width`/`frame_height` 参数应设为 `(256, 256)`。但 `l.z * frame_width` 的缩放因子将基于 256 而非原始图像尺寸。由于 gaze vector 最终会被归一化，且 `face_width` 归一化使用相对值，gaze 方向不受图像分辨率影响。

### 2.3 输出分析

新管线 `GazeFeature` 定义（models.py:679-693）:

```python
@dataclass(frozen=True)
class GazeFeature:
    frame_index: int
    left_eye: Tuple[float, float, float]      # [3]
    right_eye: Tuple[float, float, float]     # [3]
    gaze_vector: Tuple[float, float, float, float, float, float]  # [6]
```

与旧管线映射：

| GazeFeature | 旧管线输出 | 说明 |
|-------------|-----------|------|
| `frame_index` | frame_id (from filename) | 从 AlignedFaceInfo 获取 |
| `left_eye` | `(gaze_0_x, gaze_0_y, gaze_0_z)` | 左眼视线方向向量 |
| `right_eye` | `(gaze_1_x, gaze_1_y, gaze_1_z)` | 右眼视线方向向量 |
| `gaze_vector` | `(gaze_0_x, ..., gaze_1_z)` | 6 维拼接，与 Stage17 TrainingSample 兼容 |

### 2.4 需要新增的字段

当前 `GazeFeature` 缺少以下字段，建议新增：

| 字段 | 类型 | 建议 | 理由 |
|------|------|------|------|
| `timestamp` | `float` | **新增** | 与 Stage14 AudioFeature 时间对齐 |
| `clip_id` | `str` | **新增** | 与 Stage12 FLAMEFeature 保持一致 |

### 2.5 与 Stage12 FLAMEFeature 的关系

Stage12 的 `eye_pose` 字段当前为 `zeros(6)` (placeholder)：

```python
# processor.py:106
eye_pose=tuple([0.0] * 6),  # fixed zeros, Stage13 placeholder
```

**设计意图**: Stage13 的 gaze 输出最终应替换 Stage12 的 `eye_pose`。但 Stage13 不修改 Stage12 的输出，替换逻辑由 Stage17 负责。
---

## 3. MediaPipe 模型文件验证

### 3.1 文件存在性

| 路径 | 存在 | 大小 | MD5 |
|------|------|------|-----|
| `models/mediapipe/face_landmarker.task` | ✅ | 3,758,596 bytes | `b0e7274907a1644404fef66b28dd6d85` |
| 旧管线模型 | ✅ | 相同 | `b0e7274907a1644404fef66b28dd6d85` |

**结论**: 模型文件已存在且与旧管线完全一致，MD5 匹配。

### 3.2 文件格式

`face_landmarker.task` 是 MediaPipe Task API 的标准 `.task` 格式（TFLite 模型 + 元数据 bundle）。可直接用于：

```python
base_options = mp_python.BaseOptions(
    model_asset_path="models/mediapipe/face_landmarker.task",
)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
detector = mp_vision.FaceLandmarker.create_from_options(options)
```

### 3.3 模型能力

该模型包含：
- 478 个 3D face landmarks（含 iris landmarks 468, 473）
- 支持 blendshapes 输出（已关闭以节省开销）
- 支持 facial transformation matrix 输出（已关闭）

---

## 4. Stage13 最小实现方案

### 4.1 设计原则

1. **不修改 Stage12** — Stage12 的 `eye_pose` 保持 zeros(6)
2. **不重新读取视频** — 只消费 Stage11 输出的 `AlignedFaceInfo`
3. **独立输出 GazeFeature** — 作为独立 Stage 输出，不嵌入 FLAMEFeature
4. **复用旧管线算法** — 核心 `get_gaze_vector_from_landmarks()` 直接移植

### 4.2 输入

```
List[AlignedFaceInfo]  (from Stage 11)
```

每个 `AlignedFaceInfo` 提供：
- `aligned_face_path`: 256×256 对齐人脸图像
- `frame_index`: 帧编号
- `timestamp`: 时间戳
- `clip_id`: Clip 标识

### 4.3 输出

```
List[GazeFeature]  (per frame)
```

### 4.4 文件结构

```
face_pipeline_v2/stages/stage_13_gaze/
├── __init__.py      # (已存在)
├── processor.py     # 核心算法: extract_gaze_parameters()
├── integration.py   # read_stage13_outputs() / invoke_stage13_runner()
├── runner.py        # 独立 subprocess runner (可选)
└── stage.py         # Stage13Gaze 类
```

### 4.5 processor.py 核心逻辑

```python
def extract_gaze_parameters(
    aligned_faces: List[AlignedFaceInfo],
    detector: mp_vision.FaceLandmarker,
    image_size: Tuple[int, int] = (256, 256),
) -> List[GazeFeature]:
    """Extract gaze vectors from aligned face images using MediaPipe.

    For each aligned face:
    1. Read aligned_face_path → BGR image
    2. Convert to RGB → MediaPipe Image
    3. Run FaceLandmarker.detect()
    4. Call get_gaze_vector_from_landmarks() → 6-dim gaze vector
    5. Build GazeFeature dataclass
    """
    results = []
    for face_info in aligned_faces:
        img = cv2.imread(str(face_info.aligned_face_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=img_rgb
        )
        detection = detector.detect(mp_image)

        if not detection.face_landmarks:
            continue

        landmarks = detection.face_landmarks[0]
        h, w = image_size
        gaze_data = get_gaze_vector_from_landmarks(landmarks, w, h)

        results.append(GazeFeature(
            frame_index=face_info.frame_index,
            left_eye=(gaze_data["gaze_0_x"],
                      gaze_data["gaze_0_y"],
                      gaze_data["gaze_0_z"]),
            right_eye=(gaze_data["gaze_1_x"],
                       gaze_data["gaze_1_y"],
                       gaze_data["gaze_1_z"]),
            gaze_vector=(gaze_data["gaze_0_x"],
                         gaze_data["gaze_0_y"],
                         gaze_data["gaze_0_z"],
                         gaze_data["gaze_1_x"],
                         gaze_data["gaze_1_y"],
                         gaze_data["gaze_1_z"]),
        ))
    return results
```

### 4.6 移植 `get_gaze_vector_from_landmarks`

从旧管线直接移植，无需修改核心算法。唯一调整：

- `frame_width` / `frame_height` → 固定为 `(256, 256)`（对齐人脸尺寸）
- 由于 gaze vector 最终归一化 + face_width 归一化，分辨率不影响结果

### 4.7 Config 扩展

```yaml
# configs/config.yaml
gaze:
  model: MediaPipe
  model_path: models/mediapipe/face_landmarker.task
  output_dim: 6
  image_size: [256, 256]   # 对齐人脸尺寸
  delegate: cpu             # cpu | gpu
```

### 4.8 依赖

- `mediapipe` — 检查 `emica_env` Docker 环境是否可用
- 不需要额外依赖（cv2, numpy 已在现有环境中）

### 4.9 与 Stage12 的并发独立性

Stage12 和 Stage13 都消费 `AlignedFaceInfo`，但消费的是不同字段：
- Stage12: `aligned_face_path` → EMICA/SPECTRE encoder → FLAMEFeature
- Stage13: `aligned_face_path` → MediaPipe FaceLandmarker → GazeFeature

两者可以并行执行（无数据依赖），但当前 pipeline 按 Stage 顺序执行。

### 4.10 测试建议

1. 单元测试: `get_gaze_vector_from_landmarks()` 的 fallback 逻辑（无 iris landmarks 时返回 face_normal）
2. 集成测试: 端到端 Stage11→Stage13 流程
3. 输出验证: 确认 `gaze_vector` 6 维为归一化向量（L2 ≈ 1.0）

---

## 5. 实施建议

### 5.1 优先级

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 1. 移植 `get_gaze_vector_from_landmarks()` | P0 | 核心算法 |
| 2. 实现 `processor.py` | P0 | Stage13 核心逻辑 |
| 3. 实现 `stage.py` | P0 | Stage13Gaze 类 |
| 4. 实现 `integration.py` | P0 | 输入输出序列化 |
| 5. 新增 `GazeFeature.timestamp` / `clip_id` | P1 | 下游兼容性 |
| 6. 实现 `runner.py` (subprocess mode) | P2 | 可选，如果 MediaPipe 在 host 环境不可用 |

### 5.2 风险

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| MediaPipe 在 host Python 不可用 | 🟡 中 | 使用 subprocess 模式（类似 Stage12）或确认 host 已安装 |
| 对齐人脸 (256×256) 影响 landmark 精度 | 🟢 低 | MediaPipe 内部 resize，256×256 足够 |
| 旧管线 iris indices 468/473 与新模型版本不兼容 | 🟢 低 | 模型文件 MD5 一致，indices 是 MediaPipe 标准 |

---

## 6. 总结

旧管线 `handle_stage3_pre_gazed_mediapipe.py` 的核心算法可直接复用：

- **核心公式**: `gaze = normalize(face_normal + 3.0 × pupil_offset)`
- **关键索引**: iris landmarks 468 (左), 473 (右)
- **输出维度**: 6 维 gaze vector，与 `GazeFeature` 定义一致
- **模型文件**: 已存在于 `models/mediapipe/face_landmarker.task`，MD5 与旧管线一致

新管线 Stage13 仅需：
1. 移植 `get_gaze_vector_from_landmarks()` 函数
2. 实现 `extract_gaze_parameters()` 处理 `List[AlignedFaceInfo]`
3. 输出 `List[GazeFeature]`
4. （可选）在 `GazeFeature` 中新增 `timestamp` 和 `clip_id` 字段

在 Stage17 中，`eye_pose` 应取自 `GazeFeature.gaze_vector`（6 维），而非 `FLAMEFeature.eye_pose`（zeros(6)）。