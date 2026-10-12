# pipeline_manifest_summary.csv 字段说明（共 128 列）

> 生成脚本：`scripts/generate_manifest_summary.py`
> 一行 = 一个 clip（segment 模式下等价于一个 scene；sliding_window 下一个 scene 展开多行；stage3 拒绝但无 clip 的 scene 仍占一行，clip 字段为空）。
> 有意省略：`stage_09_5` / `stage_10` / `stage_11`。

## 类型约定
- CSV 中所有单元格都是文本；下表"类型"指字段的逻辑原始类型。
- 空值 → 空字符串（不写 `null`/`None`）。
- 布尔 → `1`（true）/ `0`（false）。
- 列表/字典 → 紧凑 JSON 文本（`ensure_ascii=False`，如 `["too_short"]`、`{"flame_shape":[268,112]}`）。
- 类型缩写：`str` 字符串、`int` 整数、`float` 浮点、`bool` 布尔、`list(JSON)` JSON 数组文本、`dict(JSON)` JSON 对象文本、`ISO` 时间戳文本。

## A. 视频级信息（每条 scene/clip 行重复）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| batch_dir | 批次目录 | str | out-root 下第一层目录名（该视频所属批次/分组）；manifest 直接位于 out-root 根目录时回退为 run_dir | `Talkvid_Chinese_videos_clips002` |
| run_dir | 运行目录名 | str | manifest 所在目录名（每个 clip/视频一个目录） | `video_0waJzbGvr1g_0_36.20_46.87_open_domain` |
| manifest_rel_path | manifest 相对路径 | str | 相对项目根目录的 manifest 路径 | `outputs/.../pipeline_manifest.json` |
| video_id | 视频 ID | str | `source_video.video_id`，无 source_video 时为空 | `0waJzbGvr1g_0_36.20_46.87` |
| pipeline_mode | 流水线模式 | str | `dataset` 或 `open_domain` | `open_domain` |
| input_video | 输入视频路径 | str | 相对项目根的输入视频路径 | `videos/Talkvid_Chinese_videos_clips002/0waJzbGvr1g_0_36.20_46.87.mp4` |
| duration_s | 视频时长(秒) | float | `source_video.duration` | `10.67` |
| width | 视频宽(像素) | int | `source_video.width` | `1920` |
| height | 视频高(像素) | int | `source_video.height` | `1080` |
| fps | 帧率 | float | `source_video.fps` | `30.0` |
| total_frames | 总帧数 | int | `source_video.total_frames` | `320` |
| final_status | 最终状态 | str | `run.final_status`：`PASS`/`PARTIAL`/`FAIL` | `PASS` |
| phase1_complete | 阶段1是否完成 | bool | `manifest.phase1_complete`；旧 manifest 无此字段→空 | `1` |
| phase2_complete | 阶段2是否完成 | bool | `manifest._phase2_complete`；旧 manifest 无此字段→空 | `1` |
| video_window_seconds | 视频滑窗秒数 | float | `run.window_seconds`；未启用滑窗→空 | 空 |
| manifest_version | manifest 版本 | str | 顶层 `manifest_version` | `1.1` |
| generated_at | 生成时间 | ISO | 顶层 `generated_at` | `2026-09-18T08:38:34Z` |
| updated_at | 最后更新时间 | ISO | 顶层 `updated_at`；可能为空 | 空 |

## B. 场景/片段身份与几何
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| scene_id | 场景 ID | str | stage_02/03 场景 ID | `0waJzbGvr1g_0_36.20_46.87_scene_0000` |
| clip_id | 片段 ID | str | clip 记录 ID；stage3 拒绝/无 clip 时为空 | `0waJzbGvr1g_0_36.20_46.87_scene_0000_clip_0000` |
| clip_count_in_scene | 场景内片段数 | int | 该 scene 对应 clip 总数（1=segment 单片段） | `1` |
| scene_start_time | 场景起始时间(秒) | float | stage_02 scene.start_time | `0.0` |
| scene_end_time | 场景结束时间(秒) | float | stage_02 scene.end_time | `10.72` |
| scene_duration_s | 场景时长(秒) | float | stage_02 scene.duration | `10.72` |
| clip_start_time | 片段起始时间(秒) | float | clip.start_time | `0.0` |
| clip_end_time | 片段结束时间(秒) | float | clip.end_time | `10.72` |
| clip_duration_s | 片段时长(秒) | float | clip.duration | `10.72` |
| clip_frame_count | 片段帧数 | int | clip.frame_count | `268` |
| clip_window_seconds | 片段滑窗秒数 | float | clip.window_seconds；未启用→空 | 空 |

## C. Stage 3 粗筛（scene 级）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_03__status | Stage3 状态 | str(派生) | 由 `passed` 派生：`passed`/`rejected`；无 stage3 数据→空 | `passed` |
| stage_03__passed | Stage3 是否通过 | bool | scene.passed（true→`1`） | `1` |
| stage_03__reject_reasons | 拒绝原因 | list(JSON) | 如 `["too_short"]`，通过时 `[]` | `[]` |
| stage_03__sampled_frame_count | 采样帧数 | int | coarse_face_info.sampled_frame_count | `11` |
| stage_03__face_detect_ratio | 人脸检出比例 | float | coarse_face_info.face_detect_ratio | `1.0` |
| stage_03__single_face_ratio | 单脸比例 | float | coarse_face_info.single_face_ratio | `1.0` |
| stage_03__avg_face_confidence | 平均人脸置信度 | float | coarse_face_info.avg_face_confidence | `0.748919` |
| stage_03__max_face_count | 单帧最大人脸数 | int | coarse_face_info.max_face_count | `1` |

## D. Stage 5 人脸分辨率门槛（视频级硬门槛；每条 scene/clip 行重复）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_05_face_res__status | 分辨率门槛状态 | str(派生) | 由 `passed` 派生：`passed`/`rejected`；该门槛未运行或无记录时为空 | `passed` |
| stage_05_face_res__passed | 是否通过分辨率门槛 | bool | `stage_details.stage_05_face_resolution_filter.passed`（回退 `audit.stage_05_face_resolution_filter.passed`） | `1` |
| stage_05_face_res__reject_reason | 拒绝原因 | str | `frame_processing_error` / `no_valid_face_crop` / `insufficient_valid_face_frames` / `face_resolution_ratio_below_threshold`；通过时为空 | `face_resolution_ratio_below_threshold` |
| stage_05_face_res__short_side_min | 人脸短边最小值(像素) | int | tight bbox 短边在有效人脸帧（PASS + FAIL_LOW_RESOLUTION）上的最小值 | `76` |
| stage_05_face_res__short_side_median | 人脸短边中位数(像素) | float | 同上，中位数 | `90.0` |
| stage_05_face_res__short_side_max | 人脸短边最大值(像素) | int | 同上，最大值 | `110` |
| stage_05_face_res__valid_face_frame_count | 有效人脸帧数 | int | PASS + FAIL_LOW_RESOLUTION 帧数（检测到目标人脸且 tight bbox 有效） | `200` |
| stage_05_face_res__qualified_ratio | 合格比例 | float | `pass_frame_count / expected_frame_count` | `0.95` |

## E. Stage 5 人脸检测（clip 级）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_05__status | Stage5 状态 | str | `passed`/`failed` | `passed` |
| stage_05__frames | 检测帧数 | int | 处理帧数 | `268` |
| stage_05__face_detections | 检出人脸帧数 | int | 检测到人脸的帧数 | `268` |
| stage_05__max_faces_in_frame | 单帧最大人脸数 | int | | `1` |
| stage_05__avg_confidence | 平均置信度 | float | 人脸置信度均值 | `0.74776` |

## Stage 5.1 跟踪（clip 级）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_05_1__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_05_1__track_count | 跟踪目标数 | int | 增强关闭时可能为 0 | `0` |
| stage_05_1__track_ids | 跟踪 ID 列表 | list(JSON) | 如 `[1,2]`，关闭时 `[]` | `[]` |

## Stage 5.2 单脸过滤（clip 级；增强关闭时多为空）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_05_2__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_05_2__reasons | 原因列表 | list(JSON) | 异常原因，可为空 | 空 |
| stage_05_2__main_track_id | 主跟踪 ID | int/str | identity_result.main_track_id；增强关闭时为空 | 空 |
| stage_05_2__total_frames | 总帧数 | int | identity_result.total_frames | 空 |
| stage_05_2__anomaly_frames | 异常帧数 | int | identity_result.anomaly_frames | 空 |
| stage_05_2__anomaly_ratio | 异常比例 | float | identity_result.anomaly_ratio | 空 |

## Stage 5.3 时序稳定性（clip 级；增强关闭时多为空）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_05_3__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_05_3__reasons | 原因列表 | list(JSON) | 异常原因，可为空 | 空 |
| stage_05_3__average_center_motion | 平均中心位移 | float | stability_result.average_center_motion | 空 |
| stage_05_3__max_center_motion | 最大中心位移 | float | stability_result.max_center_motion | 空 |
| stage_05_3__average_scale_change | 平均尺度变化 | float | stability_result.average_scale_change | 空 |
| stage_05_3__max_scale_change | 最大尺度变化 | float | stability_result.max_scale_change | 空 |
| stage_05_3__bbox_jump_count | 框跳变次数 | int | stability_result.bbox_jump_count | 空 |
| stage_05_3__anomaly_ratio | 异常比例 | float | stability_result.anomaly_ratio | 空 |

## Stage 6 头部姿态（clip 级）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_06__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_06__pose_frames | 有效姿态帧数 | int | 记录到姿态的帧数 | `268` |
| stage_06__mean_yaw | 平均航向角(yaw) | float | 单位度 | `10.935073` |
| stage_06__mean_pitch | 平均俯仰角(pitch) | float | 单位度 | `1.884584` |
| stage_06__mean_roll | 平均翻滚角(roll) | float | 单位度 | `4.62988` |
| stage_06__pose_ratio | 有效姿态比例 | float | 有效姿态帧 / 总帧 | `0.962687` |

## Stage 7 音画同步（clip 级）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_07__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_07__offset_frames | 音画偏移帧数 | int | 音频相对视频的帧偏移 | `4` |
| stage_07__sync_lse_c | LSE-C 得分 | float | 唇形同步 LSE-C | `5.503798` |
| stage_07__sync_lse_d | LSE-D 得分 | float | 唇形同步 LSE-D | `8.856267` |
| stage_07__median_distance | 中位特征距离 | float | 音画特征中位距离 | `14.360065` |

## Stage 7.1 音画对齐（clip 级）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_07_1__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_07_1__offset_frames | 偏移帧数 | int | | `4` |
| stage_07_1__offset_ms | 偏移毫秒 | float | 偏移换算毫秒 | `160.0` |
| stage_07_1__correction_applied | 是否已校正 | bool | true→`1` | `1` |
| stage_07_1__alignment_status | 对齐状态 | str | 如 `corrected`/`aligned` | `corrected` |
| stage_07_1__fps | 音频帧率 | float | | `25.0` |
| stage_07_1__clip_start | 片段起始(秒) | float | | `0.0` |
| stage_07_1__clip_duration | 片段时长(秒) | float | | `10.72` |
| stage_07_1__audio_available | 音频是否可用 | bool | true→`1` | `1` |

## Stage 8 画面质量（clip 级）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_08__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_08__quality_frames | 质量评估帧数 | int | | `268` |
| stage_08__quality_mean | 质量均分 | float | 清晰度/质量均值 | `56.130263` |
| stage_08__quality_min | 质量最低分 | float | | `44.948223` |
| stage_08__quality_max | 质量最高分 | float | | `62.163826` |

## Stage 9 融合过滤（clip 级；Stage 9.5/10/11 有意省略）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_09__status | 状态 | str | `passed`/`rejected` | `passed` |
| stage_09__pose_score | 姿态得分 | float | 0~1 | `0.962687` |
| stage_09__sync_score | 同步得分 | float | 0~1 | `1.0` |
| stage_09__quality_score | 质量得分 | float | 0~1（归一化） | `0.561303` |
| stage_09__fusion_score | 融合得分 | float | 加权融合总分 | `0.853465` |
| stage_09__is_accepted | 是否采纳 | bool | true→`1` | `1` |
| stage_09__reject_reasons | 拒绝原因 | list(JSON) | 如 `["low_fusion_score"]`，通过时 `[]` | `[]` |

## Stage 12 FLAME/EMICA 拟合（phase2，仅 phase2_complete=true 时有值）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_12__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_12__num_frames | 帧数 | int | 拟合帧数 | `268` |
| stage_12__backend | 后端 | str | 如 `EMICA+SPECTRE` | `EMICA+SPECTRE` |
| stage_12__jaw_source | 下巴参数来源 | str | 如 `emica` | `emica` |
| stage_12__head_source | 头部参数来源 | str | 如 `emica` | `emica` |
| stage_12__expression_source | 表情参数来源 | str | 如 `emica` | `emica` |
| stage_12__expression_dim | 表情维度 | int | 表情系数维度 | `100` |
| stage_12__jaw_pose_dim | 下巴位姿维度 | int | | `3` |
| stage_12__head_pose_dim | 头部位姿维度 | int | | `3` |
| stage_12__shapecode_dim | 身份形状维度 | int | 形状参数维度 | `300` |
| stage_12__reason | 失败原因 | str | 成功时为空 | 空 |

## Stage 13 视线/gaze 估计（phase2）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_13__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_13__num_frames | 帧数 | int | | `268` |
| stage_13__gaze_dim | 视线维度 | int | 视线参数维度（如 pitch/yaw 等） | `6` |
| stage_13__reason | 失败原因 | str | 成功时为空 | 空 |

## Stage 14 音频特征（phase2）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_14__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_14__num_frames | 帧数 | int | | `268` |
| stage_14__feature_dim | 音频特征维度 | int | 如 HuBERT 特征维 | `768` |
| stage_14__source_audio | 源音频路径 | str | 所用音频文件路径 | `outputs/.../stage0_11/audio/xxx.wav` |
| stage_14__reason | 失败原因 | str | 成功时为空 | 空 |

## Stage 17 融合导出（phase2）
| 字段名 | 中文含义 | 类型 | 详细说明 | 样例 |
|---|---|---|---|---|
| stage_17__status | 状态 | str | `passed`/`failed` | `passed` |
| stage_17__n_flame | FLAME 帧数 | int | 导出的 FLAME 帧数 | `268` |
| stage_17__n_gaze | gaze 帧数 | int | | `268` |
| stage_17__n_audio | audio 帧数 | int | | `268` |
| stage_17__num_frames | 总帧数 | int | | `268` |
| stage_17__valid_length | 有效长度 | int | 对齐后有效帧长 | `268` |
| stage_17__gaze_missing_frames | gaze 缺失帧数 | int | | `0` |
| stage_17__audio_missing_frames | audio 缺失帧数 | int | | `0` |
| stage_17__failed_reason | 失败原因 | str | 成功时为空 | 空 |

---

- 字段顺序与 CSV 表头一致，共 128 列。
- 级联空值：某阶段被拒绝后，其后的 clip 级 stage 字典整体为 null → 对应列全为空（如 stage3 拒绝→stage5+ 全空；stage9 拒绝→stage12+ 全空）。视频级分辨率门槛拒绝（`stage_05_face_res__status`=`rejected`）时，该视频所有行的下游 clip 级 stage 列（`stage_05` 及其后）全为空，但门槛本身 8 列仍填充。
- Stage 9.5 / 10 / 11 不导出。