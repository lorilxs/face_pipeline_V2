"""Stage 7.1: A/V Sync Correction (Audio Timeline Alignment).

Per AISDD 3.11.6, this stage applies SyncNet-detected audio offset
to align the audio timeline with video.  Does NOT re-run SyncNet.

This is a regular pipeline stage (NOT an enhancement) because audio
consistency is a fundamental data quality requirement.

Input:
    {
        "av_sync_infos": List[AVSyncInfo],         # from Stage 7
        "std_video_info": StandardizedVideoInfo,    # from Stage 1
    }

Output: List[AVAlignmentInfo]
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from face_pipeline_v2.data.models import (
    AVAlignmentInfo,
    AVSyncInfo,
    StandardizedVideoInfo,
)
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_07_1_av_alignment.processor import (
    ALREADY_ALIGNED,
    CORRECTED,
    SEVERE_OFFSET_UNAVAILABLE,
    AlignmentConfig,
    align_audio,
    classify_alignment,
    compute_offset_ms,
    extract_clip_audio,
    shift_clip_audio,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
)


class Stage7_1AVAlignment(BaseStage):
    """Stage 7.1 — Audio-Visual Alignment Correction.

    Uses SyncNet offset (Stage 7) to correct the audio timeline
    via ffmpeg.  Outputs AVAlignmentInfo for Stage 14 consumption.

    Input:  dict with "av_sync_infos" and "std_video_info"
    Output: List[AVAlignmentInfo]
    """

    _STAGE_NAME: str = "Stage 7.1 AV Alignment"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(input_data, dict):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected dict, got "
                f"{type(input_data).__name__}"
            )
            return False

        required = {"av_sync_infos", "std_video_info"}
        if not required.issubset(input_data.keys()):
            missing = required - input_data.keys()
            context.logger.warning(
                f"[{self._STAGE_NAME}] Missing required keys: {missing}"
            )
            return False

        sync_infos = input_data["av_sync_infos"]
        if not isinstance(sync_infos, list) or len(sync_infos) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] av_sync_infos must be non-empty list."
            )
            return False
        if not all(isinstance(s, AVSyncInfo) for s in sync_infos):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements must be AVSyncInfo."
            )
            return False

        std_info = input_data["std_video_info"]
        if not isinstance(std_info, StandardizedVideoInfo):
            context.logger.warning(
                f"[{self._STAGE_NAME}] std_video_info must be "
                f"StandardizedVideoInfo, got {type(std_info).__name__}"
            )
            return False

        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[AVAlignmentInfo]:
        """Apply audio timeline correction for all clips.

        Clip-level audio generation (per design):
            1. Resolve clip boundaries from AVSyncInfo (start_frame/end_frame/
               fps/duration/timestamp), falling back to optional clip_infos.
            2. ALWAYS extract a raw clip audio first (cut from the Stage 1
               audio at [timestamp, +duration]) — the full audio is never used
               as the output asset.
            3. Apply the SyncNet offset shift ON the clip audio (not on the
               whole video audio), preserving the clip duration exactly.
            4. aligned_audio_path always exists for already_aligned/corrected;
               None only for severe_offset_unavailable or when no clip
               boundaries can be resolved (legacy fallback).

        Args:
            input_data: dict with av_sync_infos and std_video_info
                        (optionally clip_infos for boundary fallback).
            context: PipelineContext.

        Returns:
            List[AVAlignmentInfo], one per clip.
        """
        logger = context.logger
        config = context.config
        data: dict = input_data

        sync_infos: List[AVSyncInfo] = data["av_sync_infos"]
        std_info: StandardizedVideoInfo = data["std_video_info"]
        # Optional: clip_infos for per-clip boundary fallback
        clip_infos = data.get("clip_infos") or []
        clip_by_id: dict = {}
        if clip_infos:
            clip_by_id = {c.clip_id: c for c in clip_infos}

        # --- Stage config ---
        stage_cfg: dict = config.get("stage_07_1_av_alignment", {})
        if not stage_cfg.get("enabled", True):
            logger.info(f"[{self._STAGE_NAME}] Stage 7.1 disabled.")
            return []

        # --- Unified offset threshold from sync config ---
        sync_cfg: dict = config.get("sync", {})
        max_offset_frames: int = int(sync_cfg.get("max_offset_frames", 15))
        default_fps: float = float(std_info.target_fps)
        max_offset_ms: float = max_offset_frames / default_fps * 1000.0

        # --- Alignment config (Stage 7.1 own threshold + unified max) ---
        align_cfg = AlignmentConfig(
            correction_threshold_ms=float(
                stage_cfg.get("correction_threshold_ms", 50)
            ),
            max_offset_ms=max_offset_ms,
        )

        # --- Output directory ---
        storage_cfg: dict = config.get("storage", {})
        output_base = Path(storage_cfg.get("output_dir", "outputs"))
        audio_output_dir = output_base / stage_cfg.get(
            "output_dir", "stage_outputs/av_alignment"
        )

        original_audio = std_info.audio_path
        logger.info(
            f"[{self._STAGE_NAME}] Config: "
            f"threshold={align_cfg.correction_threshold_ms}ms, "
            f"max_offset={max_offset_frames}frames ({max_offset_ms:.0f}ms @{default_fps:.0f}fps), "
            f"audio={original_audio.name}"
        )

        output: List[AVAlignmentInfo] = []
        already_aligned_count = 0
        corrected_count = 0
        severe_count = 0
        boundary_count = 0

        for sync in sync_infos:
            offset = sync.offset
            # Per-clip fps: prefer AVSyncInfo.fps, fall back to std target fps.
            fps = float(sync.fps) if sync.fps else default_fps
            offset_ms = compute_offset_ms(offset, fps)

            # ----------------------------------------------------------
            # 1. Resolve clip boundaries (AVSyncInfo → clip_infos → legacy)
            # ----------------------------------------------------------
            clip_info = clip_by_id.get(sync.clip_id)
            start_frame = sync.start_frame if sync.start_frame is not None \
                else (clip_info.start_frame if clip_info else None)
            end_frame = sync.end_frame if sync.end_frame is not None \
                else (clip_info.end_frame if clip_info else None)
            timestamp = sync.timestamp if sync.timestamp is not None \
                else (clip_info.start_time if clip_info else None)
            duration = sync.duration if sync.duration is not None \
                else (clip_info.duration if clip_info else None)
            # Derive missing fields from frame boundaries (design: start_time =
            # start_frame/fps, duration = (end_frame-start_frame)/fps).
            if duration is None and start_frame is not None and end_frame is not None:
                duration = (end_frame - start_frame) / fps
            if timestamp is None and start_frame is not None:
                timestamp = start_frame / fps

            has_boundaries = (
                timestamp is not None and duration is not None and duration > 0
            )

            state = classify_alignment(offset, fps, align_cfg)

            clip_dir = audio_output_dir / sync.clip_id
            raw_audio_path: Optional[Path] = None
            final_audio_path: Optional[Path] = None
            corrected_path: Optional[Path] = None
            correction_applied = False

            if state == SEVERE_OFFSET_UNAVAILABLE:
                # 情况3: 严重错位 → 不生成可用音频
                severe_count += 1
                logger.warning(
                    f"[{self._STAGE_NAME}] {sync.clip_id}: "
                    f"SEVERE_OFFSET offset={offset}frames ({offset_ms:.1f}ms) "
                    f"> max_correction={align_cfg.max_offset_ms:.0f}ms "
                    f"→ audio unavailable (audio_offset_exceeded_max_correction)"
                )
            elif has_boundaries:
                # 情况1/2: clip 级音频 — 先裁剪 raw（duration 与 clip 视频一致），
                # 再在 clip 音频上 shift（duration 严格保持），绝不 shift 整段视频。
                boundary_count += 1
                raw_audio_path = clip_dir / f"{sync.clip_id}_raw.wav"
                extract_clip_audio(original_audio, raw_audio_path,
                                   timestamp, duration)
                aligned_path = clip_dir / f"{sync.clip_id}.wav"
                if state == ALREADY_ALIGNED:
                    # 无需修正 → aligned = raw（复制，不 shift）
                    shutil.copy2(str(raw_audio_path), str(aligned_path))
                    correction_applied = False
                    already_aligned_count += 1
                    logger.debug(
                        f"[{self._STAGE_NAME}] {sync.clip_id}: "
                        f"ALREADY_ALIGNED offset={offset}frames "
                        f"({offset_ms:.1f}ms) → raw={raw_audio_path.name} "
                        f"aligned=copy ({duration:.2f}s)"
                    )
                else:  # CORRECTED
                    # 需要校正 → 在 clip 音频上做 offset shift（duration 保持 8s）
                    shift_clip_audio(raw_audio_path, aligned_path,
                                     offset, fps, duration)
                    correction_applied = True
                    corrected_path = aligned_path
                    corrected_count += 1
                    logger.debug(
                        f"[{self._STAGE_NAME}] {sync.clip_id}: "
                        f"CORRECTED offset={offset}frames ({offset_ms:.1f}ms) "
                        f"→ shifted on clip audio, duration={duration:.2f}s"
                    )
                final_audio_path = aligned_path
            else:
                # 无 clip 边界 → legacy 回退（不应出现在真实流程；避免把整段
                # 视频音频当作 clip 音频输出 —— 这正是本修复消除的 bug）。
                if state == ALREADY_ALIGNED:
                    # 无法确定裁剪窗口 → 不生成音频（宁可缺失，不产生整段音频）
                    already_aligned_count += 1
                    logger.warning(
                        f"[{self._STAGE_NAME}] {sync.clip_id}: "
                        f"ALREADY_ALIGNED but no clip boundaries resolvable "
                        f"→ aligned_audio_path=None (no clip-level audio)"
                    )
                else:  # CORRECTED
                    aligned_path = clip_dir / f"{sync.clip_id}_corrected.wav"
                    align_audio(original_audio, aligned_path, offset, fps)
                    correction_applied = True
                    corrected_path = aligned_path
                    corrected_count += 1
                    logger.debug(
                        f"[{self._STAGE_NAME}] {sync.clip_id}: "
                        f"CORRECTED (legacy, no clip boundaries)"
                    )
                    final_audio_path = aligned_path

            output.append(
                AVAlignmentInfo(
                    clip_id=sync.clip_id,
                    original_audio_path=original_audio,
                    corrected_audio_path=corrected_path,
                    offset_frames=offset,
                    offset_ms=round(offset_ms, 2),
                    correction_applied=correction_applied,
                    fps=fps,
                    aligned_audio_path=final_audio_path,
                    alignment_status=state,
                    raw_audio_path=raw_audio_path,
                    clip_start=timestamp,
                    clip_duration=duration,
                )
            )

        logger.info(
            f"[{self._STAGE_NAME}] Audio Alignment Summary: "
            f"total_clips={len(output)}, "
            f"already_aligned={already_aligned_count}, "
            f"corrected={corrected_count}, "
            f"severe_offset={severe_count}, "
            f"clip_level_audio={boundary_count}, "
            f"audio_generated={already_aligned_count + corrected_count}, "
            f"audio_unavailable={severe_count}"
        )

        return output

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[AVAlignmentInfo], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Output is empty "
                f"(stage disabled or no clips)."
            )
        return all(isinstance(a, AVAlignmentInfo) for a in output_data)
