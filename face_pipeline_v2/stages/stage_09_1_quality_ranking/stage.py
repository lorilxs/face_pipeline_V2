"""Stage 9.1: Quality Ranking — Optional Enhancement.

Per the Stage 9.1 design spec, this stage computes a weighted multi-metric
quality score from existing Stage 9 QualityMetrics data and assigns
a quality level (excellent / good / normal).

Key constraints:
    - Does NOT re-execute any model (6DRepNet / SyncNet / HyperIQA).
    - Does NOT modify is_accepted or reject_reasons from Stage 9.
    - Does NOT delete or filter data — all clips pass through.
    - All weights and thresholds come from config.yaml quality_ranking section.
    - When disabled via enhancements, passes input through unchanged.

Input:
    {
        "quality_metrics": List[QualityMetrics],   # from Stage 9
    }

Output: List[QualityRankingInfo]
"""

from typing import Any, Dict, List

from face_pipeline_v2.data.models import QualityMetrics, QualityRankingInfo
from face_pipeline_v2.pipeline.enhancement_stage import EnhancementStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_09_1_quality_ranking.processor import (
    RankingConfig,
    rank_clip,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
)


class Stage9_1QualityRanking(EnhancementStage):
    """Stage 9.1 — Quality Ranking (Optional Enhancement).

    Computes weighted multi-metric scores and assigns quality levels
    to clips that have already passed through Stage 9.

    Input:  dict with "quality_metrics" → List[QualityMetrics]
    Output: List[QualityRankingInfo]
    """

    _STAGE_NAME: str = "Stage 9.1 Quality Ranking"
    enhancement_key: str = "stage_09_1_quality_ranking"

    def __init__(self) -> None:
        """Initialize Stage 9.1."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # EnhancementStage interface
    # ------------------------------------------------------------------

    def _get_passthrough_output(self, input_data: Any) -> Any:
        """Return type-consistent empty list when disabled.

        Stage 9.1 input type (dict) differs from output type
        (List[QualityRankingInfo]), so the default passthrough
        (return input_data) would violate type consistency.

        Returns:
            Empty list — matches enabled output type.
        """
        return []

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input is a dict with a non-empty quality_metrics list.

        Args:
            input_data: Expected dict{"quality_metrics": List[QualityMetrics]}.
            context: PipelineContext.

        Returns:
            True if input is valid.
        """
        if not isinstance(input_data, dict):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected dict, got "
                f"{type(input_data).__name__}"
            )
            return False

        if "quality_metrics" not in input_data:
            context.logger.warning(
                f"[{self._STAGE_NAME}] Missing required key: 'quality_metrics'."
            )
            return False

        metrics_list = input_data["quality_metrics"]
        if not isinstance(metrics_list, list) or len(metrics_list) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] quality_metrics must be "
                f"a non-empty list."
            )
            return False

        if not all(isinstance(m, QualityMetrics) for m in metrics_list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements in quality_metrics "
                f"must be QualityMetrics."
            )
            return False

        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[QualityRankingInfo]:
        """Execute quality ranking on all clips from Stage 9.

        Steps:
            1. Read quality_ranking config (weights + levels).
            2. Build RankingConfig.
            3. For each QualityMetrics, compute weighted_score
               and assign quality_level.
            4. Return List[QualityRankingInfo].

        All clips are ranked regardless of is_accepted status.
        No data is deleted or modified.

        Args:
            input_data: dict with "quality_metrics" → List[QualityMetrics].
            context: PipelineContext.

        Returns:
            List[QualityRankingInfo], one per input QualityMetrics.

        Raises:
            PipelineError: If quality_ranking config section is missing.
        """
        logger = context.logger
        config = context.config
        data: dict = input_data

        metrics_list: List[QualityMetrics] = data["quality_metrics"]

        # --- Load ranking configuration ---
        ranking_cfg = self._build_ranking_config(config, logger)

        logger.info(
            f"[{self._STAGE_NAME}] Config: "
            f"weights=({ranking_cfg.pose_weight}, {ranking_cfg.sync_weight}, "
            f"{ranking_cfg.face_quality_weight}), "
            f"levels=(excellent>={ranking_cfg.excellent_threshold}, "
            f"good>={ranking_cfg.good_threshold}, "
            f"normal>={ranking_cfg.normal_threshold})"
        )

        # --- Rank each clip ---
        output: List[QualityRankingInfo] = []
        level_counts: Dict[str, int] = {"excellent": 0, "good": 0, "normal": 0}
        total_weighted = 0.0

        for metrics in metrics_list:
            ranking = rank_clip(metrics, ranking_cfg)
            output.append(ranking)
            total_weighted += ranking.weighted_score
            level_counts[ranking.quality_level] += 1

            logger.debug(
                f"[{self._STAGE_NAME}] {metrics.clip_id}: "
                f"weighted={ranking.weighted_score:.4f}, "
                f"level={ranking.quality_level}, "
                f"accepted={metrics.is_accepted}"
            )

        avg_weighted = total_weighted / len(metrics_list) if metrics_list else 0.0

        logger.info(
            f"[{self._STAGE_NAME}] Ranking complete: "
            f"{len(output)} clips ranked, "
            f"avg_weighted_score={avg_weighted:.4f}, "
            f"levels: excellent={level_counts['excellent']}, "
            f"good={level_counts['good']}, "
            f"normal={level_counts['normal']}"
        )

        return output

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output is a List[QualityRankingInfo].

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is valid.
        """
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[QualityRankingInfo], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] Output is empty: no clips ranked."
            )
        return all(isinstance(r, QualityRankingInfo) for r in output_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ranking_config(
        config: dict,
        logger: Any,
    ) -> RankingConfig:
        """Build RankingConfig from config.yaml quality_ranking section.

        All values come from config — no hard-coded defaults in logic.
        Fallback defaults are provided only for missing config keys.

        Args:
            config: Full pipeline config dict.
            logger: PipelineLogger instance.

        Returns:
            Populated RankingConfig.

        Raises:
            PipelineError: If quality_ranking section is missing entirely.
        """
        ranking_cfg: dict = config.get("quality_ranking")
        if ranking_cfg is None:
            raise PipelineError(
                "[Stage 9.1] Missing 'quality_ranking' section in config. "
                "Add quality_ranking.weights and quality_ranking.levels "
                "to config.yaml."
            )

        weights_cfg: dict = ranking_cfg.get("weights", {})
        levels_cfg: dict = ranking_cfg.get("levels", {})

        return RankingConfig(
            pose_weight=float(weights_cfg.get("pose", 0.3)),
            sync_weight=float(weights_cfg.get("sync", 0.3)),
            face_quality_weight=float(weights_cfg.get("face_quality", 0.4)),
            excellent_threshold=float(levels_cfg.get("excellent", 0.85)),
            good_threshold=float(levels_cfg.get("good", 0.70)),
            normal_threshold=float(levels_cfg.get("normal", 0.50)),
        )
