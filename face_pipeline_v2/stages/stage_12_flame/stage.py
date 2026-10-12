"""Stage 12: FLAME Parameter Extraction (EMICA + SPECTRE).

Per AISDD 3.16, this stage extracts FLAME 3D face parameters
using EMICA + SPECTRE dual-model inference on aligned face images
from Stage 11 (jaw_pose = EMICA — FINAL adopted scheme; head/expression = EMICA).
SPECTRE raw jaw is retained on the feature as an audit/debug field only.

Two runtime modes:
  - "direct":   Load models via ModelManager, run in-process
                (requires EMICA/SPECTRE deps in current Python env).
  - "subprocess": Invoke Docker container via integration module
                  (default, for host environments without EMICA deps).

Input:  List[AlignedFaceInfo]  (from Stage 11)
Output: List[FLAMEFeature]
"""

from pathlib import Path
from typing import Any, List

from face_pipeline_v2.data.models import AlignedFaceInfo, FLAMEFeature
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_12_flame.processor import (
    extract_flame_parameters,
)
class Stage12FLAME(BaseStage):
    """Stage 12 — FLAME Parameter Extraction (EMICA + SPECTRE).

    Input:  List[AlignedFaceInfo]
    Output: List[FLAMEFeature]
    """

    _STAGE_NAME: str = "Stage 12 FLAME Parameter Extraction"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(input_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[AlignedFaceInfo], "
                f"got {type(input_data).__name__}"
            )
            return False
        if len(input_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Input is empty: no faces to process."
            )
            return True
        return all(isinstance(m, AlignedFaceInfo) for m in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[FLAMEFeature]:
        aligned_faces: List[AlignedFaceInfo] = input_data
        config: dict = context.config
        logger = context.logger

        if len(aligned_faces) == 0:
            logger.info(
                f"[{self._STAGE_NAME}] No aligned faces to process. "
                f"Returning empty list."
            )
            return []

        stage12_cfg: dict = config.get("stage12", {})
        runtime_mode: str = stage12_cfg.get("runtime_mode", "subprocess")
        # jaw_fusion_lambda is DEPRECATED (jaw_pose = EMICA always). In config
        # it may be a deprecated marker dict {deprecated, reason, ...}; treat
        # any non-numeric value as the legacy default 1.0 (metadata only).
        jfl_raw = stage12_cfg.get("jaw_fusion_lambda", 1.0)
        if isinstance(jfl_raw, dict):
            jaw_fusion_lambda: float = float(jfl_raw.get("legacy_default", 1.0))
        else:
            jaw_fusion_lambda = float(jfl_raw)
        jaw_source: str = str(stage12_cfg.get("jaw_source", "emica"))
        expression_source: str = str(
            stage12_cfg.get("expression_source", "emica")
        )
        head_source: str = str(stage12_cfg.get("head_source", "emica"))
        batch_size: int = int(stage12_cfg.get("batch_size", 8))

        if runtime_mode == "subprocess":
            results = self._process_subprocess(
                aligned_faces=aligned_faces,
                stage12_cfg=stage12_cfg,
                context=context,
                jaw_fusion_lambda=jaw_fusion_lambda,
                expression_source=expression_source,
                head_source=head_source,
                jaw_source=jaw_source,
                batch_size=batch_size,
            )
        else:
            results = self._process_direct(
                aligned_faces=aligned_faces,
                stage12_cfg=stage12_cfg,
                context=context,
                jaw_fusion_lambda=jaw_fusion_lambda,
                expression_source=expression_source,
                head_source=head_source,
                jaw_source=jaw_source,
                batch_size=batch_size,
            )

        logger.info(
            f"[{self._STAGE_NAME}] FLAME extraction complete: "
            f"{len(aligned_faces)} faces → {len(results)} FLAMEFeature entries"
        )
        return results

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[FLAMEFeature], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Output is empty: "
                f"no FLAME parameters extracted."
            )
            return True
        if not all(isinstance(f, FLAMEFeature) for f in output_data):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements must be FLAMEFeature."
            )
            return False

        f0 = output_data[0]
        expected_dims = {
            "shapecode": 300,
            "expcode": 100,
            "jawpose": 3,
            "globalpose": 3,
            "cam": 3,
            "eye_pose": 6,
        }
        for name, dim in expected_dims.items():
            val = getattr(f0, name)
            actual = len(val) if val is not None else 0
            if actual != dim:
                context.logger.warning(
                    f"[{self._STAGE_NAME}] {name} has {actual} dims, "
                    f"expected {dim}."
                )
                return False

        if f0.texcode is not None and len(f0.texcode) != 50:
            context.logger.warning(
                f"[{self._STAGE_NAME}] texcode has {len(f0.texcode)} dims, "
                f"expected 50."
            )
            return False
        if f0.lightcode is not None and len(f0.lightcode) != 27:
            context.logger.warning(
                f"[{self._STAGE_NAME}] lightcode has {len(f0.lightcode)} dims, "
                f"expected 27."
            )
            return False
        if f0.spectre_jawpose is not None and len(f0.spectre_jawpose) != 3:
            context.logger.warning(
                f"[{self._STAGE_NAME}] spectre_jawpose has "
                f"{len(f0.spectre_jawpose)} dims, expected 3."
            )
            return False
        return True
# ------------------------------------------------------------------
    # Process: Direct (in-process, ModelManager)
    # ------------------------------------------------------------------

    def _process_direct(
        self,
        aligned_faces: List[AlignedFaceInfo],
        stage12_cfg: dict,
        context: PipelineContext,
        jaw_fusion_lambda: float,
        expression_source: str,
        head_source: str,
        jaw_source: str,
        batch_size: int,
    ) -> List[FLAMEFeature]:
        """Run Stage12 in-process using ModelManager (requires EMICA deps)."""
        config = context.config
        logger = context.logger

        emica_model = self._get_model("emica", config, context, logger)
        spectre_model = self._get_model("spectre", config, context, logger)
        device: str = getattr(context.model_manager, "device", "cpu")

        logger.info(
            f"[{self._STAGE_NAME}] Direct mode: batch_size={batch_size}, "
            f"jaw_fusion_lambda={jaw_fusion_lambda}, "
            f"expression_source={expression_source}, head_source={head_source}, "
            f"jaw_source={jaw_source}, device={device}"
        )

        return extract_flame_parameters(
            aligned_faces=aligned_faces,
            emica_model=emica_model,
            spectre_model=spectre_model,
            jaw_fusion_lambda=jaw_fusion_lambda,
            expression_source=expression_source,
            head_source=head_source,
            jaw_source=jaw_source,
            batch_size=batch_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Process: Subprocess (Docker container)
    # ------------------------------------------------------------------

    def _process_subprocess(
        self,
        aligned_faces: List[AlignedFaceInfo],
        stage12_cfg: dict,
        context: PipelineContext,
        jaw_fusion_lambda: float,
        expression_source: str,
        head_source: str,
        jaw_source: str,
        batch_size: int,
    ) -> List[FLAMEFeature]:
        """Run Stage12 via Docker subprocess (host env without EMICA deps)."""
        import tempfile
        from face_pipeline_v2.stages.stage_12_flame.integration import (
            invoke_stage12_runner,
            read_stage12_outputs,
            stage12_aligned_faces_to_metadata,
        )

        logger = context.logger

        clip_id = self._get_clip_id(aligned_faces, context)
        output_dir = Path(
            stage12_cfg.get("output_dir", "outputs/stage12_flame")
        )

        emica_cfg = stage12_cfg.get("emica", {})
        spectre_cfg = stage12_cfg.get("spectre", {})
        emica_ckpt = emica_cfg.get(
            "checkpoint",
            "models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/"
            "model-epoch=38-val/loss_total=3.88806534.ckpt",
        )
        spectre_ckpt = spectre_cfg.get(
            "checkpoint",
            "third_party/inferno/spectre/spectre/data/spectre_model.tar",
        )

        device = stage12_cfg.get("device", "cuda")
        timeout = int(stage12_cfg.get("subprocess_timeout", 3600))

        meta_path = Path(tempfile.mkdtemp(prefix="stage12_meta_")) / "input_metadata.json"
        stage12_aligned_faces_to_metadata(aligned_faces, meta_path)

        logger.info(
            f"[{self._STAGE_NAME}] Subprocess mode: clip_id={clip_id}, "
            f"batch_size={batch_size}, jaw_fusion_lambda={jaw_fusion_lambda}, "
            f"expression_source={expression_source}, head_source={head_source}, "
            f"jaw_source={jaw_source}, device={device}"
        )
        logger.info(
            f"[{self._STAGE_NAME}] Metadata written: {meta_path} "
            f"({len(aligned_faces)} faces)"
        )

        invoke_stage12_runner(
            input_metadata=meta_path,
            output_dir=output_dir,
            emica_checkpoint=emica_ckpt,
            spectre_checkpoint=spectre_ckpt,
            jaw_fusion_lambda=jaw_fusion_lambda,
            expression_source=expression_source,
            head_source=head_source,
            jaw_source=jaw_source,
            batch_size=batch_size,
            device=device,
            clip_id=clip_id,
            timeout=timeout,
        )

        return read_stage12_outputs(output_dir, clip_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_clip_id(
        aligned_faces: List[AlignedFaceInfo],
        context: PipelineContext,
    ) -> str:
        """Determine clip_id from faces or context."""
        if aligned_faces and aligned_faces[0].clip_id:
            return aligned_faces[0].clip_id
        return context.runtime_info.get("clip_id", "unknown_clip")

    @staticmethod
    def _get_model(
        model_name: str,
        config: dict,
        context: PipelineContext,
        logger: Any,
    ) -> Any:
        """Get or load a model through ModelManager."""
        try:
            return context.model_manager.get_model(model_name)
        except (KeyError, AttributeError):
            pass

        stage12_cfg: dict = config.get("stage12", {})
        model_cfg: dict = stage12_cfg.get(model_name, {})
        checkpoint_path = Path(
            model_cfg.get(
                "checkpoint",
                f"models/{model_name}/checkpoint.ckpt",
            )
        )

        logger.info(
            "Loading %s: checkpoint=%s", model_name.upper(), checkpoint_path,
        )
        try:
            return context.model_manager.load_model(
                model_name,
                str(checkpoint_path),
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s", model_name.upper(), exc)
            raise PipelineError(
                f"Failed to load {model_name.upper()} model."
            ) from exc
