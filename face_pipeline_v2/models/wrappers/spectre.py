"""SPECTRE model wrapper — minimal E_flame encoder loading (head/jaw/expression).

Per AISDD 4.5.1, Stages MUST NOT interact with the underlying model library
directly. This wrapper loads only the SPECTRE E_flame (ResnetEncoder) and
provides head_pose / jaw_pose / expression extraction for Stage 12.

Note on ``expression`` semantics (Stage12 EMICA vs SPECTRE ablation):
    SPECTRE has NO dedicated ExpressionEncoder. Its ``expression`` is the
    50-dim latent slice ``parameters[:, 150:200]`` of the E_flame output
    (``n_exp=50``), whereas EMICA expression comes from a dedicated
    expression encoder (100-dim). To keep the pipeline ``expression_dim=100``
    / Stage17 layout ``[0:100]`` untouched, the 50-dim SPECTRE slice is
    zero-padded in the last 50 dims.

Does NOT load: renderer, FLAME decoder, FLAMETex, E_expression, pytorch3d.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Ensure SPECTRE source is importable (third_party/inferno/spectre).
_SPECTRE_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "inferno" / "spectre"
if str(_SPECTRE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPECTRE_ROOT))


class SPECTREWrapper:
    """SPECTRE face reconstruction model wrapper — E_flame encoder only.

    Loads the ResnetEncoder (E_flame) from a SPECTRE/DECA checkpoint
    and extracts head_pose [3] + jaw_pose [3] + expression [100] (the
    50-dim latent slice zero-padded to 100) from the full parameter
    vector. No renderer, FLAME decoder, or E_expression is loaded.
    """

    HEAD_POSE_DIM = 3
    JAW_POSE_DIM = 3
    EXPRESSION_DIM = 50       # SPECTRE E_flame raw n_exp
    PADDED_EXPRESSION_DIM = 100  # pipeline expression_dim (zero-padded)
    EXPECTED_INPUT_SIZE = (224, 224)

    # SPECTRE E_flame n_param decomposition
    # param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
    # n_shape=100, n_tex=50, n_exp=50, n_pose=6, n_cam=3, n_light=27
    _N_PARAM = 236  # 100 + 50 + 50 + 6 + 3 + 27
    _EXPR_START = 150  # 100 + 50
    _EXPR_END = 200    # 100 + 50 + 50
    _POSE_START = 200  # 100 + 50 + 50
    _POSE_END = 206     # 200 + 6
    _N_COND = 53        # PerceptualEncoder (E_expression) output: 50 exp + 3 jaw

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path: Path = Path(checkpoint_path)
        self._device: str = device
        self._model: Any = None
        self._expression_model: Any = None
# ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _build_expression_encoder(self) -> Any:
        """Build the PerceptualEncoder (E_expression) WITHOUT torch.hub/network.

        Mirrors SPECTRE ``src/models/encoders.py`` PerceptualEncoder with
        backbone='mobilenetv2'. The mobilenet_v2 stem comes from the locally
        installed torchvision; all weights are fully overwritten by the
        checkpoint via ``load_state_dict``, so no pretrained download is
        needed. The Conv1d treats the batch axis as the time axis (official
        SPECTRE temporal encoding), and the jaw x-component is ReLU-clamped
        (official behaviour).
        """
        import torch.nn.functional as F

        try:
            from torchvision.models import mobilenet_v2
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("torchvision unavailable for E_expression: %s", exc)
            raise

        class _PerceptualEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = mobilenet_v2(weights=None)
                self.temporal = torch.nn.Sequential(
                    torch.nn.Conv1d(
                        1280, 256, kernel_size=5, stride=1, padding=2
                    ),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.ReLU(),
                )
                self.layers = torch.nn.Sequential(torch.nn.Linear(256, 53))

            def forward(self, inputs: torch.Tensor):
                features = self.encoder.features(inputs)
                features = F.adaptive_avg_pool2d(
                    features, (1, 1)
                ).squeeze(-1).squeeze(-1)
                features = features.permute(1, 0).unsqueeze(0)
                features = self.temporal(features)
                features = features.squeeze(0).permute(1, 0)
                parameters = self.layers(features)
                parameters[..., 50] = F.relu(parameters[..., 50])
                return parameters[..., :50], parameters[..., 50:]

        return _PerceptualEncoder()

    def load(self) -> None:
        """Load the SPECTRE E_flame encoder (and E_expression when present).

        Loads the ResnetEncoder (E_flame) always. When the checkpoint also
        contains ``E_expression`` (full SPECTRE model, e.g. ``spectre_model.tar``),
        the PerceptualEncoder is loaded too and ``encode()`` uses it for the
        final expression + jaw (official SPECTRE behaviour). Checkpoints without
        E_expression (e.g. the DECA ``deca_model.tar``) still work, falling
        back to the E_flame jaw/expression.

        Raises:
            FileNotFoundError: If checkpoint not found.
            RuntimeError: If model loading fails.
        """
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"SPECTRE checkpoint not found: {self._checkpoint_path}"
            )

        logger.info(
            "Loading SPECTRE model: %s (device=%s)",
            self._checkpoint_path, self._device,
        )
        try:
            from src.models.encoders import ResnetEncoder

            checkpoint = torch.load(
                str(self._checkpoint_path),
                map_location=self._device,
                weights_only=False,
            )

            # ---- E_flame ----
            if "E_flame" in checkpoint:
                state_dict = checkpoint["E_flame"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                state_dict = {
                    k.replace("deca.", ""): v
                    for k, v in state_dict.items()
                    if "E_flame" in k
                }
                state_dict = {
                    k.replace("E_flame.", ""): v
                    for k, v in state_dict.items()
                }
            else:
                raise KeyError(
                    "Checkpoint does not contain 'E_flame' or 'state_dict'."
                )

            model = ResnetEncoder(outsize=self._N_PARAM)
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self._device)
            model.eval()
            self._model = model
            logger.info("SPECTRE E_flame loaded (%d keys).", len(state_dict))

            # ---- E_expression (optional; official final jaw/expression) ----
            self._expression_model = None
            expr_sd = checkpoint.get("E_expression")
            if expr_sd is not None:
                try:
                    expr_model = self._build_expression_encoder()
                    expr_model.load_state_dict(expr_sd, strict=True)
                    expr_model = expr_model.to(self._device)
                    expr_model.eval()
                    self._expression_model = expr_model
                    logger.info(
                        "SPECTRE E_expression loaded (%d keys); encode() will "
                        "use it for jaw/expression.",
                        len(expr_sd),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load E_expression (%s); falling back to "
                        "E_flame jaw/expression.",
                        exc,
                    )
                    self._expression_model = None
            else:
                logger.info(
                    "Checkpoint has no E_expression (pure DECA E_flame); "
                    "encode() uses E_flame jaw/expression."
                )
        except Exception as exc:
            logger.error("Failed to load SPECTRE model: %s", exc)
            raise RuntimeError(
                f"Failed to load SPECTRE model from {self._checkpoint_path}"
            ) from exc

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run SPECTRE on a batch of face images.

        Runs E_flame (head pose + raw latent values) and, when the
        checkpoint provides it (full SPECTRE ``spectre_model.tar``),
        E_expression for the official final expression + jaw (temporal
        Conv1d over the batch/time axis). Without E_expression it falls
        back to E_flame raw jaw/expression.

        Args:
            images: Normalized face images [B, 3, 224, 224].

        Returns:
            Dict with:
                expression  [B, 100] — expression latent (E_expression if
                    available else E_flame), 50 dims zero-padded to 100
                head_pose   [B, 3]   — global head pose (E_flame, axis-angle)
                jaw_pose    [B, 3]   — jaw pose (E_expression if available
                    else E_flame, axis-angle)
        """
        if self._model is None:
            raise RuntimeError("SPECTRE model not loaded. Call load() first.")

        images = images.to(self._device)
        with torch.no_grad():
            parameters = self._model(images)

        B = parameters.shape[0]

        # E_flame decomposition (pose at [200:206]: [global(3), jaw(3)]).
        pose = parameters[:, self._POSE_START:self._POSE_END]
        head_pose = pose[:, :3]  # global head pose (axis-angle, rad)
        raw_jaw = pose[:, 3:]    # E_flame raw jaw (axis-angle, rad)
        raw_exp = parameters[:, self._EXPR_START:self._EXPR_END]  # [B, 50]

        # Official SPECTRE: E_expression produces a RESIDUAL over the E_flame
        # estimate — final exp/jaw = E_flame + E_expression (see spectre.py
        # encode() and trainer_spectre.py: "residual from DECA, add it to the
        # initial estimate of jaw and expression"). The temporal Conv1d treats
        # the batch axis as the time axis; we pad two frames at each end
        # (copies of the first/last frame) and drop the boundary outputs so
        # kernel-'same' padding artifacts are removed, mirroring the official
        # demo chunk strategy. Fall back to E_flame raw values when the
        # expression encoder weights are absent (e.g. old DECA checkpoint).
        if self._expression_model is not None:
            with torch.no_grad():
                if B > 4:
                    pad2 = images[:1].expand(2, -1, -1, -1)
                    pad2_last = images[-1:].expand(2, -1, -1, -1)
                    images_t = torch.cat([pad2, images, pad2_last], dim=0)
                    expr50, jaw = self._expression_model(images_t)  # [B+4,50],[B+4,3]
                    expr50 = expr50[2:-2]  # drop boundary -> [B,50]
                    jaw = jaw[2:-2]        # drop boundary -> [B,3]
                else:
                    expr50, jaw = self._expression_model(images)
            exp50 = raw_exp + expr50  # E_expression refines the E_flame exp
            jaw_pose = raw_jaw + jaw  # E_expression refines the E_flame jaw
        else:
            exp50 = raw_exp
            jaw_pose = raw_jaw

        # Zero-pad 50 -> 100 to match pipeline expression_dim (Stage17 layout
        # [0:100]); the last 50 dims stay zero.
        expression = torch.zeros(
            (B, self.PADDED_EXPRESSION_DIM),
            dtype=exp50.dtype,
            device=exp50.device,
        )
        expression[:, : self.EXPRESSION_DIM] = exp50

        return {
            "expression": expression,  # [B, 100]
            "head_pose": head_pose,    # [B, 3] global head (axis-angle)
            "jaw_pose": jaw_pose,      # [B, 3] jaw (axis-angle)
        }

    def extract_parameters(
        self, batch: Dict[str, torch.Tensor], frame_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """Extract numpy float tuples from the batch dict.

        Args:
            batch: Dict from encode().
            frame_indices: Frame indices per sample.

        Returns:
            List of dicts with keys: frame_index, expression, head_pose,
            jaw_pose.
        """
        B = batch["head_pose"].shape[0]
        results: List[Dict[str, Any]] = []
        for i in range(B):
            results.append({
                "frame_index": frame_indices[i],
                "expression": tuple(
                    batch["expression"][i].cpu().numpy().tolist()
                ),
                "head_pose": tuple(
                    batch["head_pose"][i].cpu().numpy().tolist()
                ),
                "jaw_pose": tuple(
                    batch["jaw_pose"][i].cpu().numpy().tolist()
                ),
            })
        return results

    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """Convert numpy image to normalized tensor for SPECTRE.

        Args:
            image: Input image (H, W, 3) in BGR, uint8 [0, 255].
            target_size: Desired output size (H, W).

        Returns:
            Tensor [1, 3, H, W] normalized to [0, 1] (float; matches the
            DECA/SPECTRE training distribution — skimage ``warp()``
            output, NOT [-1, 1]).
        """
        import cv2
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size[::-1])
        tensor = torch.from_numpy(image.astype(np.float32))
        tensor = tensor.permute(2, 0, 1)  # HWC → CHW
        tensor = tensor / 255.0           # [0, 255] → [0, 1] float (matches DECA/SPECTRE training distribution)
        tensor = tensor.unsqueeze(0)        # [C, H, W] → [1, C, H, W]
        return tensor