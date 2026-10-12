"""SCRFD face detection model wrapper.

Per AISDD 3.6, SCRFD is used for coarse face screening in Stage 3
at 1 FPS sampling rate, and for full detection in Stage 5.

This wrapper encapsulates model loading and inference.
Stages MUST NOT interact with the underlying model library directly.

Loading strategy (local-first, no auto-download):
    1. Local ONNX file (scrfd_10g_bnkps.onnx) — primary path.
    2. Local insightface buffalo_sc model files — fallback, only if pre-downloaded.

This wrapper NEVER triggers automatic model downloads.
All models must be pre-downloaded to the configured model_path.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SCRFDWrapper:
    """SCRFD face detector wrapper.

    Encapsulates SCRFD model loading, device management, and inference.
    Provides a unified detect() interface for Stage 3 and Stage 5.

    Usage:
        wrapper = SCRFDWrapper(model_path=Path("models/SCRFD"), device="cuda")
        wrapper.load()
        results = wrapper.detect(image)  # image: np.ndarray (H, W, C) BGR
    """

    # SCRFD 10g input size
    INPUT_SIZE = (640, 640)

    # Detection thresholds
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    # Pre-filter score applied BEFORE the O(n^2) Python NMS loop.
    # SCRFD emits one box per anchor (~16800 for 640x640 input) and nearly
    # all carry scores ~0. This pre-filter drops those pure-noise boxes so
    # NMS only processes the few plausible candidates. It is intentionally
    # far below CONFIDENCE_THRESHOLD (0.5) so the final result set is
    # unchanged (16800-box NMS cost ~1.4-1.6 s/frame before this filter).
    PRE_NMS_SCORE_FILTER = 0.02

    # Strides used by SCRFD for multi-scale detection
    STRIDES = (8, 16, 32)

    # Number of anchors per grid cell for SCRFD 10g
    ANCHORS_PER_CELL = 2

    # Number of landmarks (5-point: eyes, nose, mouth corners)
    NUM_LANDMARKS = 5

    # Feature map sizes per stride for 640x640 input
    FM_SIZES = {
        8: (80, 80),
        16: (40, 40),
        32: (20, 20),
    }

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        execution_provider: str = "auto",
        confidence_threshold: Optional[float] = None,
    ) -> None:
        """Initialize SCRFD wrapper.

        Args:
            model_path: Path to SCRFD model weights directory
                        (containing scrfd_10g_bnkps.onnx).
            device: Target device for inference ("cpu", "cuda", "cuda:0").
            execution_provider: ONNX Runtime execution provider preference:
                                "auto" / "ppu" / "cuda" / "cpu".
            confidence_threshold: Detection confidence floor for this instance.
                Detections scoring below this are dropped in detect().  None
                falls back to the class-level CONFIDENCE_THRESHOLD (0.5).
                Stage 3 (the coarse "is there a face" gate) sets a lower floor
                (e.g. 0.35) so occluded / partially-cropped real faces whose
                SCRFD score lands in 0.35-0.5 are not silently discarded before
                the ratio / confidence threshold checks run.
        """
        self._model_path: Path = Path(model_path)
        self._device: str = device
        self._execution_provider: str = execution_provider
        self._model: Any = None
        self._input_name: str = ""
        self._output_names: List[str] = []
        self._use_insightface: bool = False
        self.confidence_threshold: float = (
            confidence_threshold
            if confidence_threshold is not None
            else self.CONFIDENCE_THRESHOLD
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        """Return the configured model path."""
        return self._model_path

    @property
    def device(self) -> str:
        """Return the target device."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded successfully."""
        return self._model is not None

    @property
    def active_providers(self) -> List[str]:
        """Return the ONNX Runtime providers actually in use (empty if not loaded)."""
        if self._model is None:
            return []
        try:
            return list(self._model.get_providers())
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the SCRFD model.

        Loading priority (local-first, no auto-download):
            1. Local ONNX file (scrfd_10g_bnkps.onnx) — primary path.
            2. Local insightface model files (fallback, only if already present).

        This wrapper NEVER triggers automatic model downloads.
        All models must be pre-downloaded to the configured model_path.

        Raises:
            FileNotFoundError: If no local SCRFD model (ONNX or insightface) is found.
            RuntimeError: If model loading fails for any reason.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"SCRFD model path does not exist: {self._model_path.resolve()}"
            )

        # --- Path 1: Local ONNX file (primary, no network) ---
        onnx_path = self._find_onnx_file()
        if onnx_path.exists():
            from face_pipeline_v2.runtime.onnx_runtime import create_onnx_session

            self._model = create_onnx_session(
                onnx_path, execution_provider=self._execution_provider
            )
            self._input_name = self._model.get_inputs()[0].name
            self._output_names = [o.name for o in self._model.get_outputs()]
            self._use_insightface = False
            # Surface the provider actually in use (important for PPU verification).
            logger.info(
                "SCRFD loaded '%s' with active providers: %s",
                onnx_path.name, self._model.get_providers(),
            )
            return

        # --- Path 2: Local insightface model (fallback, no download) ---
        try:
            from insightface.app import FaceAnalysis
            import os

            # Only use insightface if model files are already present locally.
            # insightface downloads to: <root>/models/<name>/
            insightface_root = str(self._model_path.parent)
            insightface_model_dir = os.path.join(
                insightface_root, "models", "buffalo_sc"
            )
            if not os.path.isdir(insightface_model_dir):
                raise FileNotFoundError(
                    f"SCRFD ONNX model not found at {onnx_path}, and "
                    f"insightface buffalo_sc not found at {insightface_model_dir}. "
                    f"Please download the SCRFD ONNX model to {self._model_path} "
                    f"or pre-download buffalo_sc to {insightface_model_dir}."
                )

            self._model = FaceAnalysis(
                name="buffalo_sc",
                root=insightface_root,
                allowed_modules=["detection"],
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "cuda" in self._device
                    else ["CPUExecutionProvider"]
                ),
            )
            self._model.prepare(
                ctx_id=0 if "cuda" in self._device else -1,
                det_size=self.INPUT_SIZE,
            )
            self._use_insightface = True
            return
        except FileNotFoundError:
            raise
        except ImportError:
            pass
        except Exception:
            pass

        # --- No model found ---
        raise FileNotFoundError(
            f"SCRFD ONNX model not found: {onnx_path}. "
            f"Please ensure scrfd_10g_bnkps.onnx exists in {self._model_path}."
        )

    def _find_onnx_file(self) -> Path:
        """Locate the SCRFD ONNX file in the model directory.

        Searches for known filenames: scrfd_10g_bnkps.onnx, scrfd.onnx, *.onnx.
        """
        candidates = [
            self._model_path / "scrfd_10g_bnkps.onnx",
            self._model_path / "scrfd.onnx",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        onnx_files = list(self._model_path.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]

        return self._model_path / "scrfd_10g_bnkps.onnx"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in a single BGR image.

        Args:
            image: BGR image as numpy array of shape (H, W, C).

        Returns:
            List of detection results, each containing:
                - bbox: Tuple[float, float, float, float]  (x1, y1, x2, y2)
                - confidence: float
                - landmarks: Optional[Tuple[float, ...]]  5-point (10 values)

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "SCRFD model is not loaded. Call load() first."
            )

        if self._use_insightface:
            return self._detect_insightface(image)
        else:
            return self._detect_onnx(image)

    def _detect_insightface(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using insightface FaceAnalysis."""
        faces = self._model.get(image)
        results: List[Dict[str, Any]] = []
        for face in faces:
            detection: Dict[str, Any] = {
                "bbox": tuple(float(v) for v in face.bbox),
                "confidence": float(getattr(face, "det_score", 0.0)),
                "landmarks": None,
            }
            if hasattr(face, "kps") and face.kps is not None:
                detection["landmarks"] = tuple(
                    float(v) for v in face.kps.flatten()
                )
            results.append(detection)
        return results

    # ------------------------------------------------------------------
    # ONNX runtime path
    # ------------------------------------------------------------------

    def _detect_onnx(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using ONNX runtime with manual pre/post-processing."""
        img_h, img_w = image.shape[:2]
        input_tensor, scale_ratio = self._preprocess(image)

        outputs = self._model.run(
            self._output_names, {self._input_name: input_tensor}
        )

        scores_list, boxes_list, kps_list = self._parse_outputs(outputs)

        det_bboxes, det_scores, det_kps = self._decode_detections(
            scores_list, boxes_list, kps_list, scale_ratio, img_h, img_w
        )

        # Performance: pre-filter near-zero-confidence boxes before the
        # O(n^2) Python NMS loop. Keeps every box that could plausibly
        # survive NMS (final confidence threshold applied below) while
        # removing the ~16.7k pure-noise anchors.
        mask = det_scores > self.PRE_NMS_SCORE_FILTER
        det_bboxes = det_bboxes[mask]
        det_scores = det_scores[mask]
        if det_kps is not None:
            det_kps = det_kps[mask]

        if len(det_bboxes) > 0:
            keep = self._nms(det_bboxes, det_scores, self.NMS_THRESHOLD)
            det_bboxes = det_bboxes[keep]
            det_scores = det_scores[keep]
            det_kps = det_kps[keep] if det_kps is not None else None

        results: List[Dict[str, Any]] = []
        for i in range(len(det_bboxes)):
            if det_scores[i] < self.confidence_threshold:
                continue
            result: Dict[str, Any] = {
                "bbox": tuple(float(v) for v in det_bboxes[i]),
                "confidence": float(det_scores[i]),
                "landmarks": None,
            }
            if det_kps is not None:
                result["landmarks"] = tuple(
                    float(v) for v in det_kps[i].flatten()
                )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Preprocess a BGR image for SCRFD ONNX inference.

        Returns:
            (input_tensor, scale_ratio):
                input_tensor: (1, 3, 640, 640) float32.
                scale_ratio: (scale_w, scale_h).
        """
        img_h, img_w = image.shape[:2]
        target_h, target_w = self.INPUT_SIZE

        scale_h = target_h / img_h
        scale_w = target_w / img_w

        resized = cv2.resize(image, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 128.0
        input_tensor = np.expand_dims(
            np.transpose(blob, (2, 0, 1)), axis=0
        ).astype(np.float32)

        return input_tensor, (scale_w, scale_h)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_outputs(
        self, outputs: List[np.ndarray]
    ) -> Tuple[
        List[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]]
    ]:
        """Parse ONNX outputs into stride-level scores, boxes, landmarks.

        SCRFD ONNX outputs are 2D arrays of shape (N_i, C):
            - scores: (N_i, 1)
            - boxes:  (N_i, 4)
            - kps:    (N_i, 10) — optional

        We classify by the last dimension (C), not by name,
        so the wrapper works with any SCRFD ONNX export.
        """
        num_strides = len(self.STRIDES)

        score_outputs: List[np.ndarray] = []
        bbox_outputs: List[np.ndarray] = []
        kps_outputs: List[np.ndarray] = []

        for o in outputs:
            arr: np.ndarray = o  # (N, C)
            if arr.ndim >= 2 and arr.shape[-1] == 1:
                score_outputs.append(arr)
            elif arr.ndim >= 2 and arr.shape[-1] == 4:
                bbox_outputs.append(arr)
            elif arr.ndim >= 2 and arr.shape[-1] == 10:
                kps_outputs.append(arr)

        # Sort by N descending (largest stride first = most anchors)
        score_outputs.sort(key=lambda a: a.shape[0], reverse=True)
        bbox_outputs.sort(key=lambda a: a.shape[0], reverse=True)
        kps_outputs.sort(key=lambda a: a.shape[0], reverse=True)

        scores_list: List[np.ndarray] = []
        boxes_list: List[np.ndarray] = []
        kps_list: List[Optional[np.ndarray]] = []

        for i in range(num_strides):
            scores_list.append(
                score_outputs[i] if i < len(score_outputs)
                else np.zeros((0, 1), dtype=np.float32)
            )
            boxes_list.append(
                bbox_outputs[i] if i < len(bbox_outputs)
                else np.zeros((0, 4), dtype=np.float32)
            )
            kps_list.append(
                kps_outputs[i] if i < len(kps_outputs) else None
            )

        return scores_list, boxes_list, kps_list

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_detections(
        self,
        scores_list: List[np.ndarray],
        boxes_list: List[np.ndarray],
        kps_list: List[Optional[np.ndarray]],
        scale_ratio: Tuple[float, float],
        img_h: int,
        img_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Decode stride-relative outputs into image-coordinate detections.

        SCRFD 10g uses 2 anchors per feature map cell.
        Bbox uses 4-DISTANCE decode (dx1, dy1, dx2, dy2) —
        distances from grid cell centre to the four edges of the bbox.
        """
        all_bboxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_kps: List[np.ndarray] = []

        scale_w, scale_h = scale_ratio

        for stride_idx, stride in enumerate(self.STRIDES):
            scores = scores_list[stride_idx]  # (N, 1)
            boxes = boxes_list[stride_idx]    # (N, 4) — (dx1, dy1, dx2, dy2)
            kps = kps_list[stride_idx]        # (N, 10) or None

            if scores.shape[0] == 0:
                continue

            fm_h, fm_w = self.FM_SIZES[stride]
            cells_per_map = fm_h * fm_w
            num_anchors = scores.shape[0]

            # Each grid cell has ANCHORS_PER_CELL (=2) anchors
            anchor_idx = np.arange(num_anchors, dtype=np.float32)
            cell_idx = (anchor_idx / float(self.ANCHORS_PER_CELL)).astype(np.int32)
            cell_idx = np.clip(cell_idx, 0, cells_per_map - 1)

            grid_y = (cell_idx // fm_w).astype(np.float32)
            grid_x = (cell_idx % fm_w).astype(np.float32)

            # ── SCRFD 4-distance bbox decode ──
            #   dx1 = distance from grid_x to left edge
            #   dy1 = distance from grid_y to top edge
            #   dx2 = distance from grid_x to right edge
            #   dy2 = distance from grid_y to bottom edge
            x1 = (grid_x - boxes[:, 0]) * stride
            y1 = (grid_y - boxes[:, 1]) * stride
            x2 = (grid_x + boxes[:, 2]) * stride
            y2 = (grid_y + boxes[:, 3]) * stride

            # Rescale from 640×640 → original image
            x1 /= scale_w
            y1 /= scale_h
            x2 /= scale_w
            y2 /= scale_h

            x1 = np.clip(x1, 0, img_w)
            y1 = np.clip(y1, 0, img_h)
            x2 = np.clip(x2, 0, img_w)
            y2 = np.clip(y2, 0, img_h)

            xyxy = np.stack([x1, y1, x2, y2], axis=1)
            all_bboxes.append(xyxy)
            all_scores.append(scores.flatten())

            # ── 5-point landmark decode (offset from grid centre) ──
            if kps is not None:
                decoded_kps = np.zeros_like(kps, dtype=np.float32)
                for p in range(self.NUM_LANDMARKS):
                    decoded_kps[:, p * 2] = (
                        (grid_x + kps[:, p * 2]) * stride
                    ) / scale_w
                    decoded_kps[:, p * 2 + 1] = (
                        (grid_y + kps[:, p * 2 + 1]) * stride
                    ) / scale_h
                all_kps.append(decoded_kps)

        if not all_bboxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                None,
            )

        merged_bboxes = np.concatenate(all_bboxes, axis=0)
        merged_scores = np.concatenate(all_scores, axis=0)
        merged_kps = np.concatenate(all_kps, axis=0) if all_kps else None

        return merged_bboxes, merged_scores, merged_kps

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    @staticmethod
    def _nms(
        boxes: np.ndarray, scores: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Non-maximum suppression on xyxy boxes."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            remaining = np.where(iou <= threshold)[0]
            order = order[remaining + 1]

        return np.array(keep, dtype=np.int64)
