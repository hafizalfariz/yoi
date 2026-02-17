"""YOLO Inference Engine

Wrapper for Ultralytics YOLO with optional metadata.yaml support.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from yoi.utils.logger import logger_service

try:
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Detection:
    """Single object detection"""

    def __init__(self, box: List[float], confidence: float, class_id: int, class_name: str):
        """
        Args:
            box: [x1, y1, x2, y2] in pixel coordinates
            confidence: Confidence score 0-1
            class_id: Class ID
            class_name: Class name
        """
        self.x1, self.y1, self.x2, self.y2 = box
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.centroid_x = (self.x1 + self.x2) / 2
        self.centroid_y = (self.y1 + self.y2) / 2

    @property
    def center(self) -> Tuple[float, float]:
        """Get centroid coordinates"""
        return (self.centroid_x, self.centroid_y)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> Dict:
        """Convert to dict for output."""
        return {
            "box": [self.x1, self.y1, self.x2, self.y2],
            "center": self.center,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class FrameInference:
    """Inference results for a single frame."""

    def __init__(self, frame_idx: int, detections: List[Detection]):
        self.frame_idx = frame_idx
        self.detections = detections
        self.num_detections = len(detections)

    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """Filter detections by class"""
        return [d for d in self.detections if d.class_name == class_name]

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "frame_idx": self.frame_idx,
            "num_detections": self.num_detections,
            "detections": [d.to_dict() for d in self.detections],
        }


class YOLOInferencer:
    """YOLO object detection inferencer"""

    def __init__(
        self,
        model_name: str = "yolov8n",
        device: str = "cpu",
        conf: float = 0.5,
        iou: float = 0.7,
        classes: Optional[List[str]] = None,
    ):
        """
        Initialize YOLO inferencer

        Args:
            model_name: Model name (e.g., yolov8n, yolov8s, yolov8m)
            device: Device (cpu, cuda, mps)
            conf: Confidence threshold
            iou: IOU threshold for NMS
            classes: List of class names for filtering
        """
        self.logger = logger_service.get_inference_logger()

        if not HAS_ULTRALYTICS:
            raise ImportError("Ultralytics is not installed. Install with: pip install ultralytics")

        self.model_name = model_name
        normalized_device = str(device).strip().lower()
        if normalized_device == "gpu":
            normalized_device = "cuda"
        self.device = normalized_device
        target_device_env = os.getenv("YOI_TARGET_DEVICE", "").strip().lower()
        if target_device_env == "gpu":
            target_device_env = "cuda"
        runtime_profile_env = os.getenv("YOI_RUNTIME_PROFILE", "").strip().lower()
        runtime_target = target_device_env
        if not runtime_target:
            if runtime_profile_env == "gpu":
                runtime_target = "cuda"
            elif runtime_profile_env == "cpu":
                runtime_target = "cpu"
        if runtime_target in {"cpu", "cuda", "mps"} and runtime_target != self.device:
            self.logger.warning(
                "Config device mismatch: config=%s, runtime=%s (env)",
                self.device,
                runtime_target,
            )
            raise RuntimeError(
                "Config model.device does not match runtime target device. "
                f"config={self.device}, runtime={runtime_target}"
            )
        strict_device_env = os.getenv("YOI_STRICT_DEVICE", "0").strip().lower()
        self.strict_device = strict_device_env in {"1", "true", "yes", "on"}
        self.conf_threshold = conf
        self.iou_threshold = iou
        self._fallback_to_cpu_attempted = False
        self._model_source_ref = model_name
        self._is_onnx = False
        self._warned_onnx_gpu_fallback = False
        raw_imgsz = os.getenv("YOI_YOLO_IMGSZ", "").strip()
        self._imgsz: Optional[int] = None
        if raw_imgsz:
            try:
                parsed = int(raw_imgsz)
                if parsed > 0:
                    self._imgsz = parsed
            except ValueError:
                self.logger.warning(
                    "Invalid YOI_YOLO_IMGSZ value '%s', using model default",
                    raw_imgsz,
                )
        # Metadata from metadata.yaml if available.
        self.metadata: Dict = {}
        # Target classes are determined after model/metadata load.
        self.target_classes: List[str] = []

        if self.device == "cuda":
            cuda_available = bool(HAS_TORCH and torch.cuda.is_available())
            if not cuda_available:
                if self.strict_device:
                    raise RuntimeError(
                        "Strict GPU mode is enabled but CUDA is unavailable in this runtime"
                    )
                self.logger.warning(
                    "GPU is unavailable in this runtime; falling back to CPU before model selection"
                )
                self.device = "cpu"

        if self.device in {"cuda", "mps"} and model_name.lower().endswith(".onnx"):
            raise RuntimeError(
                "GPU runtime requires a GPU-native model (.pt/.pth). "
                f"ONNX model is not allowed for device '{self.device}': {model_name}"
            )

        try:
            # Try to find local model file first
            model_path = self._find_local_model(model_name)
            if model_path:
                self.logger.info(f"Found local model: {model_path}")
                self._model_source_ref = str(model_path)
                metadata_path = model_path.parent / "metadata.yaml"
                if metadata_path.exists():
                    try:
                        self.metadata = (
                            yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                        )
                        self.logger.info(f"Loaded model metadata from {metadata_path}")
                    except Exception as me:
                        self.logger.warning(f"Failed to read metadata.yaml: {me}")

                self.model = YOLO(str(model_path), task="detect")
                is_onnx = str(model_path).lower().endswith(".onnx")
            else:
                # Fallback to loading from name (online)
                self.logger.info(f"Loading model by name: {model_name}")
                self.model = YOLO(model_name, task="detect")
                is_onnx = model_name.lower().endswith(".onnx")
            self._is_onnx = is_onnx

            if self._is_onnx and self.device != "cpu":
                if self.strict_device:
                    raise RuntimeError(
                        "Strict GPU mode is enabled but selected model is ONNX. "
                        "Use a GPU-native model (.pt) for GPU profile, or run CPU profile for ONNX."
                    )
                self.device = "cpu"
                self._warned_onnx_gpu_fallback = True
                self.logger.warning(
                    "ONNX model requested with GPU device; falling back to CPU for stable inference"
                )

            # Only move to device for PyTorch models, not ONNX
            if not is_onnx:
                try:
                    self.model.to(self.device)
                except Exception as device_error:
                    if self.strict_device:
                        raise
                    if self._should_try_cpu_fallback(str(device_error)):
                        fallback_ok = self._fallback_to_cpu_model()
                        if not fallback_ok:
                            raise
                    else:
                        raise
            self.logger.info(f"YOLO Model loaded: {model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

        # Read class names from model/metadata.
        self.class_names = self.model.names
        # Fallback to metadata mapping when model names are empty.
        if (not self.class_names) and self.metadata.get("names"):
            self.class_names = self.metadata["names"]

        # Normalize to int->str mapping when needed.
        if isinstance(self.class_names, list):
            self.class_names = {i: name for i, name in enumerate(self.class_names)}

        self.logger.info(f"Model classes: {self.class_names}")

        # Determine target classes.
        if classes:
            self.target_classes = classes
        else:
            if isinstance(self.class_names, dict):
                values = list(self.class_names.values())
                if len(values) == 1:
                    self.target_classes = [values[0]]
                else:
                    self.target_classes = values
            else:
                # Fallback when no structured class metadata is available.
                self.target_classes = ["person"]

    def _find_local_model(self, model_name: str) -> Optional[Path]:
        """
        Try to find local model file in models directory
        Structure: models/{model_name}/{version}/{weight_file}

        Args:
            model_name: Name of the model or path to model file

        Returns:
            Path to model file if found, None otherwise
        """
        # If already a path, check if it exists
        if model_name.endswith((".onnx", ".pt", ".pth")):
            model_path = Path(model_name)
            if model_path.exists():
                return model_path

        # Try to construct path from model name
        models_dir = Path("models")
        if not models_dir.exists():
            return None

        extension_priority = [".onnx", ".pt", ".pth"]
        if self.device in {"cuda", "mps"}:
            extension_priority = [".pt", ".pth"]

        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name == model_name]

        if not model_dirs:
            return None

        model_dir = model_dirs[0]

        onnx_present = bool(list(model_dir.glob("*.onnx")))

        # First, support direct files under model directory:
        # models/<model_name>/best.onnx
        for ext in extension_priority:
            for name in ["best", "model", "weights"]:
                model_file = model_dir / f"{name}{ext}"
                if model_file.exists():
                    return model_file

        direct_model_files = []
        for ext in extension_priority:
            direct_model_files.extend(model_dir.glob(f"*{ext}"))
        if direct_model_files:
            return sorted(direct_model_files)[0]

        if self.device in {"cuda", "mps"}:
            for version_dir in sorted(model_dir.iterdir(), reverse=True):
                if not version_dir.is_dir():
                    continue
                if list(version_dir.glob("*.onnx")):
                    onnx_present = True
                    break

        # Look for version directories (e.g., "1", "2", etc.)
        for version_dir in sorted(model_dir.iterdir(), reverse=True):
            if not version_dir.is_dir():
                continue

            # Look for model files
            for ext in extension_priority:
                # Try common names: best.ext, model.ext, weights.ext
                for name in ["best", "model", "weights"]:
                    model_file = version_dir / f"{name}{ext}"
                    if model_file.exists():
                        return model_file

            # If version dir has model files directly
            model_files = []
            for ext in extension_priority:
                model_files.extend(version_dir.glob(f"*{ext}"))
            if model_files:
                return sorted(model_files)[0]

        if self.device in {"cuda", "mps"} and onnx_present:
            raise RuntimeError(
                "GPU runtime requires local model weights in .pt/.pth format, "
                f"but only ONNX weights were found for model '{model_name}'"
            )

        return None

    def infer(self, frame: np.ndarray) -> FrameInference:
        """
        Run inference on a single frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            FrameInference object
        """
        try:
            infer_kwargs = {
                "conf": self.conf_threshold,
                "iou": self.iou_threshold,
                "verbose": False,
                "device": self.device,
            }
            if self._imgsz is not None:
                infer_kwargs["imgsz"] = self._imgsz
            results = self.model(frame, **infer_kwargs)

            detections = []
            for result in results:
                if result.boxes is not None:
                    for box_data in result.boxes:
                        # Extract coordinates
                        x1, y1, x2, y2 = box_data.xyxy[0].tolist()
                        conf = float(box_data.conf[0])
                        class_id = int(box_data.cls[0])
                        class_name = self.class_names[class_id]

                        # Filter by target classes
                        if class_name not in self.target_classes:
                            continue

                        detection = Detection(
                            box=[x1, y1, x2, y2],
                            confidence=conf,
                            class_id=class_id,
                            class_name=class_name,
                        )
                        detections.append(detection)

            return FrameInference(frame_idx=0, detections=detections)

        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            if self.strict_device:
                raise
            if self._should_try_cpu_fallback(str(e)):
                fallback_ok = self._fallback_to_cpu_model()
                if fallback_ok:
                    self.logger.warning("Retrying inference once after GPU fallback to CPU")
                    return self.infer(frame)
            return FrameInference(frame_idx=0, detections=[])

    def _should_try_cpu_fallback(self, error_message: str) -> bool:
        """Return True when runtime error indicates GPU binding/provider issue."""
        if self.strict_device:
            return False
        if self.device == "cpu" or self._fallback_to_cpu_attempted:
            return False

        message = (error_message or "").lower()
        fallback_markers = (
            "no data transfer registered",
            "error when binding input",
            "cuda",
            "tensorrt",
            "execution provider",
        )
        return any(marker in message for marker in fallback_markers)

    def _fallback_to_cpu_model(self) -> bool:
        """Reload YOLO model on CPU and continue runtime if possible."""
        self._fallback_to_cpu_attempted = True
        try:
            self.logger.warning(
                "GPU inference failed, reloading model on CPU for graceful fallback"
            )
            self.model = YOLO(self._model_source_ref, task="detect")
            if not self._is_onnx:
                self.model.to("cpu")
            self.device = "cpu"
            return True
        except Exception as fallback_error:
            self.logger.error(f"CPU fallback failed: {fallback_error}")
            return False

    def infer_batch(self, frames: List[np.ndarray]) -> List[FrameInference]:
        """
        Run inference pada batch frames

        Args:
            frames: List of frames

        Returns:
            List of FrameInference objects
        """
        results = []
        for idx, frame in enumerate(frames):
            result = self.infer(frame)
            result.frame_idx = idx
            results.append(result)
        return results
