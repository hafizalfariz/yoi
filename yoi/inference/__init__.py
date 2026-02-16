"""Inference package exports."""

from yoi.inference.yolo import Detection, FrameInference, YOLOInferencer

__all__ = ["YOLOInferencer", "Detection", "FrameInference"]
