"""Lightweight ReID utilities for CPU-friendly appearance matching."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class LightweightReIDService:
    """Appearance embedding based on HSV color histogram."""

    def __init__(self, bins: tuple[int, int, int] = (16, 16, 16)):
        self.bins = bins

    @staticmethod
    def _clip_box(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        left = max(0, min(int(x1), width - 1))
        top = max(0, min(int(y1), height - 1))
        right = max(left + 1, min(int(x2), width))
        bottom = max(top + 1, min(int(y2), height))
        return left, top, right, bottom

    def extract_embedding(
        self,
        frame: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Optional[np.ndarray]:
        """Extract normalized HSV histogram embedding from bbox crop."""
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        left, top, right, bottom = self._clip_box(x1, y1, x2, y2, w, h)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256])
        if hist is None:
            return None

        hist = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm <= 1e-12:
            return None
        return hist / norm

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity for normalized vectors."""
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            return 0.0
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    @staticmethod
    def update_running_embedding(
        current: Optional[np.ndarray],
        new_value: np.ndarray,
        momentum: float,
    ) -> np.ndarray:
        """EMA update for track appearance embedding."""
        if current is None:
            return new_value

        alpha = float(np.clip(momentum, 0.0, 1.0))
        updated = (1.0 - alpha) * current + alpha * new_value
        norm = np.linalg.norm(updated)
        if norm <= 1e-12:
            return new_value
        return updated / norm
