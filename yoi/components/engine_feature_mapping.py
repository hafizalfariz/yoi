"""Feature-processing helpers for VisionEngine."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from yoi.features.base import Detection as FeatureDetection


def build_feature_detections(
    tracker,
    detections,
    tracked_objects: Dict[int, Tuple[float, float, str]],
    frame_shape,
):
    """Build normalized feature detections and track-to-bbox map."""
    h, w = frame_shape[:2]
    track_bbox_map: Dict[int, Any] = {}
    feature_detections = []

    indexed_detections = list(enumerate(detections))
    used_detection_indices = set()
    current_tracker_frame = max(0, tracker.frame_idx - 1)

    for track_id, (track_x, track_y, class_name) in tracked_objects.items():
        track_obj = tracker.get_track(track_id)
        if not track_obj or track_obj.last_frame_idx != current_tracker_frame:
            continue

        best_det = None
        best_det_idx = None
        min_dist = float("inf")

        for det_idx, det in indexed_detections:
            if det_idx in used_detection_indices:
                continue
            if det.class_name != class_name:
                continue

            dist = np.sqrt((det.centroid_x - track_x) ** 2 + (det.centroid_y - track_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_det = det
                best_det_idx = det_idx

        if best_det is None or best_det_idx is None:
            continue

        used_detection_indices.add(best_det_idx)

        norm_bbox = [
            best_det.x1 / w,
            best_det.y1 / h,
            best_det.x2 / w,
            best_det.y2 / h,
        ]
        norm_centroid = (track_x / w, track_y / h)

        feature_det = FeatureDetection(
            track_id=track_id,
            class_id=best_det.class_id,
            class_name=class_name,
            confidence=best_det.confidence,
            bbox=norm_bbox,
            centroid=norm_centroid,
        )
        feature_detections.append(feature_det)
        track_bbox_map[track_id] = best_det

    return feature_detections, track_bbox_map
