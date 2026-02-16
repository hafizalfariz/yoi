"""Frame annotation utilities for detections, tracks, and overlays."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from yoi.utils.logger import logger_service


class VideoAnnotator:
    """Annotate frames with detection, tracking, and analytics info."""

    FONT = cv2.FONT_HERSHEY_DUPLEX
    COLOR_DEFAULT_BBOX = (255, 120, 0)
    COLOR_REGION_OUTSIDE = (140, 140, 140)
    COLOR_REGION_INSIDE = (255, 120, 0)
    COLOR_DWELL_OUTSIDE = (140, 140, 140)
    COLOR_DWELL_INSIDE = (255, 0, 0)
    COLOR_DWELL_ALERT = (0, 0, 255)
    COLOR_IN = (0, 220, 0)
    COLOR_OUT = (0, 0, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_SHADOW = (0, 0, 0)
    COLOR_PANEL = (20, 20, 20)

    def __init__(self):
        self.logger = logger_service.get_output_logger()
        self._smoothed_bboxes: Dict[int, Tuple[float, float, float, float]] = {}
        self._bbox_smoothing_alpha: float = 0.45
        self._enable_bbox_smoothing: bool = os.getenv(
            "YOI_BBOX_SMOOTHING", "0"
        ).strip().lower() in {"1", "true", "on", "yes"}

    def _draw_text_with_shadow(
        self,
        frame: np.ndarray,
        text: str,
        org: Tuple[int, int],
        color: Tuple[int, int, int],
        scale: float = 0.55,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            frame,
            text,
            (org[0] + 1, org[1] + 1),
            self.FONT,
            scale,
            self.COLOR_SHADOW,
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            org,
            self.FONT,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _status_from_alert_type(alert_type: str) -> str:
        alert_type = (alert_type or "").lower()
        if alert_type.endswith("_in"):
            return "in"
        if alert_type.endswith("_out"):
            return "out"
        return "default"

    def _bbox_color(self, status: str) -> Tuple[int, int, int]:
        if status == "dwell_alert":
            return self.COLOR_DWELL_ALERT
        if status == "dwell_inside":
            return self.COLOR_DWELL_INSIDE
        if status == "dwell_outside":
            return self.COLOR_DWELL_OUTSIDE
        if status == "inside":
            return self.COLOR_REGION_INSIDE
        if status == "outside":
            return self.COLOR_REGION_OUTSIDE
        if status == "in":
            return self.COLOR_IN
        if status == "out":
            return self.COLOR_OUT
        return self.COLOR_DEFAULT_BBOX

    def _draw_panel(
        self,
        frame: np.ndarray,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        alpha: float = 0.50,
        shadow_alpha: float = 0.22,
    ) -> None:
        shadow_offset = 3
        shadow_overlay = frame.copy()
        cv2.rectangle(
            shadow_overlay,
            (top_left[0] + shadow_offset, top_left[1] + shadow_offset),
            (bottom_right[0] + shadow_offset, bottom_right[1] + shadow_offset),
            self.COLOR_SHADOW,
            -1,
        )
        cv2.addWeighted(shadow_overlay, shadow_alpha, frame, 1 - shadow_alpha, 0, frame)

        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_badge(
        self,
        frame: np.ndarray,
        text: str,
        top_left: Tuple[int, int],
        color: Tuple[int, int, int],
        scale: float = 0.52,
    ) -> None:
        (text_w, text_h), baseline = cv2.getTextSize(text, self.FONT, scale, 1)
        x1, y1 = top_left
        x2 = x1 + text_w + 12
        y2 = y1 + text_h + baseline + 10
        self._draw_panel(frame, (x1, y1), (x2, y2), alpha=0.46, shadow_alpha=0.18)
        self._draw_text_with_shadow(frame, text, (x1 + 6, y2 - 6), color, scale, 1)

    @staticmethod
    def _coords_to_pixels(coords: List[Any], width: int, height: int) -> List[Tuple[int, int]]:
        """Convert normalized coordinate objects/dicts into pixel tuples."""
        if not coords:
            return []
        if isinstance(coords[0], dict):
            return [(int(c["x"] * width), int(c["y"] * height)) for c in coords]
        return [(int(c.x * width), int(c.y * height)) for c in coords]

    def draw_boxes(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """Draw detection bounding boxes and labels."""
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            color = self.COLOR_DEFAULT_BBOX

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name.upper()} {det.confidence:.2f}"
            badge_y = max(4, y1 - 30)
            self._draw_badge(frame, label, (x1, badge_y), self.COLOR_TEXT, scale=0.50)

        return frame

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracked_objects: Dict,
        track_bbox_map: Optional[Dict[int, Any]] = None,
        track_states: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """Draw tracking info with optional bbox-linked ID labels."""
        active_ids = set(tracked_objects.keys())
        self._smoothed_bboxes = {
            tid: bbox for tid, bbox in self._smoothed_bboxes.items() if tid in active_ids
        }

        for track_id, (x, y, _) in tracked_objects.items():
            if track_bbox_map is not None and track_id not in track_bbox_map:
                continue

            status = "default"
            if track_states and track_id in track_states:
                status = track_states[track_id]

            color = self._bbox_color(status)
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 5, color, -1)

            if track_bbox_map is not None and track_id in track_bbox_map:
                det = track_bbox_map[track_id]
                try:
                    current_bbox = (
                        float(det.x1),
                        float(det.y1),
                        float(det.x2),
                        float(det.y2),
                    )
                    if self._enable_bbox_smoothing:
                        previous_bbox = self._smoothed_bboxes.get(track_id)
                        if previous_bbox is None:
                            smoothed_bbox = current_bbox
                        else:
                            alpha = self._bbox_smoothing_alpha
                            smoothed_bbox = (
                                alpha * current_bbox[0] + (1.0 - alpha) * previous_bbox[0],
                                alpha * current_bbox[1] + (1.0 - alpha) * previous_bbox[1],
                                alpha * current_bbox[2] + (1.0 - alpha) * previous_bbox[2],
                                alpha * current_bbox[3] + (1.0 - alpha) * previous_bbox[3],
                            )
                        self._smoothed_bboxes[track_id] = smoothed_bbox
                    else:
                        self._smoothed_bboxes[track_id] = current_bbox
                        smoothed_bbox = current_bbox

                    x1 = int(smoothed_bbox[0])
                    y1 = int(smoothed_bbox[1])
                    x2 = int(smoothed_bbox[2])
                    y2 = int(smoothed_bbox[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    tx = max(0, x2 - 62)
                    ty = max(10, y1 - 5)

                    state_text = "PERSON"
                    if status == "in":
                        state_text = "PERSON IN"
                    elif status == "out":
                        state_text = "PERSON OUT"
                    self._draw_badge(
                        frame,
                        state_text,
                        (x1 + 2, max(4, y1 - 32)),
                        color,
                        scale=0.50,
                    )
                except AttributeError:
                    tx = x + 10
                    ty = y
            else:
                tx = x + 10
                ty = y

            self._draw_badge(
                frame,
                f"ID {track_id}",
                (tx, max(4, ty - 20)),
                color,
                scale=0.54,
            )

        return frame

    def draw_analytics(self, frame: np.ndarray, analytics_data: Dict) -> np.ndarray:
        """Draw analytics summary on frame."""
        class_counts = analytics_data.get("object_count_by_class", {})
        if not isinstance(class_counts, dict):
            class_counts = {}

        lines = [f"Objects: {analytics_data.get('object_count', 0)}"]
        for class_name, count in class_counts.items():
            lines.append(f"{class_name}: {count}")

        max_text_width = 0
        for line in lines:
            (text_w, _), _ = cv2.getTextSize(
                line, self.FONT, 0.62 if line.startswith("Objects:") else 0.54, 1
            )
            max_text_width = max(max_text_width, text_w)

        panel_x1 = 10
        panel_y1 = 10
        row_height = 26
        panel_width = max(220, min(520, max_text_width + 28))
        panel_height = 48 + (len(class_counts) * row_height)
        panel_x2 = panel_x1 + panel_width
        panel_y2 = panel_y1 + panel_height

        self._draw_panel(
            frame, (panel_x1, panel_y1), (panel_x2, panel_y2), alpha=0.44, shadow_alpha=0.16
        )

        y_offset = panel_y1 + 24

        text = f"Objects: {analytics_data.get('object_count', 0)}"
        self._draw_text_with_shadow(
            frame, text, (panel_x1 + 10, y_offset), self.COLOR_TEXT, 0.62, 1
        )

        y_offset += 28
        for class_name, count in class_counts.items():
            text = f"{class_name}: {count}"
            self._draw_text_with_shadow(
                frame, text, (panel_x1 + 10, y_offset), (200, 200, 200), 0.54, 1
            )
            y_offset += row_height

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw current processing FPS on frame."""
        _, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"

        box_w, box_h = 165, 40
        x1 = max(10, w - box_w - 12)
        y1 = 10
        x2 = x1 + box_w
        y2 = y1 + box_h

        self._draw_panel(frame, (x1, y1), (x2, y2), alpha=0.44, shadow_alpha=0.16)
        self._draw_text_with_shadow(frame, text, (x1 + 14, y1 + 27), self.COLOR_TEXT, 0.66, 1)

        return frame

    def draw_lines(self, frame: np.ndarray, lines: List, feature_result=None) -> np.ndarray:
        """Draw configured lines and line-cross counters."""
        if not lines:
            return frame

        h, w = frame.shape[:2]

        for idx, line in enumerate(lines):
            if isinstance(line, dict):
                coords = line.get("coords", [])
                direction = line.get("direction", "downward")
                orientation = line.get("orientation", "horizontal")
            else:
                coords = getattr(line, "coords", [])
                direction = getattr(line, "direction", "downward")
                orientation = getattr(line, "orientation", "horizontal")

            if len(coords) < 2:
                continue

            pixel_coords = self._coords_to_pixels(coords, w, h)
            x1, y1 = pixel_coords[0]
            x2, y2 = pixel_coords[1]

            line_color = (0, 255, 255)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 3)

            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            arrow_length = 30
            if orientation == "horizontal":
                if direction == "downward":
                    arrow_end = (mid_x, mid_y + arrow_length)
                else:
                    arrow_end = (mid_x, mid_y - arrow_length)
            else:
                if direction == "rightward":
                    arrow_end = (mid_x + arrow_length, mid_y)
                else:
                    arrow_end = (mid_x - arrow_length, mid_y)

            cv2.arrowedLine(frame, (mid_x, mid_y), arrow_end, (0, 0, 255), 2, tipLength=0.3)

            label = f"Line {idx + 1}"
            self._draw_text_with_shadow(
                frame, label, (x1 - 10, max(14, y1 - 10)), line_color, 0.56, 1
            )

        if feature_result and feature_result.metrics:
            metrics = feature_result.metrics
            y_pos = h - 100

            self._draw_panel(
                frame, (10, y_pos - 30), (260, y_pos + 55), alpha=0.44, shadow_alpha=0.16
            )
            self._draw_text_with_shadow(
                frame, "Line-Cross Counts:", (20, y_pos), self.COLOR_TEXT, 0.56, 1
            )
            self._draw_text_with_shadow(
                frame,
                f"IN:  {metrics.get('total_in', 0)}",
                (20, y_pos + 25),
                self.COLOR_IN,
                0.56,
                1,
            )
            self._draw_text_with_shadow(
                frame,
                f"OUT: {metrics.get('total_out', 0)}",
                (20, y_pos + 50),
                self.COLOR_OUT,
                0.56,
                1,
            )

        return frame

    def draw_regions(self, frame: np.ndarray, regions: List, feature_result=None) -> np.ndarray:
        """Draw configured regions and region-crowd counters/warnings."""
        if not regions:
            return frame

        h, w = frame.shape[:2]
        metrics = feature_result.metrics if feature_result and feature_result.metrics else {}
        feature_type = str(metrics.get("feature", "")).lower() if isinstance(metrics, dict) else ""
        region_metrics = metrics.get("regions", {}) if isinstance(metrics, dict) else {}

        warning_threshold = (
            int(metrics.get("warning_threshold", 0)) if isinstance(metrics, dict) else 0
        )
        critical_threshold = (
            int(metrics.get("critical_threshold", 0)) if isinstance(metrics, dict) else 0
        )

        for idx, region in enumerate(regions):
            if isinstance(region, dict):
                coords = region.get("coords", [])
                region_id = region.get("id", idx)
            else:
                coords = getattr(region, "coords", [])
                region_id = getattr(region, "id", idx)

            if len(coords) < 3:
                continue

            polygon_points = self._coords_to_pixels(coords, w, h)

            key = f"region_{region_id}"
            info = region_metrics.get(key, {}) if isinstance(region_metrics, dict) else {}
            current_count = int(info.get("current_count", 0)) if isinstance(info, dict) else 0
            status = (
                str(info.get("status", "normal")).lower() if isinstance(info, dict) else "normal"
            )
            current_dwelling = int(info.get("current_dwelling", 0)) if isinstance(info, dict) else 0
            current_dwell_times = (
                info.get("current_dwell_times", []) if isinstance(info, dict) else []
            )
            max_current_dwell = (
                float(max(current_dwell_times))
                if isinstance(current_dwell_times, list) and current_dwell_times
                else 0.0
            )

            color = (255, 200, 0)
            if feature_type == "region_crowd":
                if status == "warning":
                    color = (0, 190, 255)
                elif status == "critical":
                    color = (0, 0, 255)
            elif feature_type == "dwell_time":
                color = (255, 200, 0) if current_dwelling > 0 else (160, 160, 160)

            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(polygon_points, dtype=np.int32)], color)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(
                frame,
                [np.array(polygon_points, dtype=np.int32)],
                isClosed=True,
                color=color,
                thickness=3,
            )

            anchor_x, anchor_y = polygon_points[0]
            if feature_type == "dwell_time":
                label = f"REGION {region_id}: dwell={current_dwelling} max={max_current_dwell:.1f}s"
            else:
                label = f"REGION {region_id}: {current_count}"
                if status == "warning":
                    label += " WARN"
                elif status == "critical":
                    label += " CRITICAL"

            self._draw_badge(
                frame,
                label,
                (max(4, anchor_x), max(4, anchor_y - 30)),
                color,
                scale=0.50,
            )

        if isinstance(metrics, dict):
            panel_top = h - 135
            self._draw_panel(
                frame, (10, panel_top), (360, panel_top + 120), alpha=0.44, shadow_alpha=0.16
            )

            if feature_type == "dwell_time":
                total_dwelling = (
                    sum(
                        int(region_info.get("current_dwelling", 0))
                        for region_info in region_metrics.values()
                        if isinstance(region_info, dict)
                    )
                    if isinstance(region_metrics, dict)
                    else 0
                )
                total_completed = (
                    sum(
                        int(region_info.get("total_completed", 0))
                        for region_info in region_metrics.values()
                        if isinstance(region_info, dict)
                    )
                    if isinstance(region_metrics, dict)
                    else 0
                )
                overall_max = float(metrics.get("overall_max_dwell_seconds", 0.0))

                self._draw_text_with_shadow(
                    frame, "Dwell Time", (20, panel_top + 24), self.COLOR_TEXT, 0.58, 1
                )
                self._draw_text_with_shadow(
                    frame,
                    f"Current inside: {total_dwelling}",
                    (20, panel_top + 50),
                    (255, 220, 180),
                    0.54,
                    1,
                )
                self._draw_text_with_shadow(
                    frame,
                    f"Completed dwell: {total_completed}",
                    (20, panel_top + 76),
                    (120, 210, 255),
                    0.52,
                    1,
                )
                self._draw_text_with_shadow(
                    frame,
                    f"Overall max dwell: {overall_max:.1f}s",
                    (20, panel_top + 102),
                    (255, 200, 100),
                    0.52,
                    1,
                )
            else:
                total_current = int(metrics.get("total_current", 0))
                self._draw_text_with_shadow(
                    frame, "Region Crowd", (20, panel_top + 24), self.COLOR_TEXT, 0.58, 1
                )
                self._draw_text_with_shadow(
                    frame,
                    f"Current: {total_current}",
                    (20, panel_top + 50),
                    (255, 220, 180),
                    0.54,
                    1,
                )
                self._draw_text_with_shadow(
                    frame,
                    f"Warning >= {warning_threshold}",
                    (20, panel_top + 76),
                    (0, 190, 255),
                    0.52,
                    1,
                )
                self._draw_text_with_shadow(
                    frame,
                    f"Critical >= {critical_threshold}",
                    (20, panel_top + 102),
                    (0, 0, 255),
                    0.52,
                    1,
                )

        return frame
