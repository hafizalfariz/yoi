"""Export helpers for frame artifacts and optional annotated video output."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of values into JSON-serializable forms."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            pass

    return str(value)


class VideoWriter:
    """Thin wrapper around OpenCV video writer used by the engine."""

    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int], codec: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        width, height = frame_size
        self._frame_size = (int(width), int(height))
        safe_fps = float(fps) if fps and float(fps) > 0 else 25.0
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            safe_fps,
            self._frame_size,
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")

    def write_frame(self, frame) -> None:
        if self._writer is None or frame is None:
            return

        expected_width, expected_height = self._frame_size
        frame_height, frame_width = frame.shape[:2]
        if frame_width != expected_width or frame_height != expected_height:
            frame = cv2.resize(frame, self._frame_size)

        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class DataExporter:
    """Collect and export frame-level data produced by the engine."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._frames: list[dict[str, Any]] = []
        self._json_path = self.output_dir / "detections.json"
        self._csv_path = self.output_dir / "detections.csv"
        self._log_path = self.output_dir / "processing.log"

    def add_frame(
        self,
        frame_idx: int,
        detections: Any,
        tracked_objects: Any,
        analytics: Any,
    ) -> None:
        record = {
            "frame_idx": int(frame_idx),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detections": _to_jsonable(detections),
            "tracked_objects": _to_jsonable(tracked_objects),
            "analytics": _to_jsonable(analytics),
        }
        self._frames.append(record)

    def export_json(self) -> None:
        payload = {
            "total_frames": len(self._frames),
            "frames": self._frames,
        }
        self._json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def export_csv(self) -> None:
        with self._csv_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "frame_idx",
                    "timestamp",
                    "detections_count",
                    "tracked_count",
                    "detections",
                    "tracked_objects",
                    "analytics",
                ],
            )
            writer.writeheader()

            for frame in self._frames:
                detections = frame.get("detections") or []
                tracked_objects = frame.get("tracked_objects") or []
                writer.writerow(
                    {
                        "frame_idx": frame.get("frame_idx"),
                        "timestamp": frame.get("timestamp"),
                        "detections_count": len(detections)
                        if isinstance(detections, list)
                        else 0,
                        "tracked_count": len(tracked_objects)
                        if isinstance(tracked_objects, list)
                        else 0,
                        "detections": json.dumps(detections, ensure_ascii=False),
                        "tracked_objects": json.dumps(tracked_objects, ensure_ascii=False),
                        "analytics": json.dumps(
                            frame.get("analytics"),
                            ensure_ascii=False,
                        ),
                    }
                )

    def export_logs(self) -> None:
        lines = [
            f"exported_at={datetime.now(timezone.utc).isoformat()}",
            f"total_frames={len(self._frames)}",
        ]

        if self._frames:
            first_frame = self._frames[0]
            last_frame = self._frames[-1]
            lines.extend(
                [
                    f"first_frame={first_frame.get('frame_idx')}",
                    f"last_frame={last_frame.get('frame_idx')}",
                    f"first_timestamp={first_frame.get('timestamp')}",
                    f"last_timestamp={last_frame.get('timestamp')}",
                ]
            )

        self._log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
