"""Centralized alert manager for runtime feature alerts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AlertRecord:
    """Normalized alert payload persisted by AlertManager."""

    timestamp: str
    frame_idx: int
    feature: str
    cctv_id: str
    alert: Dict[str, Any]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_idx": self.frame_idx,
            "feature": self.feature,
            "cctv_id": self.cctv_id,
            "alert": self.alert,
            "metrics": self.metrics,
        }


class AlertManager:
    """Persist alerts from all features to a single destination."""

    def __init__(self, output_dir: Path, logger=None):
        self.output_dir = Path(output_dir)
        self.logger = logger

        self.alerts_dir = self.output_dir / "alerts"
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        self.alerts_jsonl = self.alerts_dir / "alerts.jsonl"
        self.alerts_json = self.alerts_dir / "alerts.json"

    def record(
        self,
        *,
        frame_idx: int,
        feature: str,
        cctv_id: str,
        alerts: Optional[List[Dict[str, Any]]],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist alert list and return number of written alerts."""
        if not alerts:
            return 0

        written = 0
        json_records: List[Dict[str, Any]] = []

        for alert in alerts:
            record = AlertRecord(
                timestamp=datetime.utcnow().isoformat(),
                frame_idx=frame_idx,
                feature=feature,
                cctv_id=cctv_id,
                alert=alert,
                metrics=metrics or {},
            )
            payload = record.to_dict()
            json_records.append(payload)

            with self.alerts_jsonl.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")

            written += 1

        self._refresh_alerts_json()

        if self.logger is not None:
            self.logger.info(f"AlertManager persisted {written} alert(s) to {self.alerts_dir}")

        return written

    def _refresh_alerts_json(self) -> None:
        """Materialize compact JSON array from the JSONL stream."""
        records: List[Dict[str, Any]] = []
        if self.alerts_jsonl.exists():
            for line in self.alerts_jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        self.alerts_json.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
