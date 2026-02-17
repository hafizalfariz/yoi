from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from yoi.components import engine_output_lifecycle as output_lifecycle
from yoi.components.engine_output_lifecycle import (
    _resolve_annotated_output_path,
    handle_feature_alert_events,
)


class _DummyInput:
    def __init__(self, source_path: str):
        self._source_path = source_path

    def get_source_path(self) -> str:
        return self._source_path


def _dummy_engine(config_name: str, source_path: str, active_stem: str | None = None):
    metadata = {}
    if active_stem:
        metadata["_active_config_stem"] = active_stem

    config = SimpleNamespace(
        config_name=config_name,
        metadata=metadata,
        input=_DummyInput(source_path),
    )
    return SimpleNamespace(config=config)


def test_output_path_prefers_active_config_stem_and_video_name():
    engine = _dummy_engine(
        config_name="fallback_name",
        source_path="input/15.mp4",
        active_stem="kluis-line",
    )

    result = _resolve_annotated_output_path(engine, Path("output"))

    assert result.parts[0] == "output"
    assert result.parts[1] == "kluis-line"
    assert result.parts[2].startswith("15_")


def test_output_path_uses_config_name_when_active_stem_missing():
    engine = _dummy_engine(
        config_name="custom_cfg",
        source_path="input/14.mp4",
        active_stem=None,
    )

    result = _resolve_annotated_output_path(engine, Path("output"))

    assert result.parts[0] == "output"
    assert result.parts[1] == "custom_cfg"
    assert result.parts[2].startswith("14_")


def test_rtsp_source_type_forces_logs_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(output_lifecycle, "_initialize_rtsp", lambda engine: None)

    base_logs_dir = tmp_path / "logs"
    output_dir = tmp_path / "output"

    class _DummyInput:
        source_type = "rtsp"

        @staticmethod
        def get_source_path() -> str:
            return "rtsp://example.com/stream"

    class _DummyOutput:
        mode = "development"
        output_format = "json"
        save_video = False
        save_annotations = False

        @staticmethod
        def get_output_dir() -> str:
            return str(output_dir)

    class _DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    engine = SimpleNamespace(
        config=SimpleNamespace(
            input=_DummyInput(),
            output=_DummyOutput(),
            logs=SimpleNamespace(base_dir=str(base_logs_dir)),
            metadata={},
            config_name="rtsp-test",
        ),
        video_reader=None,
        logger=_DummyLogger(),
    )

    output_lifecycle.initialize_output_engines(engine)

    assert engine.output_dir == base_logs_dir
    assert (base_logs_dir / "data.csv").exists()


def test_logs_config_folder_and_csv_names_are_respected(tmp_path, monkeypatch):
    monkeypatch.setattr(output_lifecycle, "_initialize_rtsp", lambda engine: None)

    class _DummyInput:
        source_type = "video"

        @staticmethod
        def get_source_path() -> str:
            return "input/demo.mp4"

    class _DummyOutput:
        mode = "development"
        output_format = "json"
        save_video = False
        save_annotations = False

        @staticmethod
        def get_output_dir() -> str:
            return str(tmp_path / "output")

    class _DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    engine = SimpleNamespace(
        config=SimpleNamespace(
            input=_DummyInput(),
            output=_DummyOutput(),
            logs=SimpleNamespace(
                base_dir=str(tmp_path / "logs"),
                data_folder="data_custom",
                image_folder="image_custom",
                status_folder="status_custom",
                csv_file="event_custom.csv",
            ),
            metadata={},
            config_name="video-test",
        ),
        video_reader=None,
        logger=_DummyLogger(),
    )

    output_lifecycle.initialize_output_engines(engine)

    assert engine.data_dir.name == "data_custom"
    assert engine.image_dir.name == "image_custom"
    assert engine.status_dir.name == "status_custom"
    assert engine.data_csv_path.name == "event_custom.csv"
    assert engine.data_csv_path.exists()


def test_feature_alert_event_writes_data_image_csv_and_skips_status_for_video(tmp_path):
    class _DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    class _Det:
        x1 = 10
        y1 = 10
        x2 = 60
        y2 = 80

    output_dir = tmp_path / "output"
    image_dir = output_dir / "image"
    data_dir = output_dir / "data"
    status_dir = output_dir / "status"
    for directory in (output_dir, image_dir, data_dir, status_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data_csv = output_dir / "data.csv"
    data_csv.write_text("image_id,timestamp,feature,status,data_path,image_path\n", encoding="utf-8")

    engine = SimpleNamespace(
        config=SimpleNamespace(
            config_name="cfg-a",
            cctv_id="office",
            input=SimpleNamespace(get_source_path=lambda: "input/demo.mp4"),
        ),
        logger=_DummyLogger(),
        output_dir=output_dir,
        image_dir=image_dir,
        data_dir=data_dir,
        status_dir=status_dir,
        data_csv_path=data_csv,
        _event_counter=0,
        _event_status_enabled=False,
    )

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[10:80, 10:60] = 255

    feature_result = SimpleNamespace(
        feature_type="line_cross",
        metrics={"feature": "line_cross", "total_in": 1, "total_out": 0},
        alerts=[{"type": "line_crossing_in", "track_id": 7, "line_id": 1}],
    )

    handle_feature_alert_events(
        engine=engine,
        frame_idx=12,
        frame=frame,
        annotated_frame=frame,
        feature_result=feature_result,
        track_bbox_map={7: _Det()},
    )

    data_files = list(data_dir.glob("*.json"))
    image_files = list(image_dir.glob("*.jpg"))
    status_files = list(status_dir.glob("*.json"))

    assert len(data_files) == 1
    assert len(image_files) == 1
    assert len(status_files) == 0

    image = cv2.imread(str(image_files[0]))
    assert image is not None
    assert image.shape[0] < frame.shape[0]
    assert image.shape[1] < frame.shape[1]

    payload = data_files[0].read_text(encoding="utf-8")
    assert '"feature": "line_cross"' in payload
    assert '"warning": "line_crossing_in"' in payload
    assert '"track_id": 7' in payload

    csv_lines = data_csv.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 2
    assert "line_cross" in csv_lines[1]
    assert "line_crossing_in" in csv_lines[1]
