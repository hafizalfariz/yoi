from pathlib import Path
from types import SimpleNamespace

from yoi.components import engine_output_lifecycle as output_lifecycle
from yoi.components.engine_output_lifecycle import _resolve_annotated_output_path


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
