"""Tests for YOLO device fallback behavior during model initialization."""

from yoi.inference import yolo as yolo_module


class _FakeYOLO:
    def __init__(self, *_args, **_kwargs):
        self.names = {0: "person"}
        self.last_device = None

    def to(self, device: str):
        self.last_device = device
        if device == "cuda":
            raise AssertionError("Torch not compiled with CUDA enabled")
        return self


class _FakeYOLOStrict(_FakeYOLO):
    pass


def test_yolo_init_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(yolo_module, "YOLO", _FakeYOLO)
    monkeypatch.setattr(yolo_module, "HAS_ULTRALYTICS", True)
    monkeypatch.setenv("YOI_STRICT_DEVICE", "0")
    monkeypatch.delenv("YOI_TARGET_DEVICE", raising=False)

    inferencer = yolo_module.YOLOInferencer(
        model_name="person_general_detection",
        device="cuda",
        conf=0.5,
        iou=0.7,
        classes=["person"],
    )

    assert inferencer.device == "cpu"
    assert inferencer._fallback_to_cpu_attempted is True


def test_yolo_init_raises_in_strict_mode_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(yolo_module, "YOLO", _FakeYOLOStrict)
    monkeypatch.setattr(yolo_module, "HAS_ULTRALYTICS", True)
    monkeypatch.setenv("YOI_STRICT_DEVICE", "1")
    monkeypatch.delenv("YOI_TARGET_DEVICE", raising=False)

    raised = False
    try:
        yolo_module.YOLOInferencer(
            model_name="person_general_detection",
            device="cuda",
            conf=0.5,
            iou=0.7,
            classes=["person"],
        )
    except AssertionError as exc:
        raised = True
        assert "Torch not compiled with CUDA enabled" in str(exc)

    assert raised is True
