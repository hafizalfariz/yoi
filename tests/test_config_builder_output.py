"""Tests for config builder YAML output shape."""

import yaml

from config_builder.models import BuildConfigRequest, Line, ModelConfig, Point
from config_builder.services.config_service import config_service


def _line_cross_request(source_mode: str) -> BuildConfigRequest:
    return BuildConfigRequest(
        config_name="builder_no_logs",
        model=ModelConfig(
            name="person_general_detection",
            device="cpu",
            conf=0.4,
            iou=0.7,
            type="small",
            classes=["person"],
        ),
        feature="line_cross",
        feature_params={"centroid": "mid_centre", "lost_threshold": 10},
        lines=[
            Line(
                coords=[Point(x=0.2, y=0.3), Point(x=0.8, y=0.7)],
                id=1,
                type="line_1",
                color="#0080ff",
                direction="rightward",
                orientation="diagonal",
                bidirectional=False,
                mode=[],
            )
        ],
        regions=[],
        source_mode=source_mode,
        cctv_id="office",
        max_fps=30,
    )


def test_config_builder_output_excludes_logs_block_for_inference_mode():
    yaml_content = config_service.build_yaml(_line_cross_request("inference"))
    parsed = yaml.safe_load(yaml_content)

    assert "logs" not in parsed
    assert "output" in parsed
    assert parsed["input"]["source_type"] == "video"


def test_config_builder_output_excludes_logs_block_for_production_mode():
    yaml_content = config_service.build_yaml(_line_cross_request("production"))
    parsed = yaml.safe_load(yaml_content)

    assert "logs" not in parsed
    assert "output" in parsed
    assert parsed["input"]["source_type"] == "rtsp"
