"""Tests for validating feature-related config loading behavior."""

import pytest

from yoi.config import VideoInputConfig, YOIConfig


def _load_test_config(config_path):
    """Load the shared test config from path."""
    return YOIConfig.from_yaml(str(config_path))


@pytest.fixture
def config_path(sample_config_path):
    """Path to the sample config used by tests."""
    return sample_config_path


class TestConfigFeatureLoading:
    """Feature config loading and validation tests."""

    def test_config_file_exists(self, config_path):
        """Ensure the sample config file exists."""
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_load_config_from_yaml(self, config_path):
        """Load config from YAML and validate object type."""
        config = _load_test_config(config_path)
        assert config is not None
        assert isinstance(config, YOIConfig)

    def test_config_model_loaded(self, config_path):
        """Ensure model config fields are parsed correctly."""
        config = _load_test_config(config_path)
        assert config.model is not None
        assert config.model.name == "person_general_detection"
        assert config.model.device == "cpu"
        assert config.model.conf == 0.4
        assert config.model.iou == 0.7
        assert config.model.type == "small"
        assert "person" in config.model.classes

    def test_config_feature_loaded(self, config_path):
        """Ensure feature name is parsed correctly."""
        config = _load_test_config(config_path)
        assert config.feature is not None
        assert config.feature == "line_cross"

    def test_config_feature_params_loaded(self, config_path):
        """Ensure feature params are parsed correctly."""
        config = _load_test_config(config_path)
        assert config.feature_params is not None
        assert hasattr(config.feature_params, "centroid")
        assert config.feature_params.centroid == "mid_centre"
        assert config.feature_params.lost_threshold == 10
        assert config.feature_params.allow_recounting is True

    def test_config_input_loaded(self, config_path):
        """Ensure input config is parsed correctly."""
        config = _load_test_config(config_path)
        assert config.input is not None
        assert config.input.source_type == "video"
        assert config.input.video_source == "input/5.mp4"
        assert config.input.max_fps == 30
        assert config.input.input_path == "input"

    def test_config_output_loaded(self, config_path):
        """Ensure output config is parsed correctly."""
        config = _load_test_config(config_path)
        assert config.output is not None
        assert config.output.output_path == "output"
        assert config.output.save_video.enabled is True
        assert config.output.save_annotations.enabled is True
        assert config.output.output_format == "annotated_video"

    def test_config_logs_loaded(self, config_path):
        """Ensure logs config is parsed correctly."""
        config = _load_test_config(config_path)
        assert config.logs is not None
        assert config.logs.base_dir == "logs"
        assert config.logs.data_folder == "data"
        assert config.logs.image_folder == "image"
        assert config.logs.csv_file == "data.csv"

    def test_config_lines_loaded(self, config_path):
        """Ensure line config is parsed correctly."""
        config = _load_test_config(config_path)
        assert config.lines is not None
        assert len(config.lines) == 1

        line = config.lines[0]
        assert line.id == 1
        assert line.type == "line_1"
        assert line.direction == "downward"
        assert line.orientation == "horizontal"
        assert line.bidirectional is False
        assert len(line.coords) == 2
        assert line.coords[0].x == pytest.approx(0.3092415606643322)
        assert line.coords[0].y == pytest.approx(0.31234720123266757)

    def test_config_metadata_loaded(self, config_path):
        """Ensure metadata is parsed correctly."""
        config = _load_test_config(config_path)
        assert config.metadata is not None
        assert config.metadata.get("cctv_id") == "office"
        assert config.cctv_id == "office"

    def test_config_cctv_id(self, config_path):
        """Ensure CCTV ID is loaded from metadata."""
        config = _load_test_config(config_path)
        assert config.cctv_id == "office"

    def test_config_regions_empty(self, config_path):
        """Ensure regions list is empty as expected."""
        config = _load_test_config(config_path)
        assert config.regions is not None
        assert len(config.regions) == 0

    def test_config_to_dict(self, config_path):
        """Ensure config can be converted to dict."""
        config = _load_test_config(config_path)
        config_dict = config.to_dict()

        assert config_dict is not None
        assert config_dict["feature"] == "line_cross"
        assert config_dict["model"]["name"] == "person_general_detection"
        assert len(config_dict["lines"]) == 1

    def test_config_to_yaml_str(self, config_path):
        """Ensure config can be converted to YAML string."""
        config = _load_test_config(config_path)
        yaml_str = config.to_yaml_str()

        assert yaml_str is not None
        assert "line_cross" in yaml_str
        assert "person_general_detection" in yaml_str
        assert "office" in yaml_str

    def test_all_required_fields_present(self, config_path):
        """Ensure all required top-level fields are present."""
        config = _load_test_config(config_path)

        assert config.config_name is not None
        assert config.cctv_id is not None
        assert config.model is not None
        assert config.feature is not None
        assert config.input is not None
        assert config.output is not None
        assert config.logs is not None
        assert config.lines is not None
        assert config.regions is not None
        assert config.metadata is not None

    def test_feature_specific_params_exist(self, config_path):
        """Ensure feature-specific parameters are available."""
        config = _load_test_config(config_path)

        assert config.feature_params is not None

        params = config.feature_params
        assert hasattr(params, "centroid") or hasattr(params, "lost_threshold")

    def test_line_coordinates_parsed_correctly(self, config_path):
        """Ensure line coordinates are parsed with expected types and ranges."""
        config = _load_test_config(config_path)

        assert len(config.lines) > 0
        line = config.lines[0]

        assert isinstance(line.coords[0].x, float)
        assert isinstance(line.coords[0].y, float)
        assert isinstance(line.coords[1].x, float)
        assert isinstance(line.coords[1].y, float)

        assert 0 <= line.coords[0].x <= 1
        assert 0 <= line.coords[0].y <= 1
        assert 0 <= line.coords[1].x <= 1
        assert 0 <= line.coords[1].y <= 1


class TestConfigIntegration:
    """Integration tests for config loading compatibility."""

    def test_config_usable_for_engine_init(self, config_path):
        """Ensure config has minimum fields needed for engine initialization."""

        config = _load_test_config(config_path)

        assert config.feature == "line_cross"
        assert config.model.name == "person_general_detection"

    def test_multiple_config_loads_consistent(self, config_path):
        """Ensure multiple loads produce consistent values."""
        config1 = _load_test_config(config_path)
        config2 = _load_test_config(config_path)

        assert config1.feature == config2.feature
        assert config1.cctv_id == config2.cctv_id
        assert config1.model.name == config2.model.name
        assert len(config1.lines) == len(config2.lines)


class TestVideoInputSources:
    """Tests for multi-video source resolution behavior."""

    def test_get_source_paths_from_video_files_uses_input_path_for_bare_names(self):
        cfg = VideoInputConfig(
            source_type="video",
            input_path="input",
            video_files=["14.mp4", "15.mp4"],
        )

        assert cfg.get_source_paths() == ["input/14.mp4", "input/15.mp4"]

    def test_get_source_paths_keeps_unique_order_across_fields(self):
        cfg = VideoInputConfig(
            source_type="video",
            input_path="input",
            source="input/14.mp4",
            video_source="14.mp4",
            video_files=["14.mp4", "15.mp4"],
        )

        assert cfg.get_source_paths() == ["input/14.mp4", "input/15.mp4"]
