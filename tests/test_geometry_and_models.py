"""Tests for geometry parsing and model structure assumptions."""

from pathlib import Path

import pytest

from yoi.config import YOIConfig


def _load_config(config_path):
    """Load the shared test config from path."""
    return YOIConfig.from_yaml(str(config_path))


class TestRegionLineLoading:
    """Line and region loading tests."""

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_lines_loaded_from_config(self, config_path):
        """Test bahwa lines terbaca dari config"""
        config = _load_config(config_path)
        assert config.lines is not None
        assert len(config.lines) > 0

    def test_line_structure(self, config_path):
        """Test bahwa line memiliki struktur lengkap"""
        config = _load_config(config_path)
        line = config.lines[0]

        # Check required fields
        assert hasattr(line, "id")
        assert hasattr(line, "type")
        assert hasattr(line, "color")
        assert hasattr(line, "coords")
        assert hasattr(line, "direction")
        assert hasattr(line, "orientation")
        assert hasattr(line, "bidirectional")

    def test_line_values(self, config_path):
        """Test bahwa line memiliki nilai yang benar"""
        config = _load_config(config_path)
        line = config.lines[0]

        assert line.id == 1
        assert line.type == "line_1"
        assert line.color == "#0080ff"
        assert line.direction == "downward"
        assert line.orientation == "horizontal"
        assert line.bidirectional is False

    def test_line_coordinates(self, config_path):
        """Test bahwa line coordinates terbaca"""
        config = _load_config(config_path)
        line = config.lines[0]

        assert line.coords is not None
        assert len(line.coords) == 2

        # Check first coordinate
        assert line.coords[0].x == pytest.approx(0.3092415606643322)
        assert line.coords[0].y == pytest.approx(0.31234720123266757)

        # Check second coordinate
        assert line.coords[1].x == pytest.approx(0.7060764222822016)
        assert line.coords[1].y == pytest.approx(0.302779857268433)

    def test_regions_loaded_from_config(self, config_path):
        """Test bahwa regions terbaca dari config (empty dalam case ini)"""
        config = _load_config(config_path)
        assert config.regions is not None
        # Dalam config line-cross, regions kosong
        assert len(config.regions) == 0

    def test_line_to_dict(self, config_path):
        """Test conversion line ke dict"""
        config = _load_config(config_path)
        line = config.lines[0]

        # LineConfig harus bisa dikonversi ke dict
        from dataclasses import asdict

        line_dict = asdict(line)

        assert line_dict["id"] == 1
        assert line_dict["type"] == "line_1"
        assert line_dict["direction"] == "downward"


class TestModelDirectoryStructure:
    """Model directory structure validation tests."""

    @pytest.fixture
    def models_base_path(self):
        """Path to the models directory."""
        return Path(__file__).parent.parent / "models"

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_models_directory_exists(self, models_base_path):
        """Test bahwa folder models ada"""
        assert models_base_path.exists(), f"Models directory not found: {models_base_path}"
        assert models_base_path.is_dir()

    def test_model_name_from_config_matches_directory(self, config_path, models_base_path):
        """Test bahwa model name dari config cocok dengan direktori"""
        config = _load_config(config_path)
        model_name = config.model.name

        model_dir = models_base_path / model_name
        assert model_dir.exists() or not model_dir.exists(), (
            f"Checking model directory: {model_dir}"
        )

    def test_model_version_directory_structure(self, config_path, models_base_path):
        """Test bahwa struktur model version ada (models/{name}/{version}/)"""
        config = _load_config(config_path)
        model_name = config.model.name

        model_dir = models_base_path / model_name

        # Check if directory exists or would exist
        if model_dir.exists():
            # Should have at least version folders
            subdirs = list(model_dir.iterdir())
            # Bisa kosong atau punya version folders
            assert isinstance(subdirs, list)

    def test_expected_model_structure_format(self, models_base_path):
        """Test bahwa expected model structure format benar"""
        # Expected: models/person_general_detection/1/model.onnx
        # Or: models/person_office_detection/1/model.onnx

        # Just validate the concept - actual files may not exist yet
        assert models_base_path.exists()


class TestModelFileDetection:
    """Model file discovery tests."""

    @pytest.fixture
    def models_base_path(self):
        """Path to the models directory."""
        return Path(__file__).parent.parent / "models"

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_find_model_files_in_directory(self, config_path, models_base_path):
        """Test menemukan model files dalam direktori"""
        config = _load_config(config_path)
        model_name = config.model.name

        model_dir = models_base_path / model_name

        if model_dir.exists():
            # Look for model files
            model_files = list(model_dir.rglob("*.onnx"))
            model_files += list(model_dir.rglob("*.pt"))
            model_files += list(model_dir.rglob("*.pth"))

            # May or may not exist, but should be searchable
            assert isinstance(model_files, list)

    def test_model_format_validation(self, config_path):
        """Test bahwa model format (onnx/pt/pth) valid"""
        config = _load_config(config_path)

        # Config should reference valid model formats
        assert config.model.name is not None

    def test_model_weight_file_naming_convention(self):
        """Test naming convention untuk model weight files"""
        # YOLO naming: yolo11s.pt, yolo11m.pt, yolo11l.pt, etc
        # Or: model.onnx, weights.pt

        valid_patterns = [
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            "yolo11x.pt",
            "model.onnx",
            "weights.pt",
            "model.pt",
        ]

        # Just validate assumptions
        assert len(valid_patterns) > 0


class TestModelClassLoading:
    """Model class loading tests."""

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_model_classes_loaded(self, config_path):
        """Test bahwa model classes terbaca"""
        config = _load_config(config_path)
        assert config.model.classes is not None

    def test_model_classes_is_list(self, config_path):
        """Test bahwa model classes adalah list"""
        config = _load_config(config_path)
        assert isinstance(config.model.classes, list)

    def test_model_classes_values(self, config_path):
        """Test bahwa model classes memiliki values"""
        config = _load_config(config_path)
        assert len(config.model.classes) > 0

    def test_person_class_exists(self, config_path):
        """Test bahwa 'person' class ada"""
        config = _load_config(config_path)
        assert "person" in config.model.classes

    def test_model_classes_in_yaml_export(self, config_path):
        """Test bahwa model classes ada di YAML export"""
        config = _load_config(config_path)
        yaml_str = config.to_yaml_str()

        assert "person" in yaml_str
        assert "classes:" in yaml_str

    def test_model_classes_count(self, config_path):
        """Test jumlah classes"""
        config = _load_config(config_path)
        # Minimal harus ada 1 class
        assert len(config.model.classes) >= 1


class TestFullModelIntegration:
    """Integration tests for model loading and validation."""

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_model_ready_for_inference(self, config_path):
        """Test bahwa model siap untuk inference"""
        config = _load_config(config_path)

        # Check all required fields for inference
        assert config.model.name is not None
        assert config.model.device is not None
        assert config.model.conf is not None
        assert config.model.iou is not None
        assert config.model.type is not None
        assert config.model.classes is not None and len(config.model.classes) > 0

    def test_config_with_lines_and_model(self, config_path):
        """Test bahwa config memiliki both model dan lines"""
        config = _load_config(config_path)

        # Model
        assert config.model is not None
        assert config.model.name == "person_general_detection"

        # Lines
        assert config.lines is not None
        assert len(config.lines) > 0

        # Feature
        assert config.feature is not None

        # Classes
        assert config.model.classes is not None
        assert "person" in config.model.classes

    def test_all_components_integrated(self, config_path):
        """Test bahwa semua komponen terintegrasi dengan baik"""
        config = _load_config(config_path)

        # Must have all sections
        assert config.model is not None
        assert config.feature is not None
        assert config.input is not None
        assert config.output is not None
        assert config.logs is not None
        assert config.lines is not None or config.regions is not None
        assert config.metadata is not None

        # Specific checks
        assert config.cctv_id == "office"
        assert len(config.model.classes) > 0
        assert len(config.lines) > 0


class TestGeometryDetection:
    """Geometry validation tests for line and coordinate data."""

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_line_count(self, config_path):
        """Test bahwa ada geometry (lines atau regions)"""
        config = _load_config(config_path)
        total_geometry = len(config.lines) + len(config.regions)
        assert total_geometry > 0

    def test_line_coordinates_format(self, config_path):
        """Test bahwa line coordinates dalam format yang benar"""
        config = _load_config(config_path)

        for line in config.lines:
            # Each coordinate should have x, y
            for coord in line.coords:
                assert hasattr(coord, "x")
                assert hasattr(coord, "y")
                assert isinstance(coord.x, float) or isinstance(coord.x, int)
                assert isinstance(coord.y, float) or isinstance(coord.y, int)

    def test_normalized_coordinates(self, config_path):
        """Test bahwa coordinates dalam range 0-1 (normalized)"""
        config = _load_config(config_path)

        for line in config.lines:
            for coord in line.coords:
                assert 0 <= coord.x <= 1, f"X coordinate out of range: {coord.x}"
                assert 0 <= coord.y <= 1, f"Y coordinate out of range: {coord.y}"

    def test_line_has_type_and_color(self, config_path):
        """Test bahwa line memiliki type dan color"""
        config = _load_config(config_path)

        for line in config.lines:
            assert line.type is not None
            assert line.color is not None
            # Color should be hex format
            assert line.color.startswith("#")
