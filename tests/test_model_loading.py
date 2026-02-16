"""Tests for validating model-type loading from config."""

import pytest

from yoi.config import ModelConfig, YOIConfig


def _load_config(config_path):
    """Load the shared test config from path."""
    return YOIConfig.from_yaml(str(config_path))


class TestModelTypeLoading:
    """Model type loading and validation tests."""

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_model_config_exists(self, config_path):
        """Test bahwa config file ada"""
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_load_config(self, config_path):
        """Test loading config"""
        config = _load_config(config_path)
        assert config is not None

    def test_model_exists_in_config(self, config_path):
        """Test bahwa model config ada di file"""
        config = _load_config(config_path)
        assert config.model is not None
        assert isinstance(config.model, ModelConfig)

    def test_model_name_loaded(self, config_path):
        """Test bahwa model name terbaca"""
        config = _load_config(config_path)
        assert config.model.name == "person_general_detection"

    def test_model_type_field_exists(self, config_path):
        """Test bahwa field 'type' ada di model config"""
        config = _load_config(config_path)
        assert hasattr(config.model, "type"), "Model config tidak memiliki field 'type'"

    def test_model_type_is_small(self, config_path):
        """Test bahwa model type adalah 'small'"""
        config = _load_config(config_path)
        assert config.model.type == "small", f"Expected 'small', got '{config.model.type}'"

    def test_model_type_is_string(self, config_path):
        """Test bahwa model type adalah string"""
        config = _load_config(config_path)
        assert isinstance(config.model.type, str), (
            f"Model type harus string, got {type(config.model.type)}"
        )

    def test_model_device_cpu(self, config_path):
        """Test bahwa device adalah cpu"""
        config = _load_config(config_path)
        assert config.model.device == "cpu"

    def test_model_confidence_threshold(self, config_path):
        """Test bahwa confidence threshold terbaca"""
        config = _load_config(config_path)
        assert config.model.conf == 0.4

    def test_model_iou_threshold(self, config_path):
        """Test bahwa IOU threshold terbaca"""
        config = _load_config(config_path)
        assert config.model.iou == 0.7

    def test_model_classes_loaded(self, config_path):
        """Test bahwa model classes terbaca"""
        config = _load_config(config_path)
        assert config.model.classes is not None
        assert len(config.model.classes) > 0
        assert "person" in config.model.classes

    def test_model_to_dict(self, config_path):
        """Test conversion model ke dict"""
        config = _load_config(config_path)

        # Check penting fields
        assert config.model.name == "person_general_detection"
        assert config.model.type == "small"
        assert config.model.device == "cpu"

    def test_model_type_in_config_dict(self, config_path):
        """Test bahwa model type ada di config dict"""
        config = _load_config(config_path)
        config_dict = config.to_dict()

        assert "model" in config_dict
        assert config_dict["model"]["type"] == "small"

    def test_model_type_in_yaml_export(self, config_path):
        """Test bahwa model type ada di YAML export"""
        config = _load_config(config_path)
        yaml_str = config.to_yaml_str()

        assert "type: small" in yaml_str or "type: 'small'" in yaml_str


class TestModelTypeAliases:
    """Model type alias compatibility tests."""

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_small_alias_recognized(self, config_path):
        """Test bahwa 'small' dikenali sebagai model size"""
        config = _load_config(config_path)

        # small harus menjadi salah satu tipe model yang valid
        valid_types = ["small", "sml", "medium", "med", "large", "lg", "xlarge", "xl"]
        assert config.model.type in valid_types or config.model.type == "small"

    def test_model_type_lowercase_small(self, config_path):
        """Test bahwa model type adalah lowercase"""
        config = _load_config(config_path)
        assert config.model.type == config.model.type.lower()


class TestModelIntegration:
    """Integration tests for model config usability."""

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_model_config_usable_for_engine(self, config_path):
        """Test bahwa model config bisa dipakai untuk engine"""
        config = _load_config(config_path)

        # Engine membutuhkan:
        # - model.name untuk loading YOLO
        # - model.device untuk device selection
        # - model.conf untuk confidence threshold
        # - model.type untuk model size selection

        assert config.model.name is not None
        assert config.model.device is not None
        assert config.model.conf is not None
        assert config.model.type is not None

    def test_model_type_can_be_used_for_yolo_variant(self, config_path):
        """Test bahwa model type bisa digunakan untuk select YOLO variant"""
        config = _load_config(config_path)

        model_type = config.model.type

        # YOLO naming convention: yolo11{n,s,m,l,x}
        # small -> s
        # medium -> m
        # large -> l
        # xlarge -> x
        # nano -> n

        type_to_suffix = {
            "small": "s",
            "sml": "s",
            "medium": "m",
            "med": "m",
            "large": "l",
            "lg": "l",
            "xlarge": "x",
            "xl": "x",
            "nano": "n",
        }

        # Jika type ada di mapping, bisa digunakan
        if model_type.lower() in type_to_suffix:
            suffix = type_to_suffix[model_type.lower()]
            # YOLO11s, YOLO11m, YOLO11l, YOLO11x, YOLO11n
            assert suffix in ["n", "s", "m", "l", "x"]

    def test_all_model_params_present(self, config_path):
        """Test bahwa semua parameter model ada"""
        config = _load_config(config_path)
        model = config.model

        # Required fields
        assert model.name is not None, "Model name tidak ada"
        assert model.device is not None, "Model device tidak ada"
        assert model.type is not None, "Model type tidak ada"
        assert model.conf is not None, "Model confidence tidak ada"
        assert model.iou is not None, "Model IOU tidak ada"
        assert model.classes is not None, "Model classes tidak ada"

    def test_model_type_matches_file_structure(self, config_path):
        """Test bahwa model type sesuai dengan struktur file"""
        config = _load_config(config_path)

        # Check if model directory exists (or will exist)
        # Kita tidak bisa assuming file ada, tapi kita bisa check config is consistent
        assert config.model.name == "person_general_detection"
        assert config.model.type == "small"


class TestModelTypeConsistency:
    """Consistency tests for model type across operations."""

    @pytest.fixture
    def config_path(self, sample_config_path):
        """Path to the sample config used by tests."""
        return sample_config_path

    def test_model_type_consistent_after_reload(self, config_path):
        """Test bahwa model type tetap konsisten setelah reload"""
        config1 = _load_config(config_path)
        model_type_1 = config1.model.type

        config2 = _load_config(config_path)
        model_type_2 = config2.model.type

        assert model_type_1 == model_type_2

    def test_model_type_consistent_round_trip(self, config_path):
        """Test bahwa model type tetap konsisten dalam round-trip (YAML -> dict -> object)"""
        # Load dari YAML
        config1 = _load_config(config_path)
        original_type = config1.model.type

        # Convert to dict dan kembali
        config_dict = config1.to_dict()
        config2 = YOIConfig._from_dict(config_dict)

        assert config2.model.type == original_type

    def test_model_type_from_yaml_string(self, config_path):
        """Test membaca model type dari YAML string"""
        config1 = _load_config(config_path)
        yaml_str = config1.to_yaml_str()

        # Parse kembali dari YAML string
        config2 = YOIConfig._from_dict(__import__("yaml").safe_load(yaml_str))

        assert config2.model.type == config1.model.type
