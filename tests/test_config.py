"""
Test configuration loading
"""

import pytest

from yoi.config import YOIConfig


def test_load_config_from_yaml(sample_config_path):
    """Test loading YAML config"""
    if not sample_config_path.exists():
        pytest.skip(f"Sample config not found: {sample_config_path}")

    config = YOIConfig.from_yaml(str(sample_config_path))
    assert config is not None
    assert config.feature == "line_cross"


def test_config_has_lines(sample_config_path):
    """Test that config has lines defined"""
    if not sample_config_path.exists():
        pytest.skip(f"Sample config not found: {sample_config_path}")

    config = YOIConfig.from_yaml(str(sample_config_path))
    assert len(config.lines) > 0, "Config should have at least one line defined"
