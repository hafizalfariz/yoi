"""
Pytest configuration and fixtures for YOI tests
"""

from pathlib import Path

import pytest


@pytest.fixture
def test_configs_dir():
    """Path to test config fixtures"""
    return Path(__file__).parent / "fixtures" / "configs"


@pytest.fixture
def test_videos_dir():
    """Path to test video fixtures"""
    return Path(__file__).parent / "fixtures" / "videos"


@pytest.fixture
def sample_config_path(test_configs_dir):
    """Path to sample line-cross config"""
    return test_configs_dir / "line-cross-sample.yaml"
