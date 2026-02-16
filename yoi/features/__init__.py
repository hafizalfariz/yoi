"""Features Module

Detection features for YOI Vision Engine:
- Line Crossing: Count in/out movements across lines
- Region Crowd: Count objects in defined regions
- Dwell Time: Track time spent in regions
"""

from .base import BaseFeature, Detection, FeatureResult
from .dwell_time import DwellTimeFeature
from .line_cross import LineCrossFeature
from .region_crowd import RegionCrowdFeature

__all__ = [
    "BaseFeature",
    "Detection",
    "FeatureResult",
    "LineCrossFeature",
    "RegionCrowdFeature",
    "DwellTimeFeature",
]


def get_feature_class(feature_type: str):
    """Get feature class by type name

    Args:
        feature_type: 'line_cross', 'region_crowd', or 'dwell_time'

    Returns:
        Feature class

    Raises:
        ValueError: If feature type is unknown
    """
    features = {
        "line_cross": LineCrossFeature,
        "region_crowd": RegionCrowdFeature,
        "dwell_time": DwellTimeFeature,
    }

    if feature_type not in features:
        raise ValueError(
            f"Unknown feature type: {feature_type}. Available: {list(features.keys())}"
        )

    return features[feature_type]


def get_feature(feature_type: str, config: dict):
    """Get instantiated feature with config

    Args:
        feature_type: Feature type name
        config: Feature configuration dict

    Returns:
        Instantiated feature instance
    """
    feature_class = get_feature_class(feature_type)
    return feature_class(config)
