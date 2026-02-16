"""
Feature Params Service - Manage feature parameters storage
Allows updating feature_params independently from config building
"""

import copy
from typing import Any, Dict

from .logger_service import logger


class FeatureParamsService:
    """Service to manage feature parameters for different features"""

    VALID_CENTROIDS = ("mid_centre", "head", "bottom")

    _BASE_TRACKING_DEFAULTS = {
        "tracker_impl": "bytetrack",
        "max_lost_frames": 45,
        "max_distance": 140.0,
        "bt_track_high_thresh": 0.5,
        "bt_track_low_thresh": 0.1,
        "bt_new_track_thresh": 0.6,
        "bt_match_thresh": 0.8,
        "bt_track_buffer": 45,
        "bt_fuse_score": True,
        "reid_enabled": True,
        "reid_similarity_thresh": 0.86,
        "reid_momentum": 0.25,
        "min_detection_confidence": 0.4,
        "hit_point": "head",
        "margin_px": 20,
    }

    _DWELL_TRACKING_DEFAULTS = {
        "tracker_impl": "bytetrack",
        "max_lost_frames": 60,
        "max_distance": 120,
        "bt_track_high_thresh": 0.5,
        "bt_track_low_thresh": 0.1,
        "bt_new_track_thresh": 0.6,
        "bt_match_thresh": 0.8,
        "bt_track_buffer": 60,
        "bt_fuse_score": True,
        "reid_enabled": True,
        "reid_similarity_thresh": 0.86,
        "reid_momentum": 0.25,
        "min_detection_confidence": 0.25,
    }

    # Default parameters for each feature - MINIMAL PRODUCTION SET
    DEFAULTS = {
        "line_cross": {
            "centroid": "mid_centre",
            "lost_threshold": 10,
            "allow_recounting": False,
            "time_allowed": "",
            "alert_threshold": 1,
            "cooldown_seconds": 5,
            "tracking": _BASE_TRACKING_DEFAULTS,
            "alerts": {
                "in_warning_threshold": 1,
                "out_warning_threshold": 1,
            },
            "aggregation": {
                "window_seconds": 5,
            },
        },
        "region_crowd": {
            "max_count": 2,
            "alert_threshold": 4,
            "cooldown": 300,
            "warning_threshold": 3,
            "critical_threshold": 6,
            "tracking": _BASE_TRACKING_DEFAULTS,
            "time_allowed": "",
        },
        "dwell_time": {
            "centroid": "bottom",
            "min_dwelltime": 15,
            "alert_threshold": 10,
            "cooldown": 120,
            "warning_seconds": 10,
            "critical_seconds": 20,
            "lost_threshold": 10,
            "tracking": _DWELL_TRACKING_DEFAULTS,
            "time_allowed": "",
        },
    }

    def __init__(self):
        # In-memory storage for feature params
        self._storage: Dict[str, Dict[str, Any]] = {
            feature: self._deep_copy_params(params) for feature, params in self.DEFAULTS.items()
        }

        logger.info("FEATURE_PARAMS", "FeatureParamsService initialized")

    def _deep_copy_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy params preserving nested structures"""
        return copy.deepcopy(params)

    def _ensure_feature(self, feature: str, reason: str) -> None:
        """Ensure feature storage exists, initializing from defaults when needed."""
        if feature in self._storage:
            return
        logger.warning("FEATURE_PARAMS", f"Unknown feature: {feature}, {reason}")
        self._storage[feature] = self._deep_copy_params(self.DEFAULTS.get(feature, {}))

    def get_params(self, feature: str) -> Dict[str, Any]:
        """
        Get current feature parameters

        Args:
            feature: Feature name (line_cross, region_crowd, dwell_time)

        Returns:
            Dictionary of parameters
        """
        self._ensure_feature(feature, "using defaults")

        params = self._deep_copy_params(self._storage[feature])
        logger.info("FEATURE_PARAMS", f"Retrieved params for: {feature}")
        return params

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge source into target
        Nested dicts are merged, not replaced
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                self._deep_merge(target[key], value)
            else:
                # Direct assignment for other types
                target[key] = value

    def _validate_params(self, feature: str, params: Dict[str, Any]) -> None:
        """
        Validate parameters for a feature

        Args:
            feature: Feature name
            params: Parameters to validate

        Raises:
            ValueError: If validation fails
        """
        if feature == "line_cross":
            self._validate_line_cross_params(params)

    def _validate_line_cross_params(self, params: Dict[str, Any]) -> None:
        """Validate essential line-cross parameter shapes and ranges."""
        lost_threshold = params.get("lost_threshold")
        if lost_threshold is not None:
            if not isinstance(lost_threshold, int) or lost_threshold < 0:
                raise ValueError("lost_threshold must be a non-negative integer")

        centroid = params.get("centroid")
        if centroid is not None and centroid not in self.VALID_CENTROIDS:
            raise ValueError(f"centroid must be one of: {', '.join(self.VALID_CENTROIDS)}")

        allow_recounting = params.get("allow_recounting")
        if allow_recounting is not None and not isinstance(allow_recounting, bool):
            raise ValueError("allow_recounting must be a boolean")

    def _has_feature_defaults(self, feature: str) -> bool:
        """Check if feature has known default configuration."""
        return feature in self.DEFAULTS

    def update_params(self, feature: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update feature parameters with deep merging for nested objects

        Args:
            feature: Feature name
            params: New parameters (will merge with existing)

        Returns:
            Updated parameters
        """
        self._ensure_feature(feature, "creating new")

        # Validate input params
        self._validate_params(feature, params)

        # Deep merge with existing params
        self._deep_merge(self._storage[feature], params)

        logger.info("FEATURE_PARAMS", f"Updated params for: {feature}")

        return self._deep_copy_params(self._storage[feature])

    def reset_params(self, feature: str) -> Dict[str, Any]:
        """
        Reset parameters to defaults

        Args:
            feature: Feature name

        Returns:
            Reset parameters
        """
        if self._has_feature_defaults(feature):
            self._storage[feature] = self._deep_copy_params(self.DEFAULTS[feature])
            logger.info("FEATURE_PARAMS", f"Reset params for: {feature}")
            return self._deep_copy_params(self._storage[feature])

        logger.warning("FEATURE_PARAMS", f"Cannot reset unknown feature: {feature}")
        raise ValueError(f"Unknown feature: {feature}")

    def get_all_params(self) -> Dict[str, Dict[str, Any]]:
        """Get all feature parameters"""
        return {
            feature: self._deep_copy_params(params) for feature, params in self._storage.items()
        }

    def export_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Export all params as dict for saving"""
        return self._deep_copy_params(self.get_all_params())

    def import_from_dict(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Import params from dict"""
        for feature, params in data.items():
            if self._has_feature_defaults(feature):
                self._deep_merge(self._storage[feature], params)
                logger.info("FEATURE_PARAMS", f"Imported params for: {feature}")
            else:
                logger.warning("FEATURE_PARAMS", f"Skipping unknown feature: {feature}")


# Singleton instance
feature_params_service = FeatureParamsService()
