"""Config Service - Build YAML configuration."""

import copy
from typing import Any, Dict, List

import yaml

from ..models import BuildConfigRequest, Line, ModelConfig, Region
from .feature_params_service import feature_params_service
from .logger_service import logger


class ConfigService:
    """Service to build YOI YAML configuration"""

    _FEATURE_REQUIREMENTS = {
        "region_crowd": ("region",),
        "line_cross": ("line",),
        "dwell_time": ("region",),
    }

    def __init__(self):
        logger.info("CONFIG", "ConfigService initialized")

    def build_yaml(self, request: BuildConfigRequest) -> str:
        """
        Build YAML configuration from request

        Args:
            request: BuildConfigRequest with all config data

        Returns:
            YAML string
        """
        try:
            logger.info("BUILD", f"Building config: {request.config_name}")

            # Validate regions and lines
            self._validate_geometry(request.regions, request.lines, request.feature)

            # Build config dict
            config = {
                "model": [self._build_model(request.model)],
                "feature": [self._map_feature_name(request.feature)],
                "feature_params": self._build_feature_params(
                    request.feature, request.feature_params
                ),
                "regions": [self._build_region(r) for r in request.regions],
                "lines": [self._build_line(line_item) for line_item in request.lines],
                "input": self._build_input(request),
                "output": self._build_output(),
                "metadata": {"cctv_id": request.cctv_id},
            }

            # Convert to YAML
            yaml_str = yaml.dump(
                config, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

            logger.info("BUILD", "Config built successfully")
            return yaml_str

        except Exception as e:
            logger.error("BUILD", f"Failed to build config: {str(e)}")
            raise

    def _validate_geometry(self, regions: List[Region], lines: List[Line], feature: str):
        """Validate regions and lines based on feature"""
        required = self._FEATURE_REQUIREMENTS.get(feature, ())
        if "region" in required and not regions:
            raise ValueError(f"{feature} requires at least one region")
        if "line" in required and not lines:
            raise ValueError(f"{feature} requires at least one line")

        # Validate line has exactly 2 points
        for line in lines:
            if len(line.coords) != 2:
                raise ValueError(f"Line {line.id} must have exactly 2 points")

        # Validate region has at least 3 points
        for region in regions:
            if len(region.coords) < 3:
                raise ValueError(f"Region {region.id} must have at least 3 points")

        logger.info("VALIDATE", f"Geometry validated: {len(regions)} regions, {len(lines)} lines")

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _diff_from_defaults(
        self, current: Dict[str, Any], defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return only values that differ from defaults (deep for nested dicts)."""
        diff: Dict[str, Any] = {}
        for key, value in current.items():
            default_value = defaults.get(key)
            if isinstance(value, dict) and isinstance(default_value, dict):
                nested_diff = self._diff_from_defaults(value, default_value)
                if nested_diff:
                    diff[key] = nested_diff
            else:
                if value != default_value:
                    diff[key] = value
        return diff

    def _build_model(self, model: ModelConfig) -> Dict[str, Any]:
        """Build model configuration"""
        return {
            "name": model.name,
            "device": model.device,
            "conf": model.conf,
            "iou": model.iou,
            "type": model.type,
            "classes": model.classes,
        }

    def _map_feature_name(self, feature: str) -> str:
        """Map frontend feature name to YAML feature name"""
        mapping = {
            "region_crowd": "region_crowd",
            "line_cross": "line_cross",
            "dwell_time": "dwelltime",  # Note: dwelltime in YAML
        }
        return mapping.get(feature, feature)

    def _build_feature_params(self, feature: str, params: dict) -> Dict[str, Any]:
        """
        Build feature parameters

        If params is empty or minimal, will use stored feature_params from service
        """
        yaml_feature = self._map_feature_name(feature)

        # Get default/stored params from feature_params_service
        stored_params = feature_params_service.get_params(feature)

        # Clean None values from request
        cleaned_params = {k: v for k, v in params.items() if v is not None}

        # Merge: request params override stored params (deep merge for nested structures)
        merged_params = copy.deepcopy(stored_params)
        self._deep_merge(merged_params, cleaned_params)

        # Persist only parameters that differ from defaults.
        # This keeps engine defaults as the primary source of truth,
        # and writes YAML only for user-level tuning overrides.
        merged_params = self._diff_from_defaults(merged_params, stored_params)

        logger.info("BUILD", f"Feature params for {feature}: merged request + stored")

        return {yaml_feature: merged_params}

    def _build_region(self, region: Region) -> Dict[str, Any]:
        """Build region geometry with correct order: coords first then id"""
        region_dict = {
            "coords": [{"x": p.x, "y": p.y} for p in region.coords],
            "id": region.id,
            "type": region.type,
            "color": region.color,
            "mode": region.mode if region.mode else [],
        }

        # Add name if provided (custom region name)
        if hasattr(region, "name") and region.name:
            region_dict["name"] = region.name

        return region_dict

    def _build_line(self, line: Line) -> Dict[str, Any]:
        """Build line geometry with centroid and direction"""
        line_dict = {
            "coords": [{"x": p.x, "y": p.y} for p in line.coords],
            "id": line.id,
            "type": line.type,
            "color": line.color,
        }

        # Add name if provided (custom line name)
        if hasattr(line, "name") and line.name:
            line_dict["name"] = line.name

        # Add centroid if provided (for line_cross feature)
        if line.centroid:
            line_dict["centroid"] = {"x": line.centroid.x, "y": line.centroid.y}

        # Add direction if provided (for line_cross feature) - NEW simplified format
        if line.direction:
            line_dict["direction"] = line.direction

        # Add orientation if provided (NEW field)
        if line.orientation:
            line_dict["orientation"] = line.orientation

        # Add bidirectional if provided (NEW field)
        if line.bidirectional is not None:
            line_dict["bidirectional"] = line.bidirectional

        line_dict["mode"] = line.mode if line.mode else []

        return line_dict

    def _build_input(self, request: BuildConfigRequest) -> Dict[str, Any]:
        """Build input configuration"""
        if request.source_mode == "production":
            logger.info("INPUT", f"Production mode: {request.rtsp_url} (RTSP looping)")
            input_config = {
                "source_type": "rtsp",
                "rtsp_url": request.rtsp_url or "rtsp://example.com/stream",
                "reconnect_delay": 5,
                "max_fps": request.max_fps,
            }

            # Add time_allowed if provided (schedule kapan AI aktif)
            if request.time_allowed_start or request.time_allowed_end:
                input_config["time_allowed"] = {}
                if request.time_allowed_start:
                    input_config["time_allowed"]["start_time"] = request.time_allowed_start
                if request.time_allowed_end:
                    input_config["time_allowed"]["end_time"] = request.time_allowed_end
                logger.info(
                    "INPUT",
                    f"Time allowed: {request.time_allowed_start} - {request.time_allowed_end}",
                )

            return input_config
        else:
            logger.info("INPUT", "Inference mode: local video files from input/")

            # Support multiple video files
            if request.video_files:
                video_files = [str(v) for v in request.video_files if str(v).strip()]
            elif request.video_source:
                video_files = [request.video_source]
            else:
                video_files = ["video.mp4"]

            input_config = {
                "source_type": "video",
                "video_files": video_files,
                "max_fps": request.max_fps,
                "input_path": "input",
                "video_inference": {
                    "enabled": True,
                    "dir_path": "input/",
                    "video_filename": [
                        [f, "2026-02-14"] for f in video_files
                    ],  # List of [filename, date]
                },
            }

            return input_config

    def _build_output(self) -> Dict[str, Any]:
        """Build output configuration"""
        output = {
            "output_path": "output",
            "url_dashboard": "0.0.0.0",
            "save_video": {"enabled": True},
            "save_annotations": {"enabled": True},
            "output_format": "annotated_video",
        }
        return output


# Global config service instance
config_service = ConfigService()
