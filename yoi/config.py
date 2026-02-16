"""
Configuration classes untuk YOI Vision Engine
Mendukung YAML, JSON, dan programmatic configuration
Supports full complex config structure with features, lines, regions, etc.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


def _normalize_feature_name(value: Any) -> Optional[str]:
    """Normalize feature aliases from builder/user configs.

    Examples:
    - dwelltime -> dwell_time
    - region-crowd -> region_crowd
    - line-cross -> line_cross
    """
    if value is None:
        return None

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "linecross": "line_cross",
        "line_cross": "line_cross",
        "regioncrowd": "region_crowd",
        "region_crowd": "region_crowd",
        "dwelltime": "dwell_time",
        "dwell_time": "dwell_time",
    }
    return aliases.get(normalized, normalized)


@dataclass
class CoordPoint:
    """Single coordinate point (x, y)"""

    x: float
    y: float


@dataclass
class LineConfig:
    """Konfigurasi individual detection line"""

    coords: List[CoordPoint]
    id: int
    type: str  # e.g., "line_1"
    color: str = "#0080ff"
    direction: Optional[str] = None  # e.g., "downward", "rightward"
    orientation: Optional[str] = None  # e.g., "horizontal", "vertical"
    bidirectional: bool = False
    mode: List[Any] = field(default_factory=list)
    centroid: Optional[Dict[str, float]] = None  # Optional centroid {x, y}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineConfig":
        """Create from dict, handling coord points"""
        if isinstance(data.get("coords"), list):
            coords = []
            for c in data["coords"]:
                if isinstance(c, dict):
                    coords.append(CoordPoint(x=c["x"], y=c["y"]))
                elif isinstance(c, CoordPoint):
                    coords.append(c)
            data["coords"] = coords
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RegionConfig:
    """Konfigurasi individual region/polygon"""

    coords: List[CoordPoint]
    id: int
    type: str  # e.g., "region_1"
    color: str = "#00ff00"
    mode: List[Any] = field(default_factory=list)
    centroid: Optional[Dict[str, float]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionConfig":
        """Create from dict"""
        if isinstance(data.get("coords"), list):
            coords = []
            for c in data["coords"]:
                if isinstance(c, dict):
                    coords.append(CoordPoint(x=c["x"], y=c["y"]))
                elif isinstance(c, CoordPoint):
                    coords.append(c)
            data["coords"] = coords
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Konfigurasi YOLO model"""

    name: str = "yolov8n"
    device: str = "cpu"  # cpu, cuda, mps
    conf: float = 0.5
    iou: float = 0.7
    type: Optional[str] = None  # Model size: "small", "medium", dll.
    # Jika kosong, engine akan mencoba membaca otomatis dari
    # metadata model (metadata.yaml) / model.names.
    classes: List[str] = field(default_factory=list)


@dataclass
class VideoInferenceConfig:
    """Konfigurasi video inference settings"""

    enabled: bool = True
    dir_path: str = "input/"
    video_filename: Optional[List[Any]] = None


@dataclass
class VideoInputConfig:
    """Konfigurasi input video/RTSP stream - supports both simple and complex formats"""

    source: Optional[str] = None  # Simple format: direct path/URL
    source_type: Optional[str] = None  # "video", "rtsp", "camera"
    video_source: Optional[str] = None  # Explicit video path
    video_files: Optional[List[str]] = None  # List of video files
    input_path: Optional[str] = None  # Base input directory
    max_fps: Optional[int] = 30
    frame_size: Optional[Tuple[int, int]] = None  # (width, height)
    buffer_size: int = 30  # Frame buffer size
    video_inference: Optional[VideoInferenceConfig] = None

    def _normalize_source_with_input_path(self, source: str) -> str:
        """Normalize source using input_path when source is a bare filename."""
        source_str = str(source).strip()
        if not source_str:
            return source_str
        if source_str.startswith("rtsp://"):
            return source_str

        source_path = Path(source_str)
        if source_path.is_absolute():
            return source_str
        if "/" in source_str or "\\" in source_str:
            return source_str

        base_input = (self.input_path or "").strip()
        if not base_input:
            return source_str
        return str(Path(base_input) / source_str).replace("\\", "/")

    def get_source_paths(self) -> List[str]:
        """Get all configured source paths for sequential inference."""
        sources: List[str] = []

        if self.source:
            sources.append(self._normalize_source_with_input_path(self.source))

        if self.video_source:
            normalized = self._normalize_source_with_input_path(self.video_source)
            if normalized not in sources:
                sources.append(normalized)

        if self.video_files:
            for item in self.video_files:
                normalized = self._normalize_source_with_input_path(str(item))
                if normalized and normalized not in sources:
                    sources.append(normalized)

        if self.video_inference and self.video_inference.video_filename:
            for item in self.video_inference.video_filename:
                candidate: Optional[str] = None
                if isinstance(item, list) and item:
                    candidate = str(item[0])
                elif isinstance(item, str):
                    candidate = item
                if candidate:
                    normalized = self._normalize_source_with_input_path(candidate)
                    if normalized and normalized not in sources:
                        sources.append(normalized)

        return sources

    def get_source_path(self) -> str:
        """Get the actual source path, handling both formats"""
        sources = self.get_source_paths()
        if sources:
            return sources[0]
        raise ValueError("No valid video source found in config")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoInputConfig":
        """Create from dict, handling video_inference"""
        video_infer = data.get("video_inference")
        if isinstance(video_infer, dict):
            data["video_inference"] = VideoInferenceConfig(**video_infer)
        elif video_infer is None:
            data["video_inference"] = None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrackingConfig:
    """Konfigurasi object tracking"""

    max_lost_frames: int = 30
    max_distance: float = 50.0
    tracker_impl: str = "bytetrack"
    bt_track_high_thresh: float = 0.5
    bt_track_low_thresh: float = 0.1
    bt_new_track_thresh: float = 0.6
    bt_match_thresh: float = 0.8
    bt_track_buffer: Optional[int] = None
    bt_fuse_score: bool = True
    reid_enabled: bool = False
    reid_similarity_thresh: float = 0.82
    reid_momentum: float = 0.35
    min_detection_confidence: float = 0.5
    hit_point: str = "centroid"  # centroid, head, bottom
    margin_px: Optional[int] = None


@dataclass
class AlertsConfig:
    """Konfigurasi alerts thresholds"""

    in_warning_threshold: int = 1
    out_warning_threshold: int = 1


@dataclass
class AggregationConfig:
    """Konfigurasi data aggregation"""

    window_seconds: int = 5


@dataclass
class FeatureParamsConfig:
    """Generic feature parameters container"""

    # Line cross specific
    centroid: Optional[str] = None  # "mid_centre", "head", "bottom"
    lost_threshold: Optional[int] = None
    allow_recounting: Optional[bool] = None
    time_allowed: Optional[str] = None
    alert_threshold: Optional[int] = None
    cooldown_seconds: Optional[int] = None

    # Nested configs
    tracking: Optional[Union[Dict, TrackingConfig]] = None
    alerts: Optional[Union[Dict, AlertsConfig]] = None
    aggregation: Optional[Union[Dict, AggregationConfig]] = None

    # Allow arbitrary additional parameters
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Handle tracking and alerts conversion"""
        if isinstance(self.tracking, dict):
            self.tracking = TrackingConfig(**self.tracking)
        if isinstance(self.alerts, dict):
            self.alerts = AlertsConfig(**self.alerts)
        if isinstance(self.aggregation, dict):
            self.aggregation = AggregationConfig(**self.aggregation)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureParamsConfig":
        """Create from dict, capturing extra fields"""
        known_fields = set(cls.__dataclass_fields__.keys()) - {"extra"}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        if extra:
            filtered["extra"] = extra
        return cls(**filtered)


@dataclass
class LogsConfig:
    """Konfigurasi logging"""

    base_dir: str = "logs"
    data_folder: str = "data"
    image_folder: str = "image"
    status_folder: str = "status"
    csv_file: str = "data.csv"
    inference_logs_dir: str = "logs/inference"


@dataclass
class SaveVideoConfig:
    """Konfigurasi save video output"""

    enabled: bool = True


@dataclass
class SaveAnnotationsConfig:
    """Konfigurasi save annotations output"""

    enabled: bool = True


@dataclass
class OutputConfig:
    """Konfigurasi output engines - supports both formats"""

    output_dir: Optional[str] = None  # Simple format
    output_path: Optional[str] = None  # Full format path
    mode: Optional[str] = None  # e.g., "development", "production"
    url_dashboard: Optional[str] = None
    # Optional RTSP output stream (for watching annotated video live)
    # Example: rtsp://localhost:6554/kluis-line
    rtsp_url: Optional[str] = None
    # Optional cooldown/warmup time before starting inference when RTSP is enabled
    # This gives RTSP clients time to connect to the stream first
    rtsp_cooldown_seconds: Optional[int] = None
    output_format: Optional[str] = None  # e.g., "annotated_video"
    # Interval logging progress/frame metrics (default handled by engine)
    log_every_n_frames: Optional[int] = None
    save_video: Optional[Union[bool, SaveVideoConfig]] = None
    save_annotations: Optional[Union[bool, SaveAnnotationsConfig]] = None

    def get_output_dir(self) -> str:
        """Get output directory, handling both formats"""
        if self.output_dir:
            return self.output_dir
        if self.output_path:
            return self.output_path
        return "output"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        """Create from dict"""
        if isinstance(data.get("save_video"), dict):
            data["save_video"] = SaveVideoConfig(**data["save_video"])
        if isinstance(data.get("save_annotations"), dict):
            data["save_annotations"] = SaveAnnotationsConfig(**data["save_annotations"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class YOIConfig:
    """Main configuration untuk YOI Vision Engine - supports full feature-rich format"""

    config_name: str = "default"
    cctv_id: str = "camera_1"
    model: ModelConfig = field(default_factory=ModelConfig)
    feature: Optional[str] = None  # e.g., "line_cross", "dwell_time"
    feature_params: Optional[FeatureParamsConfig] = None
    input: Optional[VideoInputConfig] = None
    output: Optional[OutputConfig] = None
    logs: Optional[LogsConfig] = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    lines: List[LineConfig] = field(default_factory=list)
    regions: List[RegionConfig] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ke dict"""
        data = asdict(self)
        return data

    def to_yaml_str(self) -> str:
        """Convert ke YAML string"""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def to_json_str(self) -> str:
        """Convert ke JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save_yaml(self, path: str) -> None:
        """Save ke YAML file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_yaml_str(), encoding="utf-8")

    def save_json(self, path: str) -> None:
        """Save ke JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json_str(), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str) -> "YOIConfig":
        """Load dari YAML file"""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and not data.get("config_name"):
            data["config_name"] = Path(path).stem
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "YOIConfig":
        """Load dari JSON file"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and not data.get("config_name"):
            data["config_name"] = Path(path).stem
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "YOIConfig":
        """Create dari dict with full feature support"""
        # Model config (support both dict dan list of dicts)
        raw_model = data.get("model", {})
        # Beberapa config lama / builder bisa menyimpan model sebagai list
        # Ambil satu model pertama yang berupa dict (prioritas yang punya "name")
        if isinstance(raw_model, list):
            chosen = None
            for item in raw_model:
                if isinstance(item, dict) and ("name" in item or chosen is None):
                    chosen = item
                    # Kalau sudah ketemu yang punya name, cukup
                    if "name" in item:
                        break
            raw_model = chosen or {}
        elif not isinstance(raw_model, dict):
            # Bentuk lain yang tidak dikenal, fallback ke kosong
            raw_model = {}

        model_cfg = ModelConfig(
            **{k: v for k, v in raw_model.items() if k in ModelConfig.__dataclass_fields__}
        )

        # Input config
        input_data = data.get("input")
        input_cfg = VideoInputConfig.from_dict(input_data) if input_data else None

        # Output config
        output_data = data.get("output")
        output_cfg = OutputConfig.from_dict(output_data) if output_data else OutputConfig()

        # Logs config
        logs_data = data.get("logs")
        logs_cfg = (
            LogsConfig(
                **{k: v for k, v in logs_data.items() if k in LogsConfig.__dataclass_fields__}
            )
            if logs_data
            else None
        )

        # Tracking config
        tracking_data = data.get("tracking", {})
        tracking_cfg = TrackingConfig(
            **{k: v for k, v in tracking_data.items() if k in TrackingConfig.__dataclass_fields__}
        )

        # Normalisasi nama feature (bisa string atau list dari builder)
        raw_feature = data.get("feature")
        if isinstance(raw_feature, list):
            normalized_feature = _normalize_feature_name(raw_feature[0] if raw_feature else None)
        else:
            normalized_feature = _normalize_feature_name(raw_feature)

        # Feature params
        feature_params_data = data.get("feature_params", {})
        feature_params_cfg = None
        if feature_params_data:
            # Jika nested per nama feature, ambil params untuk feature aktif
            feature_name = normalized_feature
            if feature_name and isinstance(feature_params_data, dict):
                matched_params = None
                for raw_key, raw_value in feature_params_data.items():
                    if _normalize_feature_name(raw_key) == feature_name:
                        matched_params = raw_value
                        break
                if isinstance(matched_params, dict):
                    feature_params_cfg = FeatureParamsConfig.from_dict(matched_params)
                elif feature_name in feature_params_data and isinstance(
                    feature_params_data[feature_name], dict
                ):
                    feature_params_cfg = FeatureParamsConfig.from_dict(
                        feature_params_data[feature_name]
                    )
                else:
                    feature_params_cfg = FeatureParamsConfig.from_dict(feature_params_data)
            else:
                feature_params_cfg = FeatureParamsConfig.from_dict(feature_params_data)

        # Lines
        lines = []
        for line_data in data.get("lines", []):
            try:
                lines.append(LineConfig.from_dict(line_data))
            except Exception as e:
                print(f"Warning: Failed to parse line config: {e}")

        # Regions
        regions = []
        for region_data in data.get("regions", []):
            try:
                regions.append(RegionConfig.from_dict(region_data))
            except Exception as e:
                print(f"Warning: Failed to parse region config: {e}")

        return cls(
            config_name=data.get("config_name", "default"),
            cctv_id=data.get("metadata", {}).get("cctv_id", data.get("cctv_id", "camera_1")),
            model=model_cfg,
            feature=normalized_feature,
            feature_params=feature_params_cfg,
            input=input_cfg,
            output=output_cfg,
            logs=logs_cfg,
            tracking=tracking_cfg,
            lines=lines,
            regions=regions,
            metadata=data.get("metadata", {}),
        )
