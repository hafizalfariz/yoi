"""
Pydantic models for YOI Config Builder
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ===== Model Configuration =====


class ModelConfig(BaseModel):
    name: str = "person_360camera_detection_v33"
    device: Literal["cpu", "gpu"] = "cpu"
    conf: float = Field(0.5, ge=0.0, le=1.0)
    iou: float = Field(0.7, ge=0.0, le=1.0)
    type: Literal["small", "medium", "large"] = "small"
    classes: List[str] = Field(default_factory=lambda: ["person"])


# ===== Feature Configuration =====


class FeatureParams(BaseModel):
    """Feature-specific parameters"""

    # Common
    alert_threshold: Optional[int] = None
    cooldown: Optional[int] = None

    # Region crowd specific
    warning_threshold: Optional[int] = None
    critical_threshold: Optional[int] = None

    # Line cross specific
    direction: Optional[dict] = None  # {axis, in, out}
    tracking: Optional[dict] = (
        None  # {tracker_impl, bt_*, max_lost_frames, max_distance, hit_point, margin_px}
    )
    alerts: Optional[dict] = None  # {in_warning_threshold, out_warning_threshold}
    aggregation: Optional[dict] = None  # {window_seconds}

    # Dwell time specific
    warning_seconds: Optional[int] = None
    critical_seconds: Optional[int] = None
    min_dwelltime: Optional[int] = None
    lost_threshold: Optional[int] = None


# ===== Geometry =====


class Point(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate (0-1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate (0-1)")


class Region(BaseModel):
    coords: List[Point]  # First: coordinates
    id: int
    type: str
    name: Optional[str] = None  # Custom region name (e.g., "Area Parkir", "Lobi Utama")
    color: str
    mode: List[str] = Field(default_factory=list)


class Line(BaseModel):
    coords: List[Point]  # Exactly 2 points: start and end
    id: int
    type: str
    name: Optional[str] = None  # Custom line name (e.g., "Entrance Line", "Exit Gate")
    color: str

    # Line cross specific
    centroid: Optional[Point] = None  # Center point of line
    direction: Optional[str] = (
        None  # NEW: Simple string like "downward", "rightward", etc. (SIMPLIFIED format)
    )
    orientation: Optional[str] = None  # NEW: "horizontal", "vertical", or "diagonal"
    bidirectional: Optional[bool] = None  # NEW: Whether line is bidirectional

    mode: List[str] = Field(default_factory=list)


# ===== Input Configuration =====


class InputConfig(BaseModel):
    # Production mode (RTSP) - Looping stream
    rtsp_url: Optional[str] = None
    reconnect_delay: Optional[int] = 5  # Reconnect delay in seconds
    time_allowed: Optional[dict] = None  # {start_time, end_time} for time-based filtering

    # Inference mode (Video files) - No looping, can process multiple videos
    video_source: Optional[str] = None  # Primary video (for backward compatibility)
    video_files: Optional[List[str]] = Field(
        default_factory=list
    )  # Multiple video files to process
    max_fps: int = 30
    video_inference: Optional[dict] = None  # {enabled, dir_path, video_filename}


# ===== Output Configuration =====


class OutputConfig(BaseModel):
    url_dashboard: str = "0.0.0.0"
    save_video: dict = Field(default_factory=lambda: {"enabled": True})


# ===== Full Config =====


class YOIConfig(BaseModel):
    """Complete YOI configuration"""

    config_name: str = "untitled"
    model: List[ModelConfig]
    feature: List[str]
    feature_params: dict  # {feature_name: FeatureParams}
    regions: List[Region] = Field(default_factory=list)
    lines: List[Line] = Field(default_factory=list)
    input: InputConfig
    output: OutputConfig
    metadata: dict = Field(default_factory=dict)


# ===== API Request Models =====


class BuildConfigRequest(BaseModel):
    """Request to build YAML config"""

    config_name: str
    model: ModelConfig
    feature: Literal["region_crowd", "line_cross", "dwell_time"]
    feature_params: dict
    regions: List[Region] = Field(default_factory=list)
    lines: List[Line] = Field(default_factory=list)
    source_mode: Literal["production", "inference"] = "inference"
    rtsp_url: Optional[str] = None
    video_source: Optional[str] = None
    max_fps: int = 30
    cctv_id: str = "office"

    # Time allowed (for production mode - schedule jika AI aktif)
    time_allowed_start: Optional[str] = None  # Format: HH:MM:SS (contoh: 07:00:00)
    time_allowed_end: Optional[str] = None  # Format: HH:MM:SS (contoh: 17:00:00)

    video_files: Optional[List[str]] = Field(
        default=None,
        description="Optional list of video filenames for sequential inference",
    )


class LogMessage(BaseModel):
    """Log message for frontend"""

    timestamp: str
    level: Literal["INFO", "WARNING", "ERROR"]
    category: str
    message: str
