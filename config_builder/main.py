"""
YOI Config Builder - FastAPI Application
Simple, clean, OOP-based config builder
"""

from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .models import BuildConfigRequest
from .services.config_service import config_service
from .services.feature_params_service import feature_params_service
from .services.logger_service import logger

# Initialize FastAPI app
app = FastAPI(
    title="YOI Config Builder",
    description="Simple builder for YOI detection configs",
    version="1.0.0",
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ===== Health & Info =====


@app.get("/health")
def health():
    """Health check endpoint"""
    logger.info("HEALTH", "Health check requested")
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/features")
def get_features():
    """Get available features with their parameters"""
    features = {
        "region_crowd": {
            "name": "Region Crowd",
            "description": "Count objects inside polygon regions",
            "requires": "regions",
            "parameters": {
                "alert_threshold": {"type": "int", "default": 5, "desc": "Main alert threshold"},
                "cooldown": {"type": "int", "default": 300, "desc": "Cooldown seconds"},
                "warning_threshold": {"type": "int", "default": 5, "desc": "Warning threshold"},
                "critical_threshold": {"type": "int", "default": 10, "desc": "Critical threshold"},
                "time_allowed": {
                    "type": "string",
                    "default": "",
                    "desc": "Operational schedule (HH:MM-HH:MM)",
                },
                "tracking.tracker_impl": {
                    "type": "string",
                    "default": "bytetrack",
                    "desc": "Tracker implementation",
                    "options": ["bytetrack", "centroid"],
                },
                "tracking.max_lost_frames": {
                    "type": "int",
                    "default": 45,
                    "desc": "Max lost frames before track removal",
                },
                "tracking.max_distance": {
                    "type": "float",
                    "default": 140.0,
                    "desc": "Max centroid distance for association",
                },
                "tracking.bt_track_high_thresh": {
                    "type": "float",
                    "default": 0.5,
                    "desc": "ByteTrack high confidence threshold",
                },
                "tracking.bt_track_low_thresh": {
                    "type": "float",
                    "default": 0.1,
                    "desc": "ByteTrack low confidence threshold",
                },
                "tracking.bt_new_track_thresh": {
                    "type": "float",
                    "default": 0.6,
                    "desc": "ByteTrack new-track threshold",
                },
                "tracking.bt_match_thresh": {
                    "type": "float",
                    "default": 0.8,
                    "desc": "ByteTrack matching threshold",
                },
                "tracking.bt_track_buffer": {
                    "type": "int",
                    "default": 45,
                    "desc": "ByteTrack buffer size",
                },
                "tracking.bt_fuse_score": {
                    "type": "boolean",
                    "default": True,
                    "desc": "Fuse IoU and score for matching",
                },
                "tracking.reid_enabled": {
                    "type": "boolean",
                    "default": True,
                    "desc": "Enable appearance ReID",
                },
                "tracking.reid_similarity_thresh": {
                    "type": "float",
                    "default": 0.86,
                    "desc": "ReID cosine similarity threshold",
                },
                "tracking.reid_momentum": {
                    "type": "float",
                    "default": 0.25,
                    "desc": "ReID embedding EMA momentum",
                },
                "tracking.min_detection_confidence": {
                    "type": "float",
                    "default": 0.4,
                    "desc": "Minimum confidence for tracking",
                },
                "tracking.hit_point": {
                    "type": "string",
                    "default": "head",
                    "desc": "Point used for region checks",
                    "options": ["head", "mid_centre", "bottom"],
                },
                "tracking.margin_px": {
                    "type": "int",
                    "default": 20,
                    "desc": "Pixel margin near region border",
                },
            },
        },
        "line_cross": {
            "name": "Line Cross Count",
            "description": "Count in/out crossings on lines",
            "requires": "lines",
            "parameters": {
                "centroid": {
                    "type": "string",
                    "default": "mid_centre",
                    "desc": "Body part for detection",
                    "options": ["mid_centre", "head", "bottom"],
                },
                "lost_threshold": {
                    "type": "int",
                    "default": 10,
                    "desc": "Frames tolerance sebelum tracking reset",
                },
                "allow_recounting": {
                    "type": "boolean",
                    "default": False,
                    "desc": "Allow counting same person/vehicle multiple times",
                },
                "time_allowed": {
                    "type": "string",
                    "default": "",
                    "desc": "Operational schedule (HH:MM-HH:MM)",
                },
                "alert_threshold": {
                    "type": "int",
                    "default": 1,
                    "desc": "Per-interval alert threshold",
                },
                "cooldown_seconds": {
                    "type": "int",
                    "default": 5,
                    "desc": "Cooldown before sending next alert",
                },
                "alerts.in_warning_threshold": {
                    "type": "int",
                    "default": 1,
                    "desc": "Inbound warning threshold",
                },
                "alerts.out_warning_threshold": {
                    "type": "int",
                    "default": 1,
                    "desc": "Outbound warning threshold",
                },
                "aggregation.window_seconds": {
                    "type": "int",
                    "default": 5,
                    "desc": "Aggregation time window",
                },
                "tracking.tracker_impl": {
                    "type": "string",
                    "default": "bytetrack",
                    "desc": "Tracker implementation",
                    "options": ["bytetrack", "centroid"],
                },
                "tracking.max_lost_frames": {
                    "type": "int",
                    "default": 45,
                    "desc": "Max lost frames before track removal",
                },
                "tracking.max_distance": {
                    "type": "float",
                    "default": 140.0,
                    "desc": "Max centroid distance for association",
                },
                "tracking.bt_track_high_thresh": {
                    "type": "float",
                    "default": 0.5,
                    "desc": "ByteTrack high confidence threshold",
                },
                "tracking.bt_track_low_thresh": {
                    "type": "float",
                    "default": 0.1,
                    "desc": "ByteTrack low confidence threshold",
                },
                "tracking.bt_new_track_thresh": {
                    "type": "float",
                    "default": 0.6,
                    "desc": "ByteTrack new-track threshold",
                },
                "tracking.bt_match_thresh": {
                    "type": "float",
                    "default": 0.8,
                    "desc": "ByteTrack matching threshold",
                },
                "tracking.bt_track_buffer": {
                    "type": "int",
                    "default": 45,
                    "desc": "ByteTrack buffer size",
                },
                "tracking.bt_fuse_score": {
                    "type": "boolean",
                    "default": True,
                    "desc": "Fuse IoU and score for matching",
                },
                "tracking.reid_enabled": {
                    "type": "boolean",
                    "default": True,
                    "desc": "Enable appearance ReID",
                },
                "tracking.reid_similarity_thresh": {
                    "type": "float",
                    "default": 0.86,
                    "desc": "ReID cosine similarity threshold",
                },
                "tracking.reid_momentum": {
                    "type": "float",
                    "default": 0.25,
                    "desc": "ReID embedding EMA momentum",
                },
                "tracking.min_detection_confidence": {
                    "type": "float",
                    "default": 0.4,
                    "desc": "Minimum confidence for tracking",
                },
                "tracking.hit_point": {
                    "type": "string",
                    "default": "head",
                    "desc": "Point used for line intersection",
                    "options": ["head", "mid_centre", "bottom"],
                },
                "tracking.margin_px": {
                    "type": "int",
                    "default": 20,
                    "desc": "Pixel margin near line",
                },
            },
        },
        "dwell_time": {
            "name": "Dwell Time",
            "description": "Measure how long objects stay in regions",
            "requires": "regions",
            "parameters": {
                "alert_threshold": {
                    "type": "int",
                    "default": 10,
                    "desc": "Main alert threshold (seconds)",
                },
                "cooldown": {"type": "int", "default": 120, "desc": "Cooldown seconds"},
                "warning_seconds": {"type": "int", "default": 10, "desc": "Warning duration"},
                "critical_seconds": {"type": "int", "default": 20, "desc": "Critical duration"},
                "min_dwelltime": {"type": "int", "default": 15, "desc": "Minimum valid dwell"},
                "lost_threshold": {"type": "int", "default": 10, "desc": "Tracking lost tolerance"},
                "time_allowed": {
                    "type": "string",
                    "default": "",
                    "desc": "Operational schedule (HH:MM-HH:MM)",
                },
                "tracking.tracker_impl": {
                    "type": "string",
                    "default": "bytetrack",
                    "desc": "Tracker implementation",
                    "options": ["bytetrack", "centroid"],
                },
                "tracking.max_lost_frames": {
                    "type": "int",
                    "default": 60,
                    "desc": "Max lost frames before track removal",
                },
                "tracking.max_distance": {
                    "type": "float",
                    "default": 120,
                    "desc": "Max centroid distance for association",
                },
                "tracking.bt_track_high_thresh": {
                    "type": "float",
                    "default": 0.5,
                    "desc": "ByteTrack high confidence threshold",
                },
                "tracking.bt_track_low_thresh": {
                    "type": "float",
                    "default": 0.1,
                    "desc": "ByteTrack low confidence threshold",
                },
                "tracking.bt_new_track_thresh": {
                    "type": "float",
                    "default": 0.6,
                    "desc": "ByteTrack new-track threshold",
                },
                "tracking.bt_match_thresh": {
                    "type": "float",
                    "default": 0.8,
                    "desc": "ByteTrack matching threshold",
                },
                "tracking.bt_track_buffer": {
                    "type": "int",
                    "default": 60,
                    "desc": "ByteTrack buffer size",
                },
                "tracking.bt_fuse_score": {
                    "type": "boolean",
                    "default": True,
                    "desc": "Fuse IoU and score for matching",
                },
                "tracking.reid_enabled": {
                    "type": "boolean",
                    "default": True,
                    "desc": "Enable appearance ReID",
                },
                "tracking.reid_similarity_thresh": {
                    "type": "float",
                    "default": 0.86,
                    "desc": "ReID cosine similarity threshold",
                },
                "tracking.reid_momentum": {
                    "type": "float",
                    "default": 0.25,
                    "desc": "ReID embedding EMA momentum",
                },
                "tracking.min_detection_confidence": {
                    "type": "float",
                    "default": 0.25,
                    "desc": "Minimum confidence for tracking",
                },
            },
        },
    }

    logger.info("FEATURES", "Features catalog retrieved")
    return {"features": features}


# ===== Feature Parameters Management =====


@app.get("/api/feature-params")
def get_all_feature_params():
    """Get all feature parameters for all features"""
    logger.info("FEATURE_PARAMS", "Retrieving all feature parameters")
    params = feature_params_service.get_all_params()
    return {"feature_params": params}


@app.get("/api/feature-params/{feature}")
def get_feature_params(feature: str):
    """
    Get current feature parameters for a specific feature

    Args:
        feature: Feature name (line_cross, region_crowd, dwell_time)
    """
    try:
        logger.info("FEATURE_PARAMS", f"GET feature params: {feature}")
        params = feature_params_service.get_params(feature)
        return {"feature": feature, "params": params, "timestamp": logger.get_current_timestamp()}
    except Exception as e:
        logger.error("FEATURE_PARAMS", f"Error getting params for {feature}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/feature-params/{feature}")
def update_feature_params(feature: str, body: dict):
    """
    Update feature parameters for a specific feature

    Args:
        feature: Feature name
        body: Dictionary with new parameters (will merge with existing)
    """
    try:
        logger.info("FEATURE_PARAMS", f"UPDATE feature params: {feature}")
        logger.info("FEATURE_PARAMS", f"  New values: {body}")

        updated_params = feature_params_service.update_params(feature, body)

        return {
            "feature": feature,
            "params": updated_params,
            "timestamp": logger.get_current_timestamp(),
            "message": f"Successfully updated {feature} parameters",
        }
    except ValueError as e:
        logger.error("FEATURE_PARAMS", f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("FEATURE_PARAMS", f"Error updating params: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/feature-params/{feature}")
def reset_feature_params(feature: str):
    """
    Reset feature parameters to defaults

    Args:
        feature: Feature name
    """
    try:
        logger.info("FEATURE_PARAMS", f"RESET feature params: {feature}")
        reset_params = feature_params_service.reset_params(feature)

        return {
            "feature": feature,
            "params": reset_params,
            "timestamp": logger.get_current_timestamp(),
            "message": f"Successfully reset {feature} parameters to defaults",
        }
    except ValueError as e:
        logger.error("FEATURE_PARAMS", f"Unknown feature: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("FEATURE_PARAMS", f"Error resetting params: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Config Building =====


@app.post("/api/parse")
async def parse_yaml_config(request: Request):
    """
    Parse YAML config file and return structured data

    Used for loading existing configs
    """
    try:
        logger.info("PARSE", "Parsing uploaded YAML config")

        # Read raw body as text
        yaml_content = await request.body()
        yaml_text = yaml_content.decode("utf-8")

        # Parse YAML
        config_dict = yaml.safe_load(yaml_text)

        if not config_dict:
            raise ValueError("Empty YAML file")

        logger.info("PARSE", "YAML parsed successfully")
        return config_dict

    except yaml.YAMLError as e:
        logger.error("PARSE", f"YAML parse error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")
    except Exception as e:
        logger.error("PARSE", f"Parse error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/build")
def build_config(request: BuildConfigRequest):
    """
    Build YAML configuration

    Returns YAML string for preview
    """
    try:
        logger.info("API", f"Build request received: {request.config_name}")
        yaml_str = config_service.build_yaml(request)
        return PlainTextResponse(content=yaml_str, media_type="text/yaml")

    except ValueError as e:
        logger.error("API", f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("API", f"Build error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to build config: {str(e)}")


@app.post("/api/save")
def save_config(request: BuildConfigRequest):
    """
    Save YAML configuration to file

    Returns filename and download link
    """
    try:
        logger.info("SAVE", f"Saving config: {request.config_name}")

        # Build YAML
        yaml_str = config_service.build_yaml(request)

        # Save to configs/app directory
        output_dir = Path(__file__).parent.parent / "configs" / "app"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{request.config_name}.yaml"
        filepath = output_dir / filename

        filepath.write_text(yaml_str, encoding="utf-8")

        logger.info("SAVE", f"Config saved: {filename}")

        return {
            "success": True,
            "filename": filename,
            "path": str(filepath),
            "download_url": f"/api/download/{filename}",
        }

    except Exception as e:
        logger.error("SAVE", f"Save error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@app.get("/api/download/{filename}")
def download_config(filename: str):
    """Download saved config file"""
    filepath = Path(__file__).parent.parent / "configs" / "app" / filename

    if not filepath.exists():
        logger.error("DOWNLOAD", f"File not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info("DOWNLOAD", f"Downloading: {filename}")
    return FileResponse(path=str(filepath), filename=filename, media_type="application/x-yaml")


# ===== Logging =====


@app.get("/api/logs")
def get_logs(count: int = 50):
    """Get recent logs"""
    logs = logger.get_recent_logs(count)
    return {"logs": [log.dict() for log in logs]}


@app.delete("/api/logs")
def clear_logs():
    """Clear all logs"""
    logger.clear()
    return {"success": True, "message": "Logs cleared"}


# ===== UI =====


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve main UI"""
    template_path = Path(__file__).parent / "templates" / "index.html"

    if not template_path.exists():
        logger.error("UI", "Template not found")
        return HTMLResponse("<h1>Error</h1><p>Template not found</p>", status_code=500)

    html = template_path.read_text(encoding="utf-8")
    logger.info("UI", "Main page loaded")
    return HTMLResponse(content=html)


# ===== Startup =====


@app.on_event("startup")
def startup_event():
    """Application startup"""
    logger.info("SYSTEM", "=== YOI Config Builder Started ===")
    logger.info("SYSTEM", "Version: 1.0.0")
    logger.info("SYSTEM", "Ready to build configs!")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8030)
