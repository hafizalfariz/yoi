# YOLO Video Inference

**By Hafiz Alfariz**

A simple YOLO-based video inference pipeline that runs object detection on all videos in a folder, with optional tracking and OpenVINO export support.

## Features
- Standard frame-by-frame detection.
- Object tracking mode for smoother results.
- OpenVINO export for optimized inference.
- Progress bar and basic runtime stats.

## Requirements
- Python $\ge 3.9$
- A YOLO model file (`.pt`) or an OpenVINO model directory.

Dependencies are defined in [pyproject.toml](pyproject.toml).

## Project Structure
```
models/
  best.pt
  openvino/
output/
  detect/
  track/
src/
  inference.py
  export_openvino.py
video/
.env
pyproject.toml
```

## Setup
### Option A: Using `uv`
```bash
uv sync
```

### Option B: Using `pip`
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install ultralytics opencv-python python-dotenv rich
```

## Configuration (.env)
The pipeline is configured via [.nv](.env).

| Key | Description | Default |
| --- | --- | --- |
| `MODEL_TYPE` | `pt` or `openvino` | `pt` |
| `MODEL_PT_PATH` | Path to `.pt` model | `models/best.pt` |
| `MODEL_OPENVINO_DIR` | OpenVINO model directory | `models/openvino` |
| `VIDEO_FOLDER` | Input videos folder | `video` |
| `OUTPUT_FOLDER` | Output base folder | `output` |
| `CONF` | Detection confidence | `0.25` |
| `IOU` | Detection IOU threshold | `0.45` |
| `TRACK_CONF` | Tracking confidence | `0.5` |
| `TRACK_IOU` | Tracking IOU threshold | `0.5` |
| `EXPORT_IMGSZ` | Export image size | `640` |
| `EXPORT_HALF` | FP16 export | `false` |

## Usage
### 1) Place your videos
Put your input videos in the [video](video) folder.

### 2) Run inference
```bash
uv run python src/inference.py
```
You will be prompted to choose:
- **1**: Standard Detection (frame by frame)
- **2**: Detection with Tracking

**Outputs**:
- Standard detection results go to [output/detect](output/detect)
- Tracking results go to [output/track](output/track)

### 3) Export to OpenVINO (optional)
```bash
uv run python src/export_openvino.py
```
The OpenVINO model will be saved to [models/openvino](models/openvino).

To run inference with OpenVINO:
```env
MODEL_TYPE=openvino
MODEL_OPENVINO_DIR=models/openvino
```

## Notes
- If no video files are found in [video](video), the script exits gracefully.
- Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv` (case-insensitive).

## Author
Hafiz Alfariz
