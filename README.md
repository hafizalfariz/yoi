# YOLO Video Inference

**Version: v1 (1.0.0)**

**By Hafiz Alfariz**

A simple YOLO-based video inference pipeline that runs object detection on all videos in a folder, with optional tracking and OpenVINO export support.

## Features
- Standard frame-by-frame detection.
- Object tracking mode for smoother results.
- OpenVINO export for optimized inference.
- Progress bar and basic runtime stats.

## Requirements
- Python $\ge 3.9$
- You only need a YOLO `.pt` model file. OpenVINO models will be auto-generated as needed.
- All dependencies, including OpenVINO, are installed automatically with `uv sync` (see [pyproject.toml](pyproject.toml)).

## Project Structure
```
models/
  pt/
    best.pt
  openvino/
    <your_model_name>/
      1/
        best.xml
        ...
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
### Setup
```bash
uv sync
```
This will install all required dependencies, including OpenVINO. No manual pip install is needed.

## Configuration (.env)
The pipeline is configured via [.nv](.env).

| Key | Description | Default |
| --- | --- | --- |
| `MODEL_TYPE` | `pt` or `openvino` | `pt` |
| `MODEL_PT_PATH` | Path to `.pt` model | `models/pt/best.pt` |
| `MODEL_OPENVINO_DIR` | OpenVINO model directory | `models/openvino` |
| `VIDEO_FOLDER` | Input videos folder | `video` |
| `OUTPUT_FOLDER` | Output base folder | `output` |
| `CONF` | Detection confidence | `0.25` |
| `IOU` | Detection IOU threshold | `0.45` |
| `TRACK_CONF` | Tracking confidence | `0.5` |
| `TRACK_IOU` | Tracking IOU threshold | `0.5` |
| `SHOW_LIVE` | Enable live preview window | `true` |
| `SHOW_LIVE_WIDTH` | Preview window width (0 = auto) | `1280` |
| `SHOW_LIVE_HEIGHT` | Preview window height (0 = auto) | `720` |
| `SHOW_LIVE_FULLSCREEN` | Fullscreen preview | `false` |
| `SHOW_LIVE_STOP_ON_CLOSE` | Stop processing when preview closed | `true` |
| `EXPORT_IMGSZ` | Export image size | `640` |
| `EXPORT_HALF` | FP16 export | `false` |

## Usage
### 1) Place your videos
Put your input videos in the [video](video) folder.


### 2) Run inference & OpenVINO export
```bash
uv run python src/inference.py
```
When you run the script, you will be prompted:
- Do you want to export .pt to OpenVINO? (y/N)
  - If yes, you can specify the output folder name (e.g. `person_360camera_detection`).
  - If the folder already exists, you will get a warning to overwrite or rename.
  - The export result will be placed in `models/openvino/<yourfolder>/1/`.
- Choose mode:
  - **1**: Standard Detection (frame by frame)
  - **2**: Detection with Tracking

**Output:**
- Standard detection: [output/detect](output/detect)
- Tracking: [output/track](output/track)

**Live preview controls:**
- Press **v** to show/hide the preview window.
- Press **q** to stop.


### 3) Model detection & auto selection
- The script will automatically detect OpenVINO models inside subfolder `models/openvino/<yourfolder>/1/`.
- If the OpenVINO model does not exist, it will be auto-exported from `.pt`.
- You can use `auto` mode (default) in `.env` so the script will pick the best model for your device.


## Notes
- `.pt` models and OpenVINO export results are not uploaded to the repo (only the folder structure is included).
- The script will always look for OpenVINO models in the `<yourfolder>/1/` subfolder.
- If the OpenVINO output folder already exists, you will be prompted to overwrite or rename.
- Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`.
- Preview hotkeys: **v** (toggle), **q** (stop).

## Author
Hafiz Alfariz
