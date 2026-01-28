from ultralytics import YOLO
import cv2
import os
import shutil
from pathlib import Path
import traceback
from typing import Optional
from dotenv import load_dotenv
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console

console = Console()

WINDOW_NAME = "YOLO Inference"

try:
    import msvcrt
except ImportError:
    msvcrt = None


def _patch_posixpath_for_windows():
    """Patch pathlib to allow loading PosixPath pickles on Windows."""
    if os.name != "nt":
        return
    import pathlib
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass


def _is_openvino_available() -> bool:
    try:
        import openvino  # noqa: F401
    except Exception:
        return False
    return True

def load_config():
    """Load configuration from .env file."""
    load_dotenv()
    return {
        "model_type": os.getenv("MODEL_TYPE", "auto").lower(),
        "model_pt_path": os.getenv("MODEL_PT_PATH", "models/pt/best.pt"),
        "model_openvino_dir": os.getenv("MODEL_OPENVINO_DIR", "models/openvino"),
        "export_imgsz": int(os.getenv("EXPORT_IMGSZ", "640")),
        "export_half": os.getenv("EXPORT_HALF", "false").lower() == "true",
        "video_folder": os.getenv("VIDEO_FOLDER", "vid"),
        "output_folder": os.getenv("OUTPUT_FOLDER", "output"),
        "conf": float(os.getenv("CONF", "0.25")),
        "iou": float(os.getenv("IOU", "0.45")),
        "track_conf": float(os.getenv("TRACK_CONF", "0.5")),
        "track_iou": float(os.getenv("TRACK_IOU", "0.5")),
        "tracker_config": os.getenv("TRACKER_CONFIG", "botsort.yaml"),
        "show_live": os.getenv("SHOW_LIVE", "false").lower() == "true",
        "show_live_width": int(os.getenv("SHOW_LIVE_WIDTH", "0")),
        "show_live_height": int(os.getenv("SHOW_LIVE_HEIGHT", "0")),
        "show_live_fullscreen": os.getenv("SHOW_LIVE_FULLSCREEN", "false").lower() == "true",
        "show_live_stop_on_close": os.getenv("SHOW_LIVE_STOP_ON_CLOSE", "true").lower() == "true",
    }


def _get_fourcc_for_suffix(suffix: str) -> int:
    """Return a codec FourCC based on video file suffix."""
    suffix = suffix.lower()
    codec_map = {
        ".mp4": "mp4v",
        ".mov": "mp4v",
        ".mkv": "mp4v",
        ".avi": "XVID",
    }
    codec = codec_map.get(suffix, "mp4v")
    return cv2.VideoWriter_fourcc(*codec)


def _setup_live_window(config):
    """Create and configure the live preview window."""
    if not config["show_live"]:
        return

    window_name = WINDOW_NAME
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if config["show_live_fullscreen"]:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        width = config["show_live_width"]
        height = config["show_live_height"]
        if width > 0 and height > 0:
            cv2.resizeWindow(window_name, width, height)


def _poll_key():
    """Non-blocking key polling (Windows). Returns lowercase character or None."""
    if msvcrt is None:
        return None
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getch()
    if not key:
        return None
    if key in (b"\x00", b"\xe0"):
        msvcrt.getch()
        return None
    try:
        return key.decode("utf-8").lower()
    except UnicodeDecodeError:
        return None
def _is_openvino_available() -> bool:
    try:
        import openvino  # noqa: F401
    except Exception:
        return False
    return True


def _find_openvino_model(openvino_dir: str, preferred_stem: Optional[str] = None) -> Optional[Path]:
    path = Path(openvino_dir)
    if not path.exists() or not path.is_dir():
        return None
    # Prefer layout: <model_name>/1/*.xml
    xml_files = sorted(path.rglob("1/*.xml"))
    if not xml_files:
        xml_files = sorted(path.rglob("*.xml"))
    if not xml_files:
        return None
    if preferred_stem:
        for xml in xml_files:
            if xml.stem == preferred_stem:
                return xml
    if len(xml_files) == 1:
        return xml_files[0]
    return xml_files[0]


def _export_openvino_from_pt(
    pt_path: Path,
    openvino_root: Path,
    imgsz: int,
    half: bool,
    output_name: Optional[str] = None,
) -> Optional[Path]:
    if not _is_openvino_available():
        console.print("OpenVINO is not installed. Skipping auto-export.")
        return None
    if not pt_path.exists():
        console.print(f"Error: .pt model not found: {pt_path}")
        return None

    openvino_root.mkdir(parents=True, exist_ok=True)
    export_name = output_name or pt_path.stem
    console.print(f"Auto-exporting {pt_path.name} to OpenVINO ({export_name}/1)...")
    try:
        model = YOLO(str(pt_path))
        export_result = model.export(
            format="openvino",
            imgsz=imgsz,
            half=half,
            project=str(openvino_root / export_name),
            name="1",
            exist_ok=True,
        )
        console.print(f"[debug] Ultralytics export_result: {export_result}")
    except Exception as e:
        console.print(f"[red]OpenVINO export failed: {e}[/red]")
        traceback.print_exc()
        return None
    exported_dir = openvino_root / export_name / "1"
    # If Ultralytics exports elsewhere, move the artifacts into <name>/1
    if export_result:
        export_path = Path(export_result)
        export_dir = export_path if export_path.is_dir() else export_path.parent
        if export_dir.resolve() != exported_dir.resolve():
            exported_dir.mkdir(parents=True, exist_ok=True)
            for item in export_dir.iterdir():
                target = exported_dir / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            try:
                if export_dir.exists() and not any(export_dir.iterdir()):
                    export_dir.rmdir()
            except OSError:
                pass
    if not list(exported_dir.rglob("*.xml")):
        console.print("OpenVINO export produced no .xml files. Export may have failed.")
        return None
    return exported_dir


def resolve_model_path(config):
    """Resolve model path based on MODEL_TYPE (pt/openvino/auto)."""
    model_type = config["model_type"]
    if model_type == "openvino":
        preferred_stem = Path(config["model_pt_path"]).stem
        model = _find_openvino_model(config["model_openvino_dir"], preferred_stem)
        if model is None:
            exported_dir = _export_openvino_from_pt(
                Path(config["model_pt_path"]),
                Path(config["model_openvino_dir"]),
                config["export_imgsz"],
                config["export_half"],
            )
            if exported_dir is not None:
                config["model_openvino_dir"] = str(exported_dir)
                model = _find_openvino_model(config["model_openvino_dir"], preferred_stem)
        if model is not None:
            return str(model), "openvino"
        console.print("OpenVINO model not found. Falling back to .pt.")
        return config["model_pt_path"], "pt"
    if model_type == "pt":
        return config["model_pt_path"], "pt"

    # Auto: prefer OpenVINO when available and model exists, else fallback to PyTorch
    openvino_dir = Path(config["model_openvino_dir"])
    preferred_stem = Path(config["model_pt_path"]).stem
    model = _find_openvino_model(str(openvino_dir), preferred_stem)
    if model is None:
        exported_dir = _export_openvino_from_pt(
            Path(config["model_pt_path"]),
            openvino_dir,
            config["export_imgsz"],
            config["export_half"],
        )
        if exported_dir is not None:
            config["model_openvino_dir"] = str(exported_dir)
            model = _find_openvino_model(str(openvino_dir), preferred_stem)
    if model is not None:
        return str(model), "openvino"
    return config["model_pt_path"], "pt"


def run_inference(config):
    """Inference script using YOLOv11s. Configuration is loaded from .env."""
    
    model_path, model_type = resolve_model_path(config)
    video_folder = config["video_folder"]
    output_folder = config["output_folder"]
    detect_folder = os.path.join(output_folder, "detect")
    
    # Create output folder if it does not exist
    os.makedirs(detect_folder, exist_ok=True)

    show_live = config["show_live"]
    if show_live:
        _setup_live_window(config)
    
    # Load YOLO model
    if not os.path.exists(model_path):
        console.print(f"Error: Model path not found: {model_path}")
        return

    console.print(f"Loading model ({model_type}) from [bold]{model_path}[/bold]...")
    _patch_posixpath_for_windows()
    model = YOLO(model_path)
    
    # Collect all video files in the input folder (dedupe on case-insensitive filesystems)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [
        p
        for p in Path(video_folder).glob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]
    video_files = sorted({p.resolve() for p in video_files})
    
    if not video_files:
        console.print(f"No video files found in folder {video_folder}")
        return
    
    console.print(f"Found [bold]{len(video_files)}[/bold] video(s) to process")
    
    # Process each video
    for video_path in video_files:
        console.print(f"\n{'='*60}")
        console.print(f"Processing: [bold]{video_path.name}[/bold]")
        console.print(f"{'='*60}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            console.print(f"Error: Unable to open video {video_path.name}")
            continue
        
        # Read video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        console.print(f"Video Info: {width}x{height} @ {fps}fps, Total frames: {total_frames}")
        
        # Setup video writer for output (preserve original extension)
        output_name = f"detected_{video_path.stem}{video_path.suffix}"
        output_path = os.path.join(detect_folder, output_name)
        fourcc = _get_fourcc_for_suffix(video_path.suffix)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        # Process frame by frame with a progress bar
        progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} frames"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task_id = progress.add_task(video_path.name, total=total_frames)
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Run inference
                results = model(frame, conf=config["conf"], iou=config["iou"], verbose=False)

                # Draw detections on the frame
                annotated_frame = results[0].plot()

                key = _poll_key()
                if key == "q":
                    return
                if key == "v":
                    show_live = not show_live
                    if show_live:
                        _setup_live_window(config)
                    else:
                        cv2.destroyAllWindows()

                if show_live:
                    cv2.imshow(WINDOW_NAME, annotated_frame)
                    cv2.waitKey(1)
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        show_live = False

                # Write frame to output
                out.write(annotated_frame)

                count = 0
                label_summary = ""
                if getattr(results[0], "boxes", None) is not None:
                    count = len(results[0].boxes)
                    names = getattr(results[0], "names", {})
                    cls_ids = results[0].boxes.cls
                    if cls_ids is not None and len(cls_ids) > 0:
                        unique_ids = sorted(set(int(c) for c in cls_ids.tolist()))
                        labels = [str(names.get(i, i)) for i in unique_ids]
                        label_summary = ", ".join(labels)
                inference_ms = None
                if getattr(results[0], "speed", None) is not None:
                    inference_ms = results[0].speed.get("inference")
                detail = f"{video_path.name} | {count} object(s)"
                if label_summary:
                    detail += f" | {label_summary}"
                if inference_ms is not None:
                    detail += f" | {inference_ms:.1f} ms"
                progress.update(task_id, advance=1, description=detail)
        
        # Release resources
        cap.release()
        out.release()
        if show_live:
            cv2.destroyAllWindows()
        
        console.print(f"\nDone! Output saved to: {output_path}")
        
        # Detection stats
        console.print(f"\nDetection stats for {video_path.name}:")
        console.print(f"Total frames processed: {frame_count}")
    
    console.print(f"\n{'='*60}")
    console.print("All videos have been processed!")
    console.print(f"{'='*60}")

def run_inference_with_tracking(config):
    """Inference with tracking for smoother results."""
    
    model_path, model_type = resolve_model_path(config)
    video_folder = config["video_folder"]
    output_folder = config["output_folder"]
    track_folder = os.path.join(output_folder, "track")
    
    os.makedirs(track_folder, exist_ok=True)

    show_live = config["show_live"]
    if show_live:
        _setup_live_window(config)
    
    if not os.path.exists(model_path):
        console.print(f"Error: Model path not found: {model_path}")
        return

    console.print(f"Loading model ({model_type}) from [bold]{model_path}[/bold]...")
    _patch_posixpath_for_windows()
    model = YOLO(model_path)
    
    # Collect all video files (dedupe on case-insensitive filesystems)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [
        p
        for p in Path(video_folder).glob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]
    video_files = sorted({p.resolve() for p in video_files})
    
    if not video_files:
        console.print(f"No video files found in folder {video_folder}")
        return
    
    console.print(f"Found [bold]{len(video_files)}[/bold] video(s) to process with tracking")
    
    for video_path in video_files:
        console.print(f"\n{'='*60}")
        console.print(f"Processing with tracking: [bold]{video_path.name}[/bold]")
        console.print(f"{'='*60}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            console.print(f"Error: Unable to open video {video_path.name}")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        output_name = f"tracked_{video_path.stem}{video_path.suffix}"
        output_path = os.path.join(track_folder, output_name)
        fourcc = _get_fourcc_for_suffix(video_path.suffix)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} frames"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task_id = progress.add_task(video_path.name, total=total_frames)
            results_stream = model.track(
                source=str(video_path),
                conf=config["track_conf"],
                iou=config["track_iou"],
                tracker="bytetrack.yaml",
                show=False,
                save=False,
                verbose=False,
                stream=True,
            )

            for r in results_stream:
                key = _poll_key()
                if key == "q":
                    return
                if key == "v":
                    show_live = not show_live
                    if show_live:
                        _setup_live_window(config)
                    else:
                        cv2.destroyAllWindows()

                annotated_frame = r.plot()

                if show_live:
                    cv2.imshow(WINDOW_NAME, annotated_frame)
                    cv2.waitKey(1)
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        show_live = False

                out.write(annotated_frame)
                count = 0
                label_summary = ""
                if getattr(r, "boxes", None) is not None:
                    count = len(r.boxes)
                    names = getattr(r, "names", {})
                    cls_ids = r.boxes.cls
                    if cls_ids is not None and len(cls_ids) > 0:
                        unique_ids = sorted(set(int(c) for c in cls_ids.tolist()))
                        labels = [str(names.get(i, i)) for i in unique_ids]
                        label_summary = ", ".join(labels)
                inference_ms = None
                if getattr(r, "speed", None) is not None:
                    inference_ms = r.speed.get("inference")
                detail = f"{video_path.name} | {count} object(s)"
                if label_summary:
                    detail += f" | {label_summary}"
                if inference_ms is not None:
                    detail += f" | {inference_ms:.1f} ms"
                progress.update(task_id, advance=1, description=detail)

        if show_live:
            cv2.destroyAllWindows()

        out.release()

        console.print(f"Done! Tracking results saved to: {output_path}")

if __name__ == "__main__":
    console.print("=" * 60)
    console.print("Inference - YOLOv11s")
    console.print("=" * 60)
    
    config = load_config()
    
    # Portability warning before mode selection
    console.print("\n[bold yellow]Warning:[/bold yellow] Model portability notice")
    console.print(
        "- Model .pt can be OS-specific (e.g., saved on Linux -> PosixPath) and may fail on Windows."
    )
    console.print(
        "- For cross-platform inference (Windows/Linux/macOS), prefer OpenVINO/ONNX models."
    )
    console.print(
        "- Auto mode will prefer OpenVINO when available; otherwise it falls back to .pt."
    )
    console.print("- If OpenVINO model is missing, it will auto-export from the .pt model.")
    console.print("- You can switch by setting MODEL_TYPE in .env (auto/pt/openvino).\n")

    export_choice = input("Export .pt to OpenVINO now? (y/N): ").strip().lower()
    if export_choice == "y":
        default_name = Path(config["model_pt_path"]).stem
        export_root = Path(config["model_openvino_dir"])

        while True:
            folder_name = (
                input(f"OpenVINO output folder name [default: {default_name}]: ").strip()
                or default_name
            )
            target_parent = export_root / folder_name
            target_export = target_parent / "1"
            has_existing = target_parent.exists()
            if target_export.exists() and any(target_export.iterdir()):
                has_existing = True
            if has_existing:
                overwrite = input(
                    f"Folder '{folder_name}' already exists. Overwrite? (y/N): "
                ).strip().lower()
                if overwrite == "y":
                    shutil.rmtree(target_parent, ignore_errors=True)
                    break
                rename = input("Use a different name? (y/N): ").strip().lower()
                if rename == "y":
                    continue
                console.print("OpenVINO export cancelled.")
                folder_name = None
                break
            break

        if folder_name is None:
            exported_dir = None
        else:
            exported_dir = _export_openvino_from_pt(
                Path(config["model_pt_path"]),
                export_root,
                config["export_imgsz"],
                config["export_half"],
                output_name=folder_name,
            )
        if exported_dir is not None:
            config["model_openvino_dir"] = str(exported_dir)
            console.print(f"OpenVINO saved to: {exported_dir}")
        else:
            console.print("OpenVINO export skipped or failed.")

    console.print("Select inference mode:")
    console.print("1. Standard Detection (frame by frame)")
    console.print("2. Detection with Tracking (object tracking)")
    
    choice = input("\nChoose mode (1/2) [default: 1]: ").strip() or "1"
    
    if choice == "2":
        run_inference_with_tracking(config)
    else:
        run_inference(config)
    
    console.print("\nInference completed!")
