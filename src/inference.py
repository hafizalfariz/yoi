from ultralytics import YOLO
import cv2
import os
from pathlib import Path
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

def load_config():
    """Load configuration from .env file."""
    load_dotenv()
    return {
        "model_type": os.getenv("MODEL_TYPE", "pt").lower(),
        "model_pt_path": os.getenv("MODEL_PT_PATH", "models/pt/best.pt"),
        "model_openvino_dir": os.getenv("MODEL_OPENVINO_DIR", "models/openvino"),
        "video_folder": os.getenv("VIDEO_FOLDER", "vid"),
        "output_folder": os.getenv("OUTPUT_FOLDER", "output"),
        "conf": float(os.getenv("CONF", "0.25")),
        "iou": float(os.getenv("IOU", "0.45")),
        "track_conf": float(os.getenv("TRACK_CONF", "0.5")),
        "track_iou": float(os.getenv("TRACK_IOU", "0.5")),
        "show_live": os.getenv("SHOW_LIVE", "false").lower() == "true",
        "show_live_width": int(os.getenv("SHOW_LIVE_WIDTH", "0")),
        "show_live_height": int(os.getenv("SHOW_LIVE_HEIGHT", "0")),
        "show_live_fullscreen": os.getenv("SHOW_LIVE_FULLSCREEN", "false").lower() == "true",
        "show_live_stop_on_close": os.getenv("SHOW_LIVE_STOP_ON_CLOSE", "true").lower() == "true",
    }


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




def resolve_model_path(config):
    """Resolve model path based on MODEL_TYPE (pt or openvino)."""
    model_type = config["model_type"]
    if model_type == "openvino":
        return config["model_openvino_dir"]
    return config["model_pt_path"]


def run_inference(config):
    """Inference script using YOLOv11s. Configuration is loaded from .env."""
    
    model_path = resolve_model_path(config)
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

    console.print(f"Loading model from [bold]{model_path}[/bold]...")
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
        
        # Setup video writer for output
        output_path = os.path.join(detect_folder, f"detected_{video_path.name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    
    model_path = resolve_model_path(config)
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

    console.print(f"Loading model from [bold]{model_path}[/bold]...")
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

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
                show=False,
                save=True,
                save_dir=track_folder,
                exist_ok=True,
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

                if show_live:
                    annotated_frame = r.plot()
                    cv2.imshow(WINDOW_NAME, annotated_frame)
                    cv2.waitKey(1)
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        show_live = False
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

        console.print(f"Done! Tracking results saved to folder: {track_folder}")

if __name__ == "__main__":
    console.print("=" * 60)
    console.print("Inference - YOLOv11s")
    console.print("=" * 60)
    
    config = load_config()
    
    console.print("\nSelect inference mode:")
    console.print("1. Standard Detection (frame by frame)")
    console.print("2. Detection with Tracking (object tracking)")
    
    choice = input("\nChoose mode (1/2) [default: 1]: ").strip() or "1"
    
    if choice == "2":
        run_inference_with_tracking(config)
    else:
        run_inference(config)
    
    console.print("\nInference completed!")
