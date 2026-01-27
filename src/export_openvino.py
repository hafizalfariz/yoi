from ultralytics import YOLO
import os
from pathlib import Path
import shutil
from dotenv import load_dotenv


def load_config():
    """Load configuration from .env file."""
    load_dotenv()
    return {
        "model_pt_path": os.getenv("MODEL_PT_PATH", "models/pt/best.pt"),
        "model_openvino_dir": os.getenv("MODEL_OPENVINO_DIR", "models/openvino"),
        "imgsz": int(os.getenv("EXPORT_IMGSZ", "640")),
        "half": os.getenv("EXPORT_HALF", "false").lower() == "true",
    }


def _move_export_to_target(export_path: Path, target_dir: Path) -> None:
    """Move OpenVINO export output into the target directory."""
    export_dir = export_path.parent if export_path.is_file() else export_path

    try:
        if export_dir.resolve() == target_dir.resolve():
            return
    except FileNotFoundError:
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    for item in export_dir.iterdir():
        target = target_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

    try:
        export_dir.rmdir()
    except OSError:
        pass


def export_to_openvino():
    config = load_config()
    model_pt_path = Path(config["model_pt_path"])
    output_dir = Path(config["model_openvino_dir"])

    if not model_pt_path.exists():
        print(f"Error: .pt model not found: {model_pt_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_pt_path}...")
    model = YOLO(model_pt_path)

    print("Exporting to OpenVINO...")
    export_kwargs = {
        "format": "openvino",
        "imgsz": config["imgsz"],
        "half": config["half"],
        "project": str(output_dir),
        "name": ".",
        "exist_ok": True,
    }

    export_result = model.export(
        **export_kwargs,
    )

    if export_result:
        export_path = Path(export_result)
        print(f"OpenVINO exported to: {export_path}")
        _move_export_to_target(export_path, output_dir)

    print(f"Done! OpenVINO model saved to: {output_dir}")


if __name__ == "__main__":
    export_to_openvino()
