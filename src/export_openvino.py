from ultralytics import YOLO
import os
from dotenv import load_dotenv


def load_config():
    """Load configuration from .env file."""
    load_dotenv()
    return {
        "model_pt_path": os.getenv("MODEL_PT_PATH", "models/best.pt"),
        "model_openvino_dir": os.getenv("MODEL_OPENVINO_DIR", "models/openvino"),
        "imgsz": int(os.getenv("EXPORT_IMGSZ", "640")),
        "half": os.getenv("EXPORT_HALF", "false").lower() == "true",
    }


def export_to_openvino():
    config = load_config()
    model_pt_path = config["model_pt_path"]
    output_dir = config["model_openvino_dir"]

    if not os.path.exists(model_pt_path):
        print(f"Error: .pt model not found: {model_pt_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_pt_path}...")
    model = YOLO(model_pt_path)

    print("Exporting to OpenVINO...")
    export_kwargs = {
        "format": "openvino",
        "imgsz": config["imgsz"],
        "half": config["half"],
        "project": output_dir,
        "name": ".",
        "exist_ok": True,
    }

    model.export(
        **export_kwargs,
    )

    print(f"Done! OpenVINO model saved to: {output_dir}")


if __name__ == "__main__":
    export_to_openvino()
