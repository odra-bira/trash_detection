"""Train or evaluate a YOLOv8 detector on a YOLO-format litter dataset."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train or validate YOLOv8")
    parser.add_argument("--data", required=True, help="YOLO data.yaml path")
    parser.add_argument("--model", default="yolov8n.pt", help="Initial weights for training")
    parser.add_argument("--weights", help="Trained weights for evaluation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if not Path(args.data).exists():
        parser.error(f"Dataset YAML not found: {args.data}")
    if args.evaluate:
        if not args.weights:
            parser.error("--weights is required with --evaluate")
        metrics = YOLO(args.weights).val(data=args.data, imgsz=args.imgsz)
        print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    else:
        YOLO(args.model).train(data=args.data, epochs=args.epochs,
                               imgsz=args.imgsz, project="runs/trash", name="train")


if __name__ == "__main__":
    main()
