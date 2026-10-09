"""Run litter-detection inference on an image."""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trash_detection.pipeline import analyze_image, load_detector


def main():
    parser = argparse.ArgumentParser(description="Litter-detection image inference")
    parser.add_argument("--weights", required=True, help="Path to trained YOLOv8 weights")
    parser.add_argument("--image", required=True, help="Input image file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Could not read image: {args.image}")
    model = load_detector(args.weights)
    annotated, heatmap, overlay, summary = analyze_image(model, image, args.confidence)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in [("annotated.jpg", annotated), ("heatmap.jpg", heatmap),
                        ("overlay.jpg", overlay)]:
        if not cv2.imwrite(str(destination / name), frame):
            raise OSError(f"Failed to write {destination / name}")
    (destination / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "detections"}, indent=2))


if __name__ == "__main__":
    main()
