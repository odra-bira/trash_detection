"""YOLOv8 inference and visualization for litter images."""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .analysis import class_counts, cleanliness_score, density_heatmap


def load_detector(weights):
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {path}")
    return YOLO(str(path))


def analyze_image(model, image_bgr, confidence=0.25):
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels.")
    height, width = image_bgr.shape[:2]
    result = model.predict(source=image_bgr, conf=confidence, verbose=False)[0]
    names = result.names
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            coords = [float(v) for v in box.xyxy[0].cpu().tolist()]
            class_id = int(box.cls[0].item())
            label = names[class_id] if isinstance(names, (list, dict)) else str(class_id)
            detections.append({
                "box": coords,
                "confidence": round(float(box.conf[0].item()), 5),
                "class_id": class_id,
                "class_name": str(label),
            })
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 220, 80), 2)
        label = f'{det["class_name"]}: {det["confidence"]:.2f}'
        cv2.putText(annotated, label, (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 2)
    heatmap = density_heatmap(detections, height, width)
    # An empty detection set should not tint the original scene.
    overlay = (cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
               if detections else image_bgr.copy())
    summary = {
        "detection_count": len(detections),
        "class_counts": class_counts(detections),
        "cleanliness_score": cleanliness_score(detections, width, height),
        "detections": detections,
    }
    return annotated, heatmap, overlay, summary
