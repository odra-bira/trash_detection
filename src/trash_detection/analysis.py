"""Transparent, heuristic litter scoring and density visualization."""
from collections import Counter

import cv2
import numpy as np


def cleanliness_score(detections, image_width, image_height, scale=8.0):
    """Return a score in [0, 100]; higher scores imply less detected litter.

    Each detection has a 'box' in xyxy pixel coordinates and 'confidence' in [0,1].
    The heuristic is intentionally not a scientifically validated cleanliness metric.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if scale < 0:
        raise ValueError("Scale must be non-negative.")
    total = 0.0
    area = float(image_width * image_height)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        width = max(0.0, min(float(x2), image_width) - max(float(x1), 0.0))
        height = max(0.0, min(float(y2), image_height) - max(float(y1), 0.0))
        confidence = float(np.clip(det["confidence"], 0, 1))
        total += confidence * (0.1 + (width * height / area))
    return round(max(0.0, 100.0 - scale * total), 2)


def class_counts(detections):
    """Count detections by class label."""
    return dict(Counter(str(det["class_name"]) for det in detections))


def density_heatmap(detections, height, width, blur_fraction=0.07):
    """Create a BGR heatmap over the bounding-box regions."""
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")
    mask = np.zeros((height, width), dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        left, right = max(0, int(x1)), min(width, int(x2))
        top, bottom = max(0, int(y1)), min(height, int(y2))
        if right > left and bottom > top:
            mask[top:bottom, left:right] += float(np.clip(det["confidence"], 0, 1))
    sigma = max(1.0, min(height, width) * blur_fraction)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if np.max(mask) > 0:
        mask = mask / np.max(mask)
    normalized = np.uint8(np.clip(mask * 255, 0, 255))
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
