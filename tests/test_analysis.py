"""Unit tests for pure scoring helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trash_detection.analysis import class_counts, cleanliness_score, density_heatmap


def test_empty_image_is_clean():
    assert cleanliness_score([], 640, 480) == 100.0


def test_more_detections_reduce_cleanliness():
    item = {"box": [10, 10, 40, 40], "confidence": 0.9, "class_name": "plastic"}
    assert cleanliness_score([item, item], 100, 100) < cleanliness_score([item], 100, 100)


def test_score_is_bounded():
    item = {"box": [0, 0, 100, 100], "confidence": 1, "class_name": "paper"}
    assert cleanliness_score([item] * 500, 100, 100) == 0


def test_counts_and_heatmap():
    detections = [{"box": [4, 5, 20, 23], "confidence": 0.8, "class_name": "can"}]
    assert class_counts(detections) == {"can": 1}
    assert density_heatmap(detections, 40, 50).shape == (40, 50, 3)
