# Trash Detection & Cleanliness Assessment

**Odra Bira** | Computer Vision · Python · YOLOv8 · OpenCV · Gradio

An image-analysis pipeline that detects litter, visualizes where it accumulates, and provides an interpretable cleanliness estimate.

## Highlights

- **Object detection**: label and locate trash with a YOLOv8 model.
- **Visual analytics**: annotated predictions, density heatmap, and overlay.
- **Cleanliness score**: transparent heuristic using detection confidence and bounding-box area.
- **Interfaces**: command-line prediction plus an interactive Gradio dashboard.
- **Reproducible workflow**: scripts for training, validation, unit tests, and clear configuration.

## Model evaluation

The original course-project experiment used a cleaned YOLO-format TACO dataset export. A YOLOv8n baseline (30 epochs, 640px) was reported with the following validation metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.78 |
| Recall | 0.17 |
| mAP@50 | 0.14 |
| mAP@50–95 | 0.11 |

These values are from the original project evaluation and **have not been reproduced with this published code**. In particular, the original notebook, dataset and trained checkpoint are not bundled here.

## Quick start

Python 3.10+ recommended.

```bash
git clone https://github.com/odra-bira/trash_detection.git
cd trash_detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Bring your own **trained litter-detection** YOLO checkpoint (for example, `weights/best.pt`). Generic COCO weights are not a replacement for a model trained on the litter classes.

**CLI inference:**
```bash
python scripts/predict.py --weights weights/best.pt --image /path/to/photo.jpg --output outputs
```

**Gradio app:**
```bash
python app.py --weights weights/best.pt
```

**Optional training / evaluation:**
```bash
python scripts/train.py --data /path/to/data.yaml --model yolov8n.pt --epochs 30 --imgsz 640
python scripts/train.py --data /path/to/data.yaml --weights /path/to/best.pt --evaluate
```

Dataset YAML must define compatible train/validation images and class names.

## Repository layout

```text
app.py                         Interactive demo
scripts/predict.py             CLI image inference
scripts/train.py               Model training and validation
src/trash_detection/analysis.py  Score and heatmap calculations
src/trash_detection/pipeline.py  Inference and rendering
tests/test_analysis.py         Unit tests
requirements.txt               Dependencies
```

Predictions save `annotated.jpg`, `heatmap.jpg`, `overlay.jpg` and `results.json`.

## How cleanliness scoring works

Each predicted object contributes a penalty proportional to confidence and bounding-box coverage. All penalties are summed and converted to a number between 0 and 100 (higher means cleaner). A Gaussian-smoothed occupancy mask highlights areas with higher predicted litter density.

**Interpret with caution:** the score is a demonstration heuristic, not a calibrated environmental measurement. Missed trash lowers the computed penalty. The original baseline's recall was limited, especially for small or ambiguous objects.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Next improvements

Improve small-object recall, compare alternative YOLO model sizes, calibrate scores with human judgments and test across new locations and conditions.

## Resources

- [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/)
- [TACO: Trash Annotations in Context](http://tacodataset.org/)

**Implementation: Odra Bira.** This public repository contains a clean, newly organized implementation of the project's detection, scoring and visualization workflow; original training weights and notebook were not available for inclusion.
