"""Interactive Gradio dashboard for litter detection."""
import argparse
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from trash_detection.pipeline import analyze_image, load_detector


def build_demo(model):
    def predict(image, confidence):
        if image is None:
            raise gr.Error("Upload an image first.")
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        annotated, heatmap, overlay, summary = analyze_image(model, bgr, confidence)
        to_rgb = lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return (to_rgb(annotated), to_rgb(heatmap), to_rgb(overlay), summary)

    with gr.Blocks(title="Trash Detection | Odra Bira") as demo:
        gr.Markdown("# Trash Detection & Cleanliness Assessment\nYOLOv8 computer vision demo · Odra Bira")
        with gr.Row():
            image = gr.Image(type="numpy", label="Upload a scene")
            confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                   label="Detection confidence")
        run = gr.Button("Analyze image", variant="primary")
        with gr.Row():
            annotated = gr.Image(label="Detected litter")
            heatmap = gr.Image(label="Detection heatmap")
            overlay = gr.Image(label="Density overlay")
        results = gr.JSON(label="Detections & cleanliness summary")
        gr.Markdown("**Note:** Cleanliness is a heuristic score, not an environmental safety assessment.")
        run.click(predict, inputs=[image, confidence],
                  outputs=[annotated, heatmap, overlay, results])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to litter-trained YOLOv8 checkpoint")
    args = parser.parse_args()
    build_demo(load_detector(args.weights)).launch()
