"""
Multimodal Crime / Incident Report Analyzer
Student 3 — Image Analyst Module

Pipeline:
  1. Object Detection   -> pretrained YOLOv8 (COCO classes: person, car, truck, etc.)
  2. Scene Classification -> fire / smoke / light / no-fire (color-signature heuristic,
                              swappable for a trained Roboflow fire-detection model)
  3. OCR Text Extraction  -> pytesseract (license plates, street signs, etc.)
  4. Structured Output    -> one row per image: Image_ID, Scene_Type, Objects_Detected,
                              Text_Extracted, Confidence_Score

Usage:
    python src/image_analyzer.py --input sample_images --output output/image_report.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RELEVANT_OBJECT_CLASSES = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "fire hydrant", "stop sign", "traffic light", "backpack",
    "handbag", "suitcase", "knife", "baseball bat", "bottle",
}

YOLO_CONF_THRESHOLD = 0.35
SMOKE_GRAY_SATURATION_MAX = 40
SMOKE_GRAY_VALUE_MIN = 90
FIRE_HUE_RANGE = (5, 35)
FIRE_SATURATION_MIN = 120
FIRE_VALUE_MIN = 140

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_MODEL_ID = "fire-detection-data-pre/4"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_FIRE_CONF_THRESHOLD = 0.30


# ----------------------------------------------------------------------------
# 1. Object Detection (pretrained YOLOv8, no training)
# ----------------------------------------------------------------------------

_model_cache = {}


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    if weights not in _model_cache:
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def detect_objects(image_path: str, model: YOLO) -> list[tuple[str, float]]:
    results = model(image_path, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < YOLO_CONF_THRESHOLD:
                continue
            label = model.names[int(box.cls[0])]
            if label in RELEVANT_OBJECT_CLASSES:
                detections.append((label, conf))
    return detections


def load_fire_client():
    if not ROBOFLOW_API_KEY:
        return None
    return InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)


def detect_fire(image_path: str, client) -> list[tuple[str, float]]:
    if client is None:
        return []
    try:
        result = client.infer(image_path, model_id=ROBOFLOW_MODEL_ID)
    except Exception as e:
        print(f"  [warn] Roboflow fire-detection API call failed: {e}")
        return []

    if isinstance(result, dict):
        predictions = result.get("predictions", [])
    else:
        predictions = getattr(result, "predictions", []) or []

    detections = []
    for pred in predictions:
        if isinstance(pred, dict):
            cls = pred.get("class", "fire")
            conf = pred.get("confidence", 0.0)
        else:
            cls = getattr(pred, "class_name", "fire")
            conf = getattr(pred, "confidence", 0.0)
        if conf >= ROBOFLOW_FIRE_CONF_THRESHOLD:
            detections.append((cls, float(conf)))
    return detections


# ----------------------------------------------------------------------------
# 2. Scene Classification (Fire / Smoke / Light / No-fire)
# ----------------------------------------------------------------------------

def _classify_scene_heuristic(image_path: str) -> tuple[str, float]:
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown", 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total_pixels = h.size

    fire_mask = (
        (h >= FIRE_HUE_RANGE[0]) & (h <= FIRE_HUE_RANGE[1]) &
        (s >= FIRE_SATURATION_MIN) & (v >= FIRE_VALUE_MIN)
    )
    fire_ratio = float(np.sum(fire_mask)) / total_pixels

    smoke_mask = (
        (s <= SMOKE_GRAY_SATURATION_MAX) & (v >= SMOKE_GRAY_VALUE_MIN)
    )
    smoke_ratio = float(np.sum(smoke_mask)) / total_pixels

    brightness_mean = float(np.mean(v)) / 255.0

    if fire_ratio > 0.03:
        confidence = min(0.5 + fire_ratio * 5, 0.98)
        return "Fire Scene", round(confidence, 2)
    elif smoke_ratio > 0.15:
        confidence = min(0.5 + smoke_ratio * 1.5, 0.95)
        return "Smoke Scene", round(confidence, 2)
    elif brightness_mean > 0.6:
        return "Light Scene", round(0.5 + brightness_mean * 0.3, 2)
    else:
        return "No-Fire Scene", round(0.6 + (1 - brightness_mean) * 0.2, 2)


def classify_scene(image_path: str, fire_detections: list[tuple[str, float]]) -> tuple[str, float]:
    if fire_detections:
        best_conf = max(conf for _, conf in fire_detections)
        return "Fire Scene", round(best_conf, 2)

    if ROBOFLOW_API_KEY:
        return "No-Fire Scene", 0.5

    return _classify_scene_heuristic(image_path)


# ----------------------------------------------------------------------------
# 3. OCR Text Extraction
# ----------------------------------------------------------------------------

def extract_text(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    raw_text = pytesseract.image_to_string(thresh)
    cleaned = " ".join(raw_text.split())
    return cleaned


# ----------------------------------------------------------------------------
# 4. Pipeline: process one image -> structured row
# ----------------------------------------------------------------------------

def process_image(image_path: str, model: YOLO, fire_client) -> dict:
    image_id = Path(image_path).stem

    detections = detect_objects(image_path, model)
    fire_detections = detect_fire(image_path, fire_client)
    scene_type, scene_conf = classify_scene(image_path, fire_detections)
    text_extracted = extract_text(image_path)

    objects_detected = [label for label, _ in detections] + [cls for cls, _ in fire_detections]
    all_confidences = [c for _, c in detections] + [c for _, c in fire_detections] + [scene_conf]
    overall_confidence = round(sum(all_confidences) / len(all_confidences), 2)

    return {
        "Image_ID": image_id,
        "Scene_Type": scene_type,
        "Objects_Detected": ", ".join(sorted(set(objects_detected))) if objects_detected else "none",
        "Text_Extracted": text_extracted if text_extracted else "",
        "Confidence_Score": overall_confidence,
    }


def run_pipeline(input_dir: str, output_csv: str, weights: str = "yolov8n.pt") -> list[dict]:
    model = load_model(weights)
    fire_client = load_fire_client()
    if fire_client is None:
        print("  [info] ROBOFLOW_API_KEY not set -- using HSV color heuristic "
              "fallback for scene classification instead of the real fire model.")
    image_paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return []

    rows = []
    for path in image_paths:
        print(f"Processing {path.name} ...")
        row = process_image(str(path), model, fire_client)
        rows.append(row)

    os.makedirs(Path(output_csv).parent, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Image_ID", "Scene_Type", "Objects_Detected",
            "Text_Extracted", "Confidence_Score",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_csv}")
    return rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Image modality analyzer for incident reports")
    parser.add_argument("--input", default="sample_images", help="Folder of input images")
    parser.add_argument("--output", default="output/image_report.csv", help="Output CSV path")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 pretrained weights")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.weights)


if __name__ == "__main__":
    main()