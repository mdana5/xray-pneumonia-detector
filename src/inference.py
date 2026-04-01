"""
inference.py — Manual inference script for best_pneumonia_model.keras
Person 3: Confidence Routing integration point

Usage:
    python inference.py --image path/to/xray.jpg
    python inference.py --image path/to/xray.jpg --threshold 0.5
"""

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

HIGH_CONF_THRESHOLD = 0.85
LOW_CONF_THRESHOLD  = 0.15


def load_model(model_path: str) -> tf.keras.Model:
    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def route_decision(pneumonia_prob: float) -> str:
    if pneumonia_prob >= HIGH_CONF_THRESHOLD:
        return "AUTO-ROUTE → Pneumonia ward (high confidence)"
    elif pneumonia_prob <= LOW_CONF_THRESHOLD:
        return "AUTO-ROUTE → Normal / discharge (high confidence)"
    else:
        return "FLAG FOR EXPERT REVIEW (low confidence — ambiguous case)"


def run_inference(model, image_path: str, threshold: float = 0.5) -> dict:
    img_array      = preprocess_image(image_path)
    raw_output     = model.predict(img_array, verbose=0)
    pneumonia_prob = float(raw_output[0][0])
    normal_prob    = 1.0 - pneumonia_prob

    predicted_class = 1 if pneumonia_prob >= threshold else 0
    label           = CLASS_NAMES[predicted_class]
    confidence      = pneumonia_prob if predicted_class == 1 else normal_prob
    routing         = route_decision(pneumonia_prob)

    return {
        "image":          os.path.basename(image_path),
        "label":          label,
        "confidence":     round(confidence * 100, 2),
        "pneumonia_prob": round(pneumonia_prob * 100, 2),
        "normal_prob":    round(normal_prob * 100, 2),
        "routing":        routing,
        "threshold_used": threshold,
    }


def print_result(result: dict):
    print("\n" + "=" * 55)
    print(f"  Image         : {result['image']}")
    print(f"  Prediction    : {result['label']}")
    print(f"  Confidence    : {result['confidence']}%")
    print(f"  Pneumonia prob: {result['pneumonia_prob']}%")
    print(f"  Normal prob   : {result['normal_prob']}%")
    print(f"  Routing       : {result['routing']}")
    print(f"  Threshold     : {result['threshold_used']}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pneumonia X-ray inference")
    parser.add_argument("--image",     required=True,      help="Path to chest X-ray image")
    parser.add_argument("--model",     required=True,      help="Path to .keras model file")
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        return

    model  = load_model(args.model)
    result = run_inference(model, args.image, threshold=args.threshold)
    print_result(result)


if __name__ == "__main__":
    main()