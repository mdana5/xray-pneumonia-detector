"""
inference.py — Inference script for best_pneumonia_model.keras
Person 3: Confidence Routing + Grad-CAM integration

Usage:
    python inference.py --image path/to/xray.jpg --model path/to/model.keras
    python inference.py --image path/to/xray.jpg --model path/to/model.keras --save_heatmap
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image

from confidence_router import route_prediction
from gradcam import GradCAM

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE            = (224, 224)
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
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array


def run_inference(model, image_path: str, save_heatmap: bool = False) -> dict:
    # 1. Preprocess + predict
    img_array      = preprocess_image(image_path)
    raw_output     = model.predict(img_array, verbose=0)
    pneumonia_prob = float(raw_output[0][0])

    # 2. Confidence routing
    routing = route_prediction(
        pneumonia_prob,
        high_threshold=HIGH_CONF_THRESHOLD,
        low_threshold=LOW_CONF_THRESHOLD,
    )

    # 3. Grad-CAM
    heatmap_path = None
    if save_heatmap:
        cam        = GradCAM(model)
        heatmap    = cam.generate(img_array)
        overlay    = cam.overlay_on_image(image_path, heatmap)
        heatmap_path = f"outputs/gradcam_{os.path.basename(image_path)}"
        cam.save(overlay, heatmap_path)

    return {
        "image":          os.path.basename(image_path),
        "predicted_class": routing["predicted_class"],
        "confidence":      round(routing["confidence"] * 100, 2),
        "pneumonia_prob":  round(pneumonia_prob * 100, 2),
        "normal_prob":     round((1 - pneumonia_prob) * 100, 2),
        "decision":        routing["decision"],
        "needs_review":    routing["needs_review"],
        "heatmap_saved":   heatmap_path,
    }


def print_result(result: dict):
    print("\n" + "=" * 55)
    print(f"  Image          : {result['image']}")
    print(f"  Prediction     : {result['predicted_class']}")
    print(f"  Confidence     : {result['confidence']}%")
    print(f"  Pneumonia prob : {result['pneumonia_prob']}%")
    print(f"  Normal prob    : {result['normal_prob']}%")
    print(f"  Decision       : {result['decision']}")
    print(f"  Needs review   : {result['needs_review']}")
    if result["heatmap_saved"]:
        print(f"  Heatmap saved  : {result['heatmap_saved']}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pneumonia X-ray inference")
    parser.add_argument("--image",        required=True, help="Path to chest X-ray image")
    parser.add_argument("--model",        required=True, help="Path to .keras model file")
    parser.add_argument("--save_heatmap", action="store_true", help="Generate and save Grad-CAM heatmap")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        return

    model  = load_model(args.model)
    result = run_inference(model, args.image, save_heatmap=args.save_heatmap)
    print_result(result)


if __name__ == "__main__":
    main()
