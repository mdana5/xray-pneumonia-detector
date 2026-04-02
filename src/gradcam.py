# gradcam.py
# Person 3 - Grad-CAM Heatmap Generation
# Author: Anas

import numpy as np
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage

# Last conv layer in MobileNetV2 before global average pooling
GRADCAM_LAYER = "out_relu"


class GradCAM:
    """
    Grad-CAM implementation for the MobileNetV2 Keras model.
    Generates heatmaps highlighting regions that influenced the prediction.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model: tf.keras.Model, layer_name: str = GRADCAM_LAYER):
        self.model = model
        self.layer_name = layer_name

        # Sub-model that outputs (conv layer activations, final prediction)
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output],
        )

    def generate(self, img_array: np.ndarray) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a preprocessed image.

        Args:
            img_array: preprocessed image, shape (1, 224, 224, 3)

        Returns:
            heatmap: np.ndarray (H, W), values in [0, 1]
        """
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = self.grad_model(inputs)
            # Sigmoid binary output — use the single neuron directly
            loss = predictions[:, 0]

        # Gradients of prediction w.r.t. conv layer output
        grads = tape.gradient(loss, conv_outputs)       # (1, H, W, C)

        # Global average pooling → per-channel importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        # Weighted sum of feature maps
        conv_outputs = conv_outputs[0]                         # (H, W, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis] # (H, W, 1)
        heatmap = tf.squeeze(heatmap)                          # (H, W)

        # ReLU — keep only positive activations
        heatmap = tf.nn.relu(heatmap).numpy()

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def overlay_on_image(self, original_image_path: str, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on the original X-ray image.

        Args:
            original_image_path: path to the original image file
            heatmap: output of generate(), shape (H, W), values in [0, 1]
            alpha: heatmap opacity (0=invisible, 1=fully opaque)

        Returns:
            overlaid image as np.ndarray (224, 224, 3), BGR
        """
        img = np.array(PILImage.open(original_image_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (224, 224))

        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        overlaid = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        return overlaid

    def save(self, overlaid_image: np.ndarray, output_path: str):
        """Save the overlaid heatmap image to disk."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlaid_image)