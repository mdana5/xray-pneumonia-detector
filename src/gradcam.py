# gradcam.py
# Person 3 - Grad-CAM Heatmap Generation
# Author: Anas

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM implementation for MobileNetV2.
    Generates heatmaps highlighting regions that influenced the prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor (1, 3, H, W)
            class_idx: target class (0=Normal, 1=Pneumonia). If None, uses predicted class.

        Returns:
            heatmap as numpy array (H, W), values in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pooling on gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def overlay_on_image(self, original_image, cam, alpha=0.4):
        """
        Overlay heatmap on original image.

        Args:
            original_image: numpy array (H, W, 3), uint8
            cam: heatmap from generate(), values in [0, 1]
            alpha: blending factor

        Returns:
            overlaid image as numpy array
        """
        heatmap = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        return overlaid