# test_gradcam.py
# Testing Grad-CAM with a dummy model and image
# Author: Anas

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from gradcam import GradCAM


# Load MobileNetV2 with dummy weights
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes: Normal, Pneumonia

# Target the last conv layer in MobileNetV2
target_layer = model.features[-1][0]

# Initialize Grad-CAM
gradcam = GradCAM(model, target_layer)

# Create a dummy image tensor (1, 3, 224, 224)
dummy_input = torch.randn(1, 3, 224, 224)

# Generate heatmap
cam = gradcam.generate(dummy_input, class_idx=1)  # target Pneumonia class

print("Heatmap shape:", cam.shape)
print("Heatmap min:", round(float(cam.min()), 4))
print("Heatmap max:", round(float(cam.max()), 4))
print("Grad-CAM test passed!" if cam.shape == (7, 7) else "Unexpected shape!")