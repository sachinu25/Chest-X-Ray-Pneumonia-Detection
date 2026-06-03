"""
Grad-CAM explainability module for the Streamlit dashboard.

Generates class activation heatmaps using hook-based Grad-CAM on ResNet18.
Falls back gracefully if the model architecture doesn't support it.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.model_utils import DEVICE


class GradCAMGenerator:
    """
    Hook-based Grad-CAM implementation for ResNet18-based classifiers.

    Attaches forward and backward hooks to the target layer (default: layer4),
    computes the class activation map, and overlays it on the original image.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "layer4"):
        """
        Args:
            model: The model to explain (must have a backbone with named submodules).
            target_layer_name: Name of the convolutional layer to hook into.
        """
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []

        # Locate the target layer inside the backbone
        target_layer = None
        if hasattr(model, "backbone"):
            for name, module in model.backbone.named_modules():
                if name == target_layer_name:
                    target_layer = module
                    break

        if target_layer is None:
            raise ValueError(
                f"Could not find layer '{target_layer_name}' in model.backbone. "
                f"Available: {[n for n, _ in model.backbone.named_modules()]}"
            )

        # Register hooks
        self._hooks.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W).
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            Heatmap as a numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()

        # Need gradients for the backward pass
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients and backward
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # ReLU to keep only positive contributions

        # Resize to input size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def cleanup(self):
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def create_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image: Original PIL Image.
        heatmap: Normalized heatmap array (H, W) with values in [0, 1].
        alpha: Blending factor for the overlay.
        colormap: Matplotlib colormap name.

    Returns:
        PIL Image with the heatmap overlay.
    """
    import matplotlib.cm as cm

    # Get colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)  # (H, W, 4) RGBA
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)  # RGB

    heatmap_img = Image.fromarray(heatmap_colored).resize(
        original_image.size, Image.BILINEAR
    )

    # Blend
    original_rgb = original_image.convert("RGB")
    overlay = Image.blend(original_rgb, heatmap_img, alpha=alpha)
    return overlay
