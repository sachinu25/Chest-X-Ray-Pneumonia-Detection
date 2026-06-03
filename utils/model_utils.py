"""
Model loading, preprocessing, and inference utilities for the Streamlit dashboard.

Provides cached model loading, image preprocessing matching the training pipeline,
and inference with softmax probability extraction.
"""

import os
from pathlib import Path
from io import BytesIO
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from xray.ml.model.arch import XRayClassifier
from xray.constant.training_pipeline import (
    NORMALIZE_LIST_1,
    NORMALIZE_LIST_2,
    NUM_CLASSES,
    PREDICTION_LABEL,
    RESIZE,
    CENTERCROP,
)

# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default model path (repo root)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "xray_model.pth",
)

# ---------------------------------------------------------------------------
# Inference Transforms (MUST match training pipeline exactly)
# ---------------------------------------------------------------------------
INFERENCE_TRANSFORM = T.Compose(
    [
        T.Resize(RESIZE),
        T.CenterCrop(CENTERCROP),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_LIST_1, std=NORMALIZE_LIST_2),
    ]
)


def load_model(model_path: str | None = None) -> XRayClassifier:
    """
    Load the trained model from a .pth state_dict checkpoint.

    Only supports state_dict checkpoints (created via ``torch.save(model.state_dict(), ...)``).
    Full-model pickle loading is disabled for security — ``weights_only=True`` prevents
    arbitrary code execution from untrusted checkpoint files.

    Args:
        model_path: Absolute path to the .pth file. Defaults to repo-root xray_model.pth.

    Returns:
        Model in eval mode on the appropriate device.

    Raises:
        FileNotFoundError: If model file does not exist.
        RuntimeError: If checkpoint format is unrecognized.
    """
    path = model_path or DEFAULT_MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    state = torch.load(path, map_location=DEVICE, weights_only=True)

    if not isinstance(state, dict):
        raise RuntimeError(
            f"Expected a state_dict (dict), got {type(state)}. "
            "Full-model pickle loading is disabled for security."
        )

    model = XRayClassifier(num_classes=NUM_CLASSES, freeze_backbone=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Apply inference transforms to a PIL Image.

    Args:
        image: PIL Image in any mode (will be converted to RGB).

    Returns:
        Tensor of shape (1, 3, 224, 224) on the target device.
    """
    img = image.convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    return tensor


def predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Run inference and return the predicted label, confidence, and full probabilities.

    Args:
        model: Model in eval mode.
        tensor: Preprocessed image tensor (1, 3, 224, 224).

    Returns:
        (label, confidence, probabilities_dict) where probabilities_dict maps
        class names to their softmax probabilities.
    """
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    confidence, idx = torch.max(probs, dim=0)
    label = PREDICTION_LABEL[int(idx)]
    prob_dict = {
        PREDICTION_LABEL[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)
    }
    return label, round(confidence.item(), 4), prob_dict


def get_risk_level(label: str, confidence: float) -> Tuple[str, str]:
    """
    Determine the clinical risk level based on prediction.

    Returns:
        (risk_level, color) tuple for UI display.
    """
    if label == "PNEUMONIA":
        if confidence >= 0.90:
            return "HIGH RISK", "#FF1744"
        elif confidence >= 0.70:
            return "MODERATE RISK", "#FF9100"
        else:
            return "LOW RISK", "#FFC107"
    else:
        if confidence >= 0.85:
            return "LOW RISK", "#00E676"
        else:
            return "UNCERTAIN", "#FFC107"
