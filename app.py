"""
FastAPI application for X-ray Pneumonia Detection.

Provides a REST API for real-time pneumonia classification from chest X-ray images.
Uses the trained XRayClassifier model with proper inference transforms matching
the training pipeline.
"""

import os
import logging
from typing import Dict
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from utils.model_utils import (
    load_model,
    preprocess_image,
    predict,
    DEVICE,
    DEFAULT_MODEL_PATH,
)
from xray.ml.model.arch import XRayClassifier
from xray.constant.training_pipeline import (
    NUM_CLASSES,
    PREDICTION_LABEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------

app = FastAPI(
    title="X-ray Pneumonia Detection API",
    description="Classify chest X-ray images as Normal or Pneumonia",
    version="2.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum upload size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# Initialize model using shared utility with standard logging and fallback for dev
try:
    model = load_model(DEFAULT_MODEL_PATH)
    logger.info(f"Model loaded successfully on {DEVICE}")
except FileNotFoundError:
    logger.warning(f"Model file not found at {DEFAULT_MODEL_PATH}. Using random weights for development.")
    model = XRayClassifier(num_classes=NUM_CLASSES).to(DEVICE).eval()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "X-ray Diagnosis API is running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check() -> Dict:
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.post("/predict", tags=["Prediction"])
async def predict_endpoint(file: UploadFile = File(...)) -> Dict:
    """
    Predict whether a chest X-ray shows Pneumonia or Normal.

    Args:
        file: Uploaded chest X-ray image (JPEG or PNG, max 10MB).

    Returns:
        Prediction label, index, and confidence score.
    """
    # Validate content type
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
            f"Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    # Read and validate file size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes)} bytes). Max: {MAX_FILE_SIZE} bytes.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Process image
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open file as an image. Please upload a valid image.",
        )

    # Use shared inference utilities
    try:
        input_tensor = preprocess_image(image)
        label, confidence, prob_dict = predict(model, input_tensor)
        pred_idx = 0 if label == "NORMAL" else 1
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error performing inference: {str(e)}",
        )

    return {
        "prediction_index": pred_idx,
        "prediction_label": label,
        "confidence": confidence,
        "probabilities": prob_dict,
    }
