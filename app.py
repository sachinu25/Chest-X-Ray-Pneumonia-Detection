"""
FastAPI application for X-ray Pneumonia Detection.

Provides a production-grade REST API for real-time pneumonia classification
from chest X-ray images. Designed for AWS deployment (EC2, ECS Fargate).

Production hardening:
    - Lifespan-managed model loading (CUDA fork-safe)
    - Async inference via thread pool executor (non-blocking event loop)
    - Streaming upload with size + magic-byte validation (DoS-resistant)
    - Security headers middleware
    - Configurable CORS origins
    - Liveness (/health) and readiness (/ready) probes
"""

import asyncio
import functools
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Dict

import torch
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
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
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks

# JPEG: FF D8 FF, PNG: 89 50 4E 47 0D 0A 1A 0A
IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
}


# ---------------------------------------------------------------------------
# Fix #1 + #6: Lifespan-managed model loading (CUDA fork-safe)
# ---------------------------------------------------------------------------
# Model is loaded INSIDE the lifespan handler, which runs AFTER Uvicorn forks
# worker processes. This prevents CUDA context corruption that occurs when a
# parent process initialises CUDA and then forks children.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle — startup and shutdown."""
    # --- Startup ---
    logger.info("Starting X-ray Pneumonia Detection API")
    app.state.ready = False
    app.state.model = None

    try:
        app.state.model = load_model(DEFAULT_MODEL_PATH)
        app.state.ready = True
        logger.info(f"Model loaded successfully on {DEVICE}")
    except FileNotFoundError:
        logger.warning(
            f"Model file not found at {DEFAULT_MODEL_PATH}. "
            "Using random weights for development."
        )
        app.state.model = XRayClassifier(num_classes=NUM_CLASSES).to(DEVICE).eval()
        app.state.ready = True
    except Exception:
        logger.exception("Failed to load model during startup")
        # app.state.ready remains False — /ready will return 503

    yield

    # --- Shutdown ---
    logger.info("Shutting down API — releasing resources")
    app.state.model = None
    app.state.ready = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")


# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="X-ray Pneumonia Detection API",
    description="Classify chest X-ray images as Normal or Pneumonia",
    version="3.0.0",
    lifespan=lifespan,
)

# Fix #8: CORS Hardening — read allowed origins from environment
_cors_origins_raw = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
_cors_origins = (
    ["*"] if _cors_origins_raw.strip() == "*"
    else [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # allow_credentials must be False when origins is ["*"] (CORS spec requirement)
    allow_credentials=("*" not in _cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Fix #7: Security Headers — injected via @app.middleware for reliability
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Inject standard security headers on every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    # HSTS only makes sense behind a TLS terminator (ALB, CloudFront)
    if request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response

# ---------------------------------------------------------------------------
# Fix #3: Memory-safe streaming upload reader
# ---------------------------------------------------------------------------
async def _read_upload_safe(file: UploadFile) -> bytes:
    """Read an upload file in chunks, enforcing size and magic-byte validation.

    Raises HTTPException on:
        - Empty file (400)
        - Invalid image magic bytes (400)
        - File exceeding MAX_FILE_SIZE (413)
    """
    chunks: list[bytes] = []
    total_size = 0
    first_chunk = True

    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break

        total_size += len(chunk)

        # Reject oversized files BEFORE reading the full payload into memory
        if total_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)}MB.",
            )

        # Validate magic bytes on the very first chunk
        if first_chunk:
            first_chunk = False
            if not any(chunk.startswith(magic) for magic in IMAGE_MAGIC_BYTES):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid image file. Only JPEG and PNG images are accepted. "
                        "The file header does not match any supported image format."
                    ),
                )

        chunks.append(chunk)

    if total_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Fix #2: Non-blocking inference helper
# ---------------------------------------------------------------------------
def _run_inference_sync(model: torch.nn.Module, image: Image.Image) -> Dict:
    """Run the full preprocessing + inference pipeline synchronously.

    This function is designed to be called via ``run_in_executor`` so that
    the CPU/GPU-heavy PyTorch forward pass does not block the async event loop.
    """
    input_tensor = preprocess_image(image)
    label, confidence, prob_dict = predict(model, input_tensor)
    pred_idx = 0 if label == "NORMAL" else 1
    return {
        "prediction_index": pred_idx,
        "prediction_label": label,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


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


# Fix #9: Liveness probe — always responds if process is alive
@app.get("/health", tags=["Health"])
def health_check() -> Dict:
    """Liveness probe — indicates the process is alive."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
    }


# Fix #9: Readiness probe — model must be loaded and ready
@app.get("/ready", tags=["Health"])
def readiness_check(request: Request) -> Dict:
    """Readiness probe — indicates the API can serve predictions.

    Returns HTTP 503 if the model has not finished loading yet.
    Used by load balancers and container orchestrators (ECS, K8s)
    to know when to start routing traffic to this instance.
    """
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. The service is starting up.",
        )
    return {
        "status": "ready",
        "model_loaded": request.app.state.model is not None,
        "device": str(DEVICE),
    }


@app.post("/predict", tags=["Prediction"])
async def predict_endpoint(request: Request, file: UploadFile = File(...)) -> Dict:
    """Predict whether a chest X-ray shows Pneumonia or Normal.

    Args:
        file: Uploaded chest X-ray image (JPEG or PNG, max 10MB).

    Returns:
        Prediction label, index, and confidence score.
    """
    # Ensure model is ready
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please try again later.",
        )

    # Fix #3: Stream-read with size + magic-byte validation
    image_bytes = await _read_upload_safe(file)

    # Parse image
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open file as an image. Please upload a valid image.",
        )

    # Fix #2: Run inference in thread pool to avoid blocking the event loop
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            functools.partial(_run_inference_sync, model, image),
        )
    except Exception as e:
        logger.error("Inference failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during inference. Please try again.",
        )

    return result
