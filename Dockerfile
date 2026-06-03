# ==============================================================================
# Multi-stage Dockerfile for X-ray Pneumonia Detection API
# ==============================================================================
# Stage 1: Build dependencies
# Stage 2: Slim production image
# ==============================================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages (CPU-only PyTorch for smaller image)
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt


# --------------- Stage 2: Production ---------------
FROM python:3.10-slim AS production

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY xray/ ./xray/
COPY app.py .
COPY train.py .

# Copy model weights if available
COPY xray_model.pth* ./

# Create directories for logs and artifacts
RUN mkdir -p logs artifacts && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
