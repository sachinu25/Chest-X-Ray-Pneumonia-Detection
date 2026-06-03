"""
Training entry point.

Usage:
    python train.py
"""

import sys

from xray.exception import XRayException
from xray.pipeline.train_pipeline import TrainPipeline
from xray.logger import logging


def start_training() -> None:
    """Initialize and run the complete training pipeline."""
    try:
        logging.info("Training initiated via train.py")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise XRayException(e, sys)


if __name__ == "__main__":
    start_training()
