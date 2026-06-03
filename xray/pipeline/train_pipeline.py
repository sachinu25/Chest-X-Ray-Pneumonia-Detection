"""
Training pipeline orchestrator.

Coordinates the end-to-end model training workflow:
    Data Ingestion → Data Transformation → Model Training → Model Evaluation

Includes pipeline timing, artifact summary, and comprehensive logging.
"""

import sys
import time

from xray.components.data_ingestion import DataIngestion
from xray.components.data_transformation import DataTransformation
from xray.components.model_evaluation import ModelEvaluation
from xray.components.model_training import ModelTrainer
from xray.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from xray.exception import XRayException
from xray.logger import logging


class TrainPipeline:
    """
    End-to-end training pipeline that orchestrates all components.

    Stages:
        1. Data Ingestion — check/extract data locally
        2. Data Transformation — augmentation, normalization, DataLoaders
        3. Model Training — train with early stopping + checkpointing
        4. Model Evaluation — comprehensive metrics on test set
    """

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Stage 1: Verify and prepare training data."""
        logging.info("[Pipeline] Stage 1/4: Data Ingestion")
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config,
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("[Pipeline] Data Ingestion completed")
            return data_ingestion_artifact

        except Exception as e:
            raise XRayException(e, sys)

    def start_data_transformation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        """Stage 2: Apply transforms and create DataLoaders."""
        logging.info("[Pipeline] Stage 2/4: Data Transformation")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logging.info("[Pipeline] Data Transformation completed")
            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """Stage 3: Train the model."""
        logging.info("[Pipeline] Stage 3/4: Model Training")
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("[Pipeline] Model Training completed")
            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)

    def start_model_evaluation(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelEvaluationArtifact:
        """Stage 4: Evaluate the trained model."""
        logging.info("[Pipeline] Stage 4/4: Model Evaluation")
        try:
            model_evaluation = ModelEvaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("[Pipeline] Model Evaluation completed")
            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)

    def run_pipeline(self) -> None:
        """Execute the full training pipeline with timing."""
        logging.info("=" * 70)
        logging.info("STARTING TRAINING PIPELINE")
        logging.info("=" * 70)

        pipeline_start = time.time()

        try:
            # Stage 1: Data Ingestion
            stage_start = time.time()
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"  - Ingest Time: {time.time() - stage_start:.1f}s")

            # Stage 2: Data Transformation
            stage_start = time.time()
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            logging.info(f"  - Transform Time: {time.time() - stage_start:.1f}s")

            # Stage 3: Model Training
            stage_start = time.time()
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            logging.info(f"  - Train Time: {time.time() - stage_start:.1f}s")

            # Stage 4: Model Evaluation
            stage_start = time.time()
            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact,
            )
            logging.info(f"  - Eval Time: {time.time() - stage_start:.1f}s")

            # Pipeline Summary
            total_time = time.time() - pipeline_start
            logging.info("=" * 70)
            logging.info("PIPELINE COMPLETE")
            logging.info(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
            logging.info(f"  Best epoch: {model_trainer_artifact.best_epoch}")
            logging.info(
                f"  Best val accuracy: {model_trainer_artifact.best_val_accuracy:.2f}%"
            )
            logging.info(
                f"  Test accuracy: {model_evaluation_artifact.model_accuracy:.2f}%"
            )
            logging.info(
                f"  Precision: {model_evaluation_artifact.precision:.4f}"
            )
            logging.info(f"  Recall: {model_evaluation_artifact.recall:.4f}")
            logging.info(f"  F1-Score: {model_evaluation_artifact.f1_score:.4f}")
            logging.info(f"  ROC-AUC: {model_evaluation_artifact.roc_auc:.4f}")
            logging.info(
                f"  Model saved: {model_trainer_artifact.trained_model_path}"
            )
            logging.info("=" * 70)

        except Exception as e:
            total_time = time.time() - pipeline_start
            logging.error(
                f"Pipeline failed after {total_time:.1f}s: {str(e)}"
            )
            raise XRayException(e, sys)
