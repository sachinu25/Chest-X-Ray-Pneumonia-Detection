"""
Data Transformation component.

Handles image augmentation, normalization, and DataLoader creation.
Supports train/validation/test splits with proper separation of transforms.

Key improvements over the original:
    - Stronger augmentation: RandomResizedCrop, GaussianBlur, RandomErasing
    - WeightedRandomSampler for balanced training batches (fixes class imbalance)
    - Class weight computation passed to the trainer for weighted loss
"""

import os
import sys
from typing import List, Tuple

import joblib
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from xray.constant.training_pipeline import SEED
from xray.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from xray.entity.config_entity import DataTransformationConfig
from xray.exception import XRayException
from xray.logger import logging


class DataTransformation:
    """
    Applies image transformations and creates DataLoaders.

    - Training data: strong augmentation (RandomResizedCrop, flip, rotation,
      jitter, blur, erasing, normalize) + WeightedRandomSampler
    - Validation/Test data: only resize + center crop + normalize
    - Creates a validation split from training data for proper evaluation
    - Computes class weights for weighted CrossEntropyLoss
    """

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def transforming_training_data(self) -> transforms.Compose:
        """
        Build augmentation pipeline for training images.

        Uses RandomResizedCrop instead of Resize+CenterCrop for scale invariance.
        Adds GaussianBlur and RandomErasing for stronger regularization.
        """
        try:
            logging.info("Building training data transforms")

            train_transform = transforms.Compose(
                [
                    # RandomResizedCrop learns scale invariance — better than
                    # fixed Resize+CenterCrop for training
                    transforms.RandomResizedCrop(
                        self.data_transformation_config.CENTERCROP,
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.05),
                    transforms.RandomRotation(
                        self.data_transformation_config.RANDOMROTATION
                    ),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
                    ),
                    transforms.ColorJitter(
                        **self.data_transformation_config.color_jitter_transforms
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                        p=0.2,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                    # RandomErasing forces the model to use global features,
                    # not rely on a single discriminative region
                    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
                ]
            )

            logging.info("Training transforms built successfully")
            return train_transform

        except Exception as e:
            raise XRayException(e, sys)

    def transforming_testing_data(self) -> transforms.Compose:
        """Build transform pipeline for validation/test images (no augmentation)."""
        logging.info("Building test/validation data transforms")

        try:
            test_transform = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

            logging.info("Test/validation transforms built successfully")
            return test_transform

        except Exception as e:
            raise XRayException(e, sys)

    @staticmethod
    def _compute_class_weights(targets: List[int], num_classes: int) -> List[float]:
        """
        Compute inverse-frequency class weights for CrossEntropyLoss.

        For a dataset with 1079 NORMAL and 3107 PNEUMONIA:
            weight_NORMAL  = total / (num_classes * count_NORMAL)  = 4186 / (2 * 1079) = 1.94
            weight_PNEUMONIA = total / (num_classes * count_PNEUMONIA) = 4186 / (2 * 3107) = 0.67

        This makes NORMAL errors cost ~3x more, forcing the model to learn
        NORMAL features instead of always predicting PNEUMONIA.
        """
        class_counts = [0] * num_classes
        for t in targets:
            class_counts[t] += 1

        total = sum(class_counts)
        weights = []
        for count in class_counts:
            if count > 0:
                weights.append(total / (num_classes * count))
            else:
                weights.append(1.0)

        return weights

    @staticmethod
    def _build_sampler(targets: List[int], num_classes: int) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler that oversamples the minority class.

        Each sample is assigned a weight inversely proportional to its class
        frequency, so every batch has roughly balanced class representation.
        This prevents gradient domination by the majority class.
        """
        class_counts = [0] * num_classes
        for t in targets:
            class_counts[t] += 1

        # Weight per class = 1 / count
        class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]

        # Assign each sample the weight of its class
        sample_weights = [class_weights[t] for t in targets]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def data_loader(
        self, train_transform: transforms.Compose, test_transform: transforms.Compose
    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[float], int]:
        """
        Create DataLoaders for train, validation, and test splits.

        Returns:
            Tuple of (train_loader, val_loader, test_loader, class_weights, num_train_samples)

        The validation set is carved out of the training data using a random split
        to ensure the test set is never seen during training or hyperparameter tuning.
        """
        try:
            logging.info("Creating DataLoaders")

            # Load full training dataset with train transforms
            full_train_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=train_transform,
            )

            # Create validation split from training data
            val_split = self.data_transformation_config.validation_split
            total_size = len(full_train_data)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size

            generator = torch.Generator().manual_seed(SEED)
            train_indices, val_indices = torch.utils.data.random_split(
                range(total_size),
                [train_size, val_size],
                generator=generator,
            )

            # Validation data uses test transforms (no augmentation)
            val_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=test_transform,
            )

            train_subset = Subset(full_train_data, train_indices.indices)
            val_subset = Subset(val_data, val_indices.indices)

            test_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform=test_transform,
            )

            logging.info(
                f"Dataset sizes — Train: {len(train_subset)}, "
                f"Val: {len(val_subset)}, Test: {len(test_data)}"
            )

            # Extract targets for the training subset
            train_targets = [full_train_data.targets[i] for i in train_indices.indices]
            num_classes = len(full_train_data.classes)

            # Log class distribution
            class_names = full_train_data.classes
            for idx, name in enumerate(class_names):
                count = sum(1 for t in train_targets if t == idx)
                logging.info(f"  Train class '{name}': {count} samples")

            # Compute class weights for weighted CrossEntropyLoss
            class_weights = self._compute_class_weights(train_targets, num_classes)
            logging.info(f"  Class weights for loss: {class_weights}")

            # Build WeightedRandomSampler for balanced batches
            sampler = self._build_sampler(train_targets, num_classes)
            logging.info("  WeightedRandomSampler created for balanced training")

            # Train loader uses sampler (shuffle must be False when using sampler)
            train_loader_params = {
                **self.data_transformation_config.train_loader_params,
                "sampler": sampler,
            }
            train_loader = DataLoader(train_subset, **train_loader_params)

            val_loader = DataLoader(
                val_subset, **self.data_transformation_config.test_loader_params
            )
            test_loader = DataLoader(
                test_data, **self.data_transformation_config.test_loader_params
            )

            logging.info("DataLoaders created successfully")
            return train_loader, val_loader, test_loader, class_weights, len(train_subset)

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Run the full data transformation pipeline."""
        try:
            logging.info("Starting data transformation pipeline")

            train_transform = self.transforming_training_data()
            test_transform = self.transforming_testing_data()

            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            joblib.dump(
                train_transform, self.data_transformation_config.train_transforms_file
            )
            joblib.dump(
                test_transform, self.data_transformation_config.test_transforms_file
            )

            train_loader, val_loader, test_loader, class_weights, num_train = (
                self.data_loader(
                    train_transform=train_transform, test_transform=test_transform
                )
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
                transformed_val_object=val_loader,
                train_transform_file_path=self.data_transformation_config.train_transforms_file,
                test_transform_file_path=self.data_transformation_config.test_transforms_file,
                class_weights=class_weights,
                num_train_samples=num_train,
            )

            logging.info("Data transformation pipeline completed")
            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
