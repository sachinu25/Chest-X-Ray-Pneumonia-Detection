"""
Model architectures for X-ray Pneumonia Classification.

Provides:
    - XRayClassifier: Production model using ResNet18 transfer learning (recommended)
    - Net: Legacy custom CNN (kept for backward compatibility only)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class XRayClassifier(nn.Module):
    """
    Transfer learning classifier using ResNet18 pretrained on ImageNet.

    Architecture:
        - ResNet18 backbone (frozen or fine-tunable)
        - Custom classification head with dropout
        - Outputs raw logits (use with CrossEntropyLoss)

    Args:
        num_classes: Number of output classes (default: 2 for Normal/Pneumonia)
        dropout_rate: Dropout probability for regularization (default: 0.3)
        freeze_backbone: If True, freeze all backbone layers initially (default: False)
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Load pretrained ResNet18 backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Optionally freeze backbone for feature extraction mode
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer with custom classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (no softmax/sigmoid)."""
        return self.backbone(x)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



