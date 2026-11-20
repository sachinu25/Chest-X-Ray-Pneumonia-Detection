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
    Transfer learning classifier (ResNet18 / DenseNet121 / EfficientNet-B0) pretrained on ImageNet.

    Architecture:
        - ResNet18 backbone (frozen or fine-tunable)
        - Custom classification head with dropout
        - Outputs raw logits (use with CrossEntropyLoss)

    Args:
        num_classes: Number of output classes (default: 2 for Normal/Pneumonia)
        dropout_rate: Dropout probability for regularization (default: 0.3)
        freeze_backbone: If True, freeze all backbone layers initially (default: False)
        pretrained: If True, start from ImageNet weights (training). Use False when a
            trained checkpoint will be loaded right after (inference) — avoids an
            unnecessary 45 MB download on every cold start.
    """

    # arch name -> (constructor, ImageNet weights, classifier attribute, Grad-CAM layer name)
    ARCHS = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc", "layer4"),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, "classifier", "features"),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier", "features"),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, "classifier", "features"),
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier", "features"),
    }

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        pretrained: bool = True,
        arch: str = "resnet18",
    ):
        super().__init__()
        if arch not in self.ARCHS:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(self.ARCHS)}")
        ctor, imagenet_weights, head_attr, self.gradcam_layer = self.ARCHS[arch]
        self.arch = arch

        self.backbone = ctor(weights=imagenet_weights if pretrained else None)

        # Optionally freeze backbone for feature extraction mode
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final classifier with the custom classification head
        old_head = getattr(self.backbone, head_attr)
        in_features = old_head.in_features if isinstance(old_head, nn.Linear) else old_head[-1].in_features
        new_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )
        if arch == "convnext_tiny":
            # ConvNeXt's classifier is LayerNorm2d -> Flatten -> Linear; keep the norm/flatten
            old_head[-1] = new_head
        else:
            setattr(self.backbone, head_attr, new_head)

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



