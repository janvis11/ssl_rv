from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SimCLR(nn.Module):
    def __init__(self, feature_dim: int = 512, projection_dim: int = 128):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=False)
        self.projector = ProjectionHead(
            in_dim=self.encoder.feature_dim,
            hidden_dim=256,
            out_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        projections = self.projector(features)
        projections = nn.functional.normalize(projections, dim=1)
        return features, projections


class LinearClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        feature_dim = getattr(encoder, "feature_dim", 512)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits