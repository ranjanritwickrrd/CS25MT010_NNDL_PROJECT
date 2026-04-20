from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AODFeatureExtractor(nn.Module):
    """Compact multi-scale feature extractor used by the AOD-Net family."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), dim=1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), dim=1)
        return F.relu(self.conv5(cat3))


class OriginalAODNet(nn.Module):
    """Original AOD-Net style output equation."""

    def __init__(self, b: float = 1.0) -> None:
        super().__init__()
        self.features = AODFeatureExtractor()
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.features(x)
        output = k * x - k + self.b
        return torch.relu(output)


class ResidualAODNet(nn.Module):
    """Residual variant that predicts haze residue to subtract."""

    def __init__(self) -> None:
        super().__init__()
        self.features = AODFeatureExtractor()
        self.residual_head = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_features = self.features(x)
        residual = torch.tanh(self.residual_head(residual_features))
        return (x - residual).clamp(0.0, 1.0)


class SkipAODNet(nn.Module):
    """Skip-fusion variant that blends preserved input detail with dehazed output."""

    def __init__(self, b: float = 1.0) -> None:
        super().__init__()
        self.features = AODFeatureExtractor()
        self.gate_head = nn.Conv2d(3, 3, kernel_size=1)
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.features(x)
        base_output = torch.relu(k * x - k + self.b)
        skip_gate = torch.sigmoid(self.gate_head(k))
        return (skip_gate * base_output + (1.0 - skip_gate) * x).clamp(0.0, 1.0)


MODEL_FACTORIES = {
    "baseline": OriginalAODNet,
    "residual": ResidualAODNet,
    "skip": SkipAODNet,
}

