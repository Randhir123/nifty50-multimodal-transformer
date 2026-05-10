"""Small CNN encoder for time-series images (GAF/MTF)."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class ImageCNNConfig:
    """Configuration for :class:`ImageCNN`."""
    image_size: int = 32
    in_channels: int = 2
    output_dim: int = 16
    dropout: float = 0.1


class ImageCNN(nn.Module):
    """2-Channel CNN encoder for GAF/MTF arrays replacing the ViT."""

    def __init__(self, config: ImageCNNConfig) -> None:
        super().__init__()
        self.config = config
        
        self.features = nn.Sequential(
            nn.Conv2d(config.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(128, config.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.projection(x)

    def encode_images(self, x: Tensor) -> Tensor:
        return self(x)