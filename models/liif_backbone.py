"""
LIIF Backbone: Feature extractor for query LR images.
Based on yinboc/liif (https://github.com/yinboc/liif).
Uses a lightweight residual encoder to produce feature maps.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class LIIFBackbone(nn.Module):
    """
    Feature extractor for query LR images, adapted from yinboc/liif.
    Produces a dense feature map at the LR resolution.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        feat_channels: Number of output feature channels.
        num_res_blocks: Number of residual blocks in the encoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 64,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[ResBlock(feat_channels) for _ in range(num_res_blocks)]
        )
        self.tail = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR image tensor of shape (B, C, H, W).
        Returns:
            Feature map of shape (B, feat_channels, H, W).
        """
        feat = self.head(x)
        feat = self.body(feat)
        feat = self.tail(feat)
        return feat
