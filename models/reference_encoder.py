"""
Reference Encoder: Encodes the reference HR image into feature maps.
Can optionally share weights with the LR backbone for parameter efficiency.
"""
import torch
import torch.nn as nn
from .liif_backbone import ResBlock


class ReferenceEncoder(nn.Module):
    """
    Encodes a reference HR image into a feature map.
    Architecture mirrors the LIIFBackbone to allow optional weight sharing.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        feat_channels: Number of output feature channels.
        num_res_blocks: Number of residual blocks.
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
            x: Reference HR image tensor of shape (B, C, H_ref, W_ref).
        Returns:
            Feature map of shape (B, feat_channels, H_ref, W_ref).
        """
        feat = self.head(x)
        feat = self.body(feat)
        feat = self.tail(feat)
        return feat
