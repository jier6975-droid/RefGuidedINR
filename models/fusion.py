"""
Fusion Module: Merges query LR features and warped reference features.
Supports 'concat' (simple channel concatenation) and 'attention'
(cross-attention where query features attend to reference features).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """
    Concatenates query and warped reference features along the channel dim,
    then projects back to feat_channels with a 1x1 convolution.
    """

    def __init__(self, feat_channels: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        query_feat: torch.Tensor,
        ref_feat_warped: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_feat: (B, C, H, W)
            ref_feat_warped: (B, C, H, W)
        Returns:
            Fused feature map of shape (B, C, H, W).
        """
        x = torch.cat([query_feat, ref_feat_warped], dim=1)
        return self.proj(x)


class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention: query features attend to reference features.
    Spatial positions serve as tokens; channels are split into heads.
    """

    def __init__(self, feat_channels: int = 64, num_heads: int = 4):
        super().__init__()
        assert feat_channels % num_heads == 0, (
            "feat_channels must be divisible by num_heads"
        )
        self.num_heads = num_heads
        self.head_dim = feat_channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Conv2d(feat_channels, feat_channels, 1)
        self.k_proj = nn.Conv2d(feat_channels, feat_channels, 1)
        self.v_proj = nn.Conv2d(feat_channels, feat_channels, 1)
        self.out_proj = nn.Conv2d(feat_channels, feat_channels, 1)

    def forward(
        self,
        query_feat: torch.Tensor,
        ref_feat_warped: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_feat: (B, C, H, W)
            ref_feat_warped: (B, C, H, W)
        Returns:
            Fused feature map of shape (B, C, H, W).
        """
        B, C, H, W = query_feat.shape
        N = H * W

        q = self.q_proj(query_feat).view(B, self.num_heads, self.head_dim, N)
        k = self.k_proj(ref_feat_warped).view(B, self.num_heads, self.head_dim, N)
        v = self.v_proj(ref_feat_warped).view(B, self.num_heads, self.head_dim, N)

        # (B, heads, N, N)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # (B, heads, head_dim, N)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        return self.out_proj(out)


class FeatureFusion(nn.Module):
    """
    Factory fusion module.

    Args:
        mode: 'concat' or 'attention'.
        feat_channels: Number of feature channels.
        num_heads: Number of attention heads (used only when mode='attention').
    """

    def __init__(
        self,
        mode: str = "concat",
        feat_channels: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.fusion = ConcatFusion(feat_channels)
        elif mode == "attention":
            self.fusion = CrossAttentionFusion(feat_channels, num_heads)
        else:
            raise ValueError(f"Unknown fusion mode: {mode!r}. Choose 'concat' or 'attention'.")

    def forward(
        self,
        query_feat: torch.Tensor,
        ref_feat_warped: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(query_feat, ref_feat_warped)
