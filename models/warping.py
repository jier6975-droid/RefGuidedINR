"""
Warping Module: Aligns reference HR features into the LR query spatial domain.
Uses deformable offsets predicted from the concatenated features,
then applies bilinear grid_sample warping.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableWarpNet(nn.Module):
    """
    Predicts spatial offsets from combined query and downsampled reference
    features, then warps the reference feature map via bilinear sampling.

    Args:
        feat_channels: Number of channels in query and reference feature maps.
    """

    def __init__(self, feat_channels: int = 64):
        super().__init__()
        # Offset prediction: takes [query_feat || ref_feat_resized]
        self.offset_net = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 2, kernel_size=3, padding=1),
        )
        # Initialize offset weights small so initial warping is near-identity
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def forward(
        self,
        ref_feat: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp reference features to align with query LR feature space.

        Args:
            ref_feat: Reference HR feature map (B, C, H_ref, W_ref).
            query_feat: Query LR feature map (B, C, H_q, W_q).

        Returns:
            Warped reference feature map of shape (B, C, H_q, W_q).
        """
        B, C, H_q, W_q = query_feat.shape

        # Resize reference feature map to query spatial size
        ref_feat_resized = F.interpolate(
            ref_feat, size=(H_q, W_q), mode="bilinear", align_corners=False
        )

        # Predict flow offsets (normalised to [-1, 1])
        combined = torch.cat([query_feat, ref_feat_resized], dim=1)
        offsets = self.offset_net(combined)  # (B, 2, H_q, W_q)

        # Build normalised sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_q, device=query_feat.device),
            torch.linspace(-1, 1, W_q, device=query_feat.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        base_grid = base_grid.expand(B, -1, -1, -1)

        # offsets in pixel-space → normalise to [-1, 1]
        offsets_norm = offsets.permute(0, 2, 3, 1)  # (B, H, W, 2)
        offsets_norm[..., 0] = offsets_norm[..., 0] / (W_q / 2)
        offsets_norm[..., 1] = offsets_norm[..., 1] / (H_q / 2)

        sampling_grid = base_grid + offsets_norm
        sampling_grid = sampling_grid.clamp(-1, 1)

        # Sample the reference feature map using the computed grid
        warped = F.grid_sample(
            ref_feat_resized,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return warped
