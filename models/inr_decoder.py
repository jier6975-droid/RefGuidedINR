"""
INR Decoder: Coordinate-conditioned MLP that maps fused features and
continuous 2D query coordinates to RGB pixel values.

The input to the MLP at each query point is:
    [interpolated_feature, relative_cell_coords, query_coord]
following the LIIF formulation (yinboc/liif).
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class INRDecoder(nn.Module):
    """
    Implicit Neural Representation MLP decoder.

    Given a feature map and a set of continuous 2D query coordinates,
    the decoder:
      1. Samples (bilinear) feature vectors at each coordinate.
      2. Concatenates each feature vector with the query coordinate and
         the relative cell size (used for cell decoding as in LIIF).
      3. Passes the concatenated vector through an MLP to predict RGB.

    Args:
        feat_channels: Number of channels in the input feature map.
        coord_dim: Dimensionality of each query coordinate (default: 2 for 2D).
        cell_dim: Dimensionality of the cell size vector (default: 2).
        out_dim: Number of output channels (default: 3 for RGB).
        hidden_dim: Hidden size of MLP layers.
        num_layers: Number of MLP hidden layers (excluding output layer).
    """

    def __init__(
        self,
        feat_channels: int = 64,
        coord_dim: int = 2,
        cell_dim: int = 2,
        out_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        in_dim = feat_channels + coord_dim + cell_dim
        layers: List[nn.Module] = []
        last_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            last_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat: Fused feature map (B, C, H, W).
            coords: Query coordinates (B, N, 2) in range [-1, 1].
            cell: Cell size tensor (B, N, 2) representing the area each
                  query point covers (used for LIIF cell decoding).

        Returns:
            Predicted pixel values (B, N, out_dim).
        """
        B, C, H, W = feat.shape
        N = coords.shape[1]

        # Sample feature at each query coordinate via bilinear interpolation
        # coords: (B, N, 2) → grid_sample expects (B, H_out, W_out, 2)
        coords_grid = coords.unsqueeze(1)  # (B, 1, N, 2)
        sampled = F.grid_sample(
            feat,
            coords_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # (B, C, 1, N)
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, N, C)

        # Concatenate feature, coordinate, and cell
        mlp_input = torch.cat([sampled, coords, cell], dim=-1)  # (B, N, C+2+2)
        return self.mlp(mlp_input)  # (B, N, out_dim)
