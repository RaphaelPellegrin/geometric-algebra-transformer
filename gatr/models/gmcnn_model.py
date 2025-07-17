"""GM-CNN model for n-body prediction."""

from typing import Optional, Tuple
import torch
from torch import nn

from gatr.layers_gmcnn.gm_convolution.gmconv_regression import GMConvReg
from gatr.layers_gmcnn.gm_convolution.utils import create_group_matrix


class GMCNNModel(nn.Module):
    """Model using GM-CNN convolutions for n-body prediction.

    This model directly applies GM-CNN convolutions to n-body data without
    trying to fit it into the multivector structure.

    Parameters
    ----------
    mv_channels : int
        Number of intermediate channels
    num_blocks : int
        Number of GM-CNN blocks
    group : str
        The group type ('cyclic' or 'dihedral')
    order : int
        The order of the group
    nbr_size : int
        The size of the neighborhood
    input_channels : int
        Number of input channels (7 for n-body)
    output_channels : int
        Number of output channels (3 for n-body)
    """

    def __init__(
        self,
        mv_channels: int,
        num_blocks: int,
        group: str,
        order: int,
        nbr_size: int,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        print(f"[GMCNNModel] Initializing GM-CNN model")
        print(f"[GMCNNModel] Architecture: {input_channels} -> {mv_channels} -> {output_channels}")
        print(
            f"[GMCNNModel] Group: {group}, order: {order}, nbr_size: {nbr_size}, num_blocks: {num_blocks}"
        )

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.supports_variable_items = False

        # Create group matrix
        print(f"[GMCNNModel] Creating group matrix for {group} group with order {order}")
        group_matrix = create_group_matrix(group, order)
        print(f"[GMCNNModel] Group matrix shape: {group_matrix.shape}")

        # Input projection to intermediate channels
        print(f"[GMCNNModel] Creating input projection: {input_channels} -> {mv_channels}")
        self.input_proj = nn.Linear(input_channels, mv_channels)

        # GM-CNN convolution layers
        print(f"[GMCNNModel] Creating {num_blocks} GM-CNN convolution layers")
        self.conv_layers = nn.ModuleList(
            [
                GMConvReg(
                    group=group,
                    order=order,
                    nbr_size=nbr_size,
                    group_matrix=group_matrix,
                    out_channels=mv_channels,
                    error=True,
                )
                for i in range(num_blocks)
            ]
        )
        print(f"[GMCNNModel] Created {len(self.conv_layers)} convolution layers")

        # Output projection
        print(f"[GMCNNModel] Creating output projection: {mv_channels} -> {output_channels}")
        self.output_proj = nn.Linear(mv_channels, output_channels)

        # Layer normalization (commented out due to channel size mismatch)
        # self.norm = nn.LayerNorm(mv_channels)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[GMCNNModel] Model initialized with {total_params:,} trainable parameters")

    def forward(
        self,
        x: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the GM-CNN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, num_items, input_channels, 1)
        scalars : Optional[torch.Tensor]
            Scalar inputs (not used in GM-CNN)

        Returns
        -------
        outputs : torch.Tensor
            Output tensor with shape (batch_size, num_items, output_channels, 1)
        scalars : Optional[torch.Tensor]
            Scalar outputs (None for GM-CNN)
        """
        print(f"[GMCNNModel] Forward pass - Input shape: {x.shape}")
        batch_size, num_items, in_channels, _ = x.shape

        # Remove the last dimension and project to intermediate channels
        x = x.squeeze(-1)  # (batch_size, num_items, input_channels)
        print(f"[GMCNNModel] After squeeze: {x.shape}")

        x = self.input_proj(x)  # (batch_size, num_items, mv_channels)
        print(f"[GMCNNModel] After input projection: {x.shape}")

        # Prepare for GM-CNN: reshape to (batch_size * num_items, mv_channels, 1, 1)
        x = x.view(batch_size * num_items, -1, 1).unsqueeze(-1)
        print(f"[GMCNNModel] Reshaped for GM-CNN layers: {x.shape}")

        # Apply GM-CNN layers
        for i, conv_layer in enumerate(self.conv_layers):
            print(f"[GMCNNModel] Applying GM-CNN layer {i+1}/{len(self.conv_layers)}")
            x = conv_layer(x)
            print(f"[GMCNNModel] After conv layer {i+1}: {x.shape}")

            # Skip normalization for now due to channel size changes in GM-CNN
            # x_squeezed = x.squeeze(-1).contiguous()
            # x_norm = self.norm(x_squeezed.view(batch_size, num_items, -1))
            # x = x_norm.view(batch_size * num_items, -1, 1).unsqueeze(-1)
            print(f"[GMCNNModel] After conv layer {i+1} (no normalization): {x.shape}")

        # Remove the last dimension and project to output
        x = x.squeeze(-1).squeeze(-1)  # (batch_size * num_items, mv_channels)
        print(f"[GMCNNModel] Before output projection: {x.shape}")

        x = x.view(batch_size, num_items, -1)  # (batch_size, num_items, mv_channels)
        x = self.output_proj(x)  # (batch_size, num_items, output_channels)
        print(f"[GMCNNModel] After output projection: {x.shape}")

        # Add back the last dimension for compatibility
        x = x.unsqueeze(-1)  # (batch_size, num_items, output_channels, 1)
        print(f"[GMCNNModel] Final output shape: {x.shape}")

        return x, None
