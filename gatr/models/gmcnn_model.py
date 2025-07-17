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

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.supports_variable_items = False

        # Create group matrix
        group_matrix = create_group_matrix(group, order)

        # Input projection to intermediate channels
        self.input_proj = nn.Linear(input_channels, mv_channels)

        # GM-CNN convolution layers
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
                for _ in range(num_blocks)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(mv_channels, output_channels)

        # Layer normalization
        self.norm = nn.LayerNorm(mv_channels)

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
        batch_size, num_items, in_channels, _ = x.shape

        # Remove the last dimension and project to intermediate channels
        x = x.squeeze(-1)  # (batch_size, num_items, input_channels)
        x = self.input_proj(x)  # (batch_size, num_items, mv_channels)

        # Prepare for GM-CNN: reshape to (batch_size * num_items, mv_channels, 1, 1)
        x = x.view(batch_size * num_items, -1, 1).unsqueeze(-1)

        # Apply GM-CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = (
                self.norm(x.squeeze(-1).view(batch_size, num_items, -1))
                .view(batch_size * num_items, -1, 1)
                .unsqueeze(-1)
            )

        # Remove the last dimension and project to output
        x = x.squeeze(-1).squeeze(-1)  # (batch_size * num_items, mv_channels)
        x = x.view(batch_size, num_items, -1)  # (batch_size, num_items, mv_channels)
        x = self.output_proj(x)  # (batch_size, num_items, output_channels)

        # Add back the last dimension for compatibility
        x = x.unsqueeze(-1)  # (batch_size, num_items, output_channels, 1)

        return x, None
