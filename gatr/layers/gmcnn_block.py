"""Not part of the original GM-CNN repo.
This file contains the GMCNNBlock class, which is a group-equivariant CNN block to replace GATrBlock.
It processes multivectors by applying GMConvReg to each blade separately,
 maintaining the geometric structure while using group-equivariant convolutions.

Since multivectors in GATr have 16 components (blades), we'll create a GMConvReg for each blade.

Key points about this implementation:
We create a separate GMConvReg layer for each blade of the multivector
The interface is kept similar to GATrBlock for easier integration
We handle the reshaping of tensors to match GMConvReg's requirements
We maintain the geometric structure by processing each blade separately
The implementation is type-hinted and documented

"""


from typing import Optional, Tuple

import torch
from torch import nn

from gatr.layers_gmcnn.gm_convolution.gmconv_regression import GMConvReg


class GMCNNBlock(nn.Module):
    """Group-equivariant CNN block to replace GATrBlock.
    
    This block processes multivectors by applying GMConvReg to each blade separately,
    maintaining the geometric structure while using group-equivariant convolutions.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    group : str
        The group type ('cyclic' or 'dihedral')
    order : int
        The order of the group
    nbr_size : int
        The size of the neighborhood
    group_matrix : torch.Tensor
        The group matrix defining the group structure
    dropout_prob : Optional[float]
        Dropout probability (not used in GMConvReg but kept for interface compatibility)
    """

    def __init__(
        self,
        mv_channels: int,
        group: str,
        order: int,
        nbr_size: int,
        group_matrix: torch.Tensor,
        dropout_prob: Optional[float] = None,
    ) -> None:
        super().__init__()
        
        # Create a GMConvReg layer for each blade (16 components in multivector)
        self.blade_conv_layers = nn.ModuleList([
            GMConvReg(
                group=group,
                order=order,
                nbr_size=nbr_size,
                group_matrix=group_matrix,
                out_channels=mv_channels,
                error=True
            ) for _ in range(16)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(mv_channels)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,  # Kept for interface compatibility
        reference_mv: Optional[torch.Tensor] = None,  # Kept for interface compatibility
        additional_qk_features_mv: Optional[torch.Tensor] = None,  # Not used in GMConvReg
        additional_qk_features_s: Optional[torch.Tensor] = None,  # Not used in GMConvReg
        attention_mask: Optional[torch.Tensor] = None,  # Not used in GMConvReg
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the GMCNN block.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors with shape (..., items, channels, 16)
        scalars : Optional[torch.Tensor]
            Input scalars (not used but kept for interface compatibility)
        reference_mv : Optional[torch.Tensor]
            Reference multivector (not used but kept for interface compatibility)
        additional_qk_features_mv : Optional[torch.Tensor]
            Additional Q/K features (not used in GMConvReg)
        additional_qk_features_s : Optional[torch.Tensor]
            Additional Q/K features (not used in GMConvReg)
        attention_mask : Optional[torch.Tensor]
            Attention mask (not used in GMConvReg)

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors with shape (..., items, channels, 16)
        outputs_s : Optional[torch.Tensor]
            Output scalars (None in this implementation)
        """
        # Reshape input for GMConvReg
        batch_size = multivectors.shape[0]
        num_items = multivectors.shape[1]
        
        # Process each blade separately
        processed_blades = []
        for blade_idx in range(16):
            # Extract the current blade
            blade = multivectors[..., blade_idx]  # Shape: (..., items, channels)
            
            # Reshape for GMConvReg: (batch_size, channels, 1, vec_size)
            blade = blade.reshape(batch_size, num_items, -1, 1)
            blade = blade.permute(0, 2, 3, 1)  # (batch_size, channels, 1, items)
            
            # Process through GMConvReg
            processed_blade = self.blade_conv_layers[blade_idx](blade)
            
            # Reshape back
            processed_blade = processed_blade.permute(0, 3, 1, 2)  # (batch_size, items, channels, 1)
            processed_blade = processed_blade.reshape(batch_size, num_items, -1)
            
            processed_blades.append(processed_blade)
        
        # Stack processed blades
        outputs_mv = torch.stack(processed_blades, dim=-1)
        
        # Apply layer normalization
        outputs_mv = self.norm(outputs_mv)
        
        return outputs_mv, None  # Return None for scalars to maintain interface compatibility