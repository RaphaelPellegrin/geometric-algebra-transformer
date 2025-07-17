"""Not part of the original GM-CNN repo.

This file contains the GMCNNModel class, which is a model using GMCNN blocks instead of GATr blocks.
"""

from typing import Tuple, Optional

import torch
from torch import nn

from gatr.layers.gmcnn_block import GMCNNBlock


class GMCNNModel(nn.Module):
    """Model using GMCNN blocks instead of GATr blocks.
    
    Parameters
    ----------
    mv_channels : int
        Number of multivector channels
    num_blocks : int
        Number of GMCNN blocks
    group : str
        The group type ('cyclic' or 'dihedral')
    order : int
        The order of the group
    nbr_size : int
        The size of the neighborhood
    group_matrix : torch.Tensor
        The group matrix defining the group structure
    """
    
    def __init__(
        self,
        mv_channels: int,
        num_blocks: int,
        group: str,
        order: int,
        nbr_size: int,
        group_matrix: torch.Tensor,
    ) -> None:
        super().__init__()
        
        self.blocks = nn.ModuleList([
            GMCNNBlock(
                mv_channels=mv_channels,
                group=group,
                order=order,
                nbr_size=nbr_size,
                group_matrix=group_matrix,
            ) for _ in range(num_blocks)
        ])
        
        # Add any additional layers needed for your specific task
        
    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the GMCNN model.
        
        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors
        scalars : Optional[torch.Tensor]
            Input scalars (not used in this implementation)
            
        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors
        outputs_s : Optional[torch.Tensor]
            Output scalars (None in this implementation)
        """
        x = multivectors
        
        for block in self.blocks:
            x, _ = block(x)
            
        return x, None