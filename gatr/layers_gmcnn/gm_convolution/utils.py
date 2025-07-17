"""Not part of the original GM-CNN repo.
This file contains the utils for the GMCNN.
"""

import numpy as np
import torch
from typing import Union, Literal


def create_group_matrix(group: Literal["cyclic", "dihedral"], order: int) -> np.ndarray:
    """Creates the group matrix for GMCNN.

    Parameters
    ----------
    group : str
        The group type ('cyclic' or 'dihedral')
    order : int
        The order of the group

    Returns
    -------
    group_matrix : np.ndarray
        The group matrix as numpy array (required by GM-CNN layers)
    """
    if group == "cyclic":
        # Create cyclic group matrix
        matrix = np.zeros((order, order))
        for i in range(order):
            matrix[i, (i + 1) % order] = 1
    elif group == "dihedral":
        # Create dihedral group matrix
        matrix = np.zeros((2 * order, 2 * order))
        for i in range(order):
            matrix[i, (i + 1) % order] = 1
            matrix[i + order, ((i + 1) % order) + order] = 1
            matrix[i, i + order] = 1
            matrix[i + order, i] = 1
    else:
        raise ValueError(f"Unknown group type: {group}")

    return matrix
