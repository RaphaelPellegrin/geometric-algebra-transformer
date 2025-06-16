"""This file contains the implementation of the permutation group element."""

from .group_element import GroupElement


class PermutationElement(GroupElement):
    """
    Represents an element of the permutation group S_n.
    A permutation is stored as a list where permutation[i] gives the image of i under the permutation.
    For example, [1,2,0] represents the permutation that maps 0->1, 1->2, 2->0.
    """

    def __init__(self, permutation):
        """Initialize a permutation element.

        Args:
            permutation (list[int]): A list representing the permutation, where
                                   permutation[i] gives the image of i under the permutation.
        """
        self.permutation = permutation

    def product(self, other):
        """Compute the composition (product) of two permutations.

        The product p1 * p2 means "apply p2 first, then p1".
        For example, if p1 = [1,0] and p2 = [1,0], then (p1 * p2)[i] = p1[p2[i]].

        Args:
            other (PermutationElement): Another permutation to compose with.

        Returns:
            PermutationElement: The composition of self with other.
        """
        product = [self.permutation[i] for i in other.permutation]
        return PermutationElement(product)

    def inverse(self):
        """Compute the inverse permutation.

        For a permutation p, finds p^(-1) such that p * p^(-1) = identity.
        The inverse is computed by: if p[i] = j, then p^(-1)[j] = i.

        Example:
            # index -> permutation
            If permutation = [0,2,1] # 0->0, 2->1, 1->2
            Index:     [0,1,2]
            Maps to:   [0,2,1]
            Then inverse will be:
            # permutation -> index
            Index:     [0,1,2]
            Maps to:   [0,2,1]  # 0->0, 1->2, 2->1

        Returns:
            PermutationElement: The inverse permutation.
        """
        inverse = [self.permutation.index(i) for i in range(len(self.permutation))]
        return PermutationElement(inverse)

    def __str__(self):
        """Convert the permutation to a string representation.

        Returns:
            str: The permutation list as a string.
        """
        return str(self.permutation)
