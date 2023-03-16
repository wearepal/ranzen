from __future__ import annotations
import operator
from typing import NamedTuple
import warnings

import torch
from torch import Tensor

from ranzen.torch.sampling import batched_randint

__all__ = [
    "sample_paired_indices",
    "PairedIndices",
]


class PairedIndices(NamedTuple):
    anchors: Tensor
    matches: Tensor

    def __len__(self) -> int:
        return len(self.anchors)


def sample_paired_indices(
    inputs: Tensor,
    *,
    p: float,
    groups_or_edges: Tensor | None = None,
    cross_group: bool = False,
    generator: torch.Generator | None = None,
) -> PairedIndices | None:
    batch_size = len(inputs)
    # If the batch is singular or the sampling probability is 0 there's nothing to do.
    if (batch_size == 1) or (p == 0):
        return None
    if p < 1:
        # Sample a mask determining which samples in the batch are to be transformed
        selected = torch.rand(batch_size, device=inputs.device, generator=generator) < p
        num_selected = int(selected.count_nonzero())
        indices = selected.nonzero(as_tuple=False).long().flatten()
    # if p >= 1 then the transform is always applied and we can skip
    # the above step
    else:
        num_selected = batch_size
        indices = torch.arange(batch_size, device=inputs.device, dtype=torch.long)

    if groups_or_edges is None:
        # Sample the mixup pairs with the guarantee that a given sample will
        # not be paired with itself
        offset = torch.randint(
            low=1,
            high=batch_size,
            size=(num_selected,),
            device=inputs.device,
            dtype=torch.long,
            generator=generator,
        )
        pair_indices = (indices + offset) % batch_size
    else:
        groups_or_edges = groups_or_edges.squeeze()
        if groups_or_edges.ndim == 1:
            if len(groups_or_edges) != batch_size:
                raise ValueError(
                    "The number of elements in 'groups_or_edges' should match the size of "
                    "dimension 0 of 'inputs'."
                )

            groups = groups_or_edges.view(batch_size, 1)  # [batch_size]
            # Compute the pairwise indicator matrix, indicating whether any two samples
            # belong to the same group (0) or different groups (1)
            comp = operator.ne if cross_group else operator.eq
            connections = comp(groups[indices], groups.t())  # [num_selected, batch_size]
            if not cross_group:
                # Fill the diagonal with 'False' to prevent self-matches.
                connections.fill_diagonal_(False)
            # For each sample, compute how many other samples there are that belong
            # to a different group.
        elif groups_or_edges.ndim == 2:
            if groups_or_edges.dtype is not torch.bool:
                raise ValueError(
                    "If 'groups_or_edges' is a matrix, it must have dtype 'torch.bool'."
                )
            connections = groups_or_edges[indices]
            # Fill the diagonal with 'False' to prevent self-matches.
            connections.fill_diagonal_(False)
        else:
            raise ValueError(
                "'groups_or_edges' must be a vector denoting group membership or a boolean-type"
                "groups_or_edges matrix with elements denoting the permissibility of each"
                "possible pairing in 'inputs'."
            )
        degrees = connections.count_nonzero(dim=1)  # [num_selected]
        is_connected = degrees.nonzero().squeeze(-1)
        num_selected = len(is_connected)
        if num_selected < len(degrees):
            warnings.warn(
                (
                    "One or more samples without valid pairs according to "
                    "the connectivity defined by 'groups_or_edges'."
                ),
                RuntimeWarning,
            )
        # If there are no valid pairs (all vertices are isolated), there's nothing to do.
        if num_selected == 0:
            return None
        # Update the tensors to account for pairless samples.
        indices = indices[is_connected]
        degrees = degrees[is_connected]
        connections = connections[is_connected]
        # Sample the mixup pairs in accordance with the connectivity matrix.
        # This can be efficiently done as follows:
        # 1) Sample uniformly from {0, ..., diff_group_count - 1} to obtain the groupwise pair indices.
        # This involves first drawing samples from the standard uniform distribution, rescaling them to
        # [-1/(2*diff_group_count), diff_group_count + (1/(2*diff_group_count)], and then clamping them
        # to [0, 1], making it so that 0 and diff_group_count have the same probability of being drawn
        # as any other value. The uniform samples are then mapped to indices by multiplying by
        # diff_group_counts and rounding. 'randint' is unsuitable here because the groups aren't
        # guaranteed to have equal cardinality (using it to sample from the cyclic group,
        # Z / diff_group_count Z, as above, leads to biased sampling).
        rel_pair_indices = batched_randint(degrees, generator=generator)
        # 2) Convert the row-wise indices into row-major indices, considering only
        # only the postive entries in the rows.
        rel_pair_indices[1:] += degrees.cumsum(dim=0)[:-1]
        # 3) Finally, map from relative indices to absolute ones.
        _, abs_pos_inds = connections.nonzero(as_tuple=True)
        pair_indices = abs_pos_inds[rel_pair_indices]
    return PairedIndices(
        anchors=indices,
        matches=pair_indices,
    )
