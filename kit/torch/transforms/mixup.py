from __future__ import annotations
from enum import Enum, auto
from typing import NamedTuple, cast, overload

import torch
from torch import Tensor
import torch.distributions as td
import torch.nn.functional as F

from kit.misc import str_to_enum

__all__ = [
    "BernoulliMixUp",
    "BetaMixUp",
    "MixUpMode",
    "RandomMixUp",
    "UniformMixUp",
]


class MixUpMode(Enum):

    linear = auto()
    geometric = auto()


class InputsTargetsPair(NamedTuple):
    inputs: Tensor
    targets: Tensor


class RandomMixUp:
    r"""Apply mixup to a batch of tensors.

    Implemention of `mixup: Beyond Empirical Risk Minimization` (https://arxiv.org/abs/1710.09412).
    This implementation allows for transformation of the the input in the absence
    of labels -- this is relevant for contrastive methods that use mixup to generate
    different views of samples for instance-distance discrimination -- and additionally
    allows for different lambda samplers, different methods for mixing up samples
    (linear vs geometric) based on lambda, and cross-group pair-sampling.

    Note:
        This implementation randomly mixes images within a batch.
    """

    def __init__(
        self,
        lambda_sampler: td.Beta | td.Uniform | td.Bernoulli,
        mode: MixUpMode | str = MixUpMode.linear,
        p: float = 1.0,
        num_classes: int | None = None,
        inplace: bool = False,
    ) -> None:
        """
        Args:
            lambda_sampler: The distribution from which to sample lambda (the mixup interpolation parameter).
            mode: Which mode to use to mix up samples: geomtric or linear.
            p: The probability with which the transform will be applied to a given sample.
            num_classes: The total number of classes in the dataset that needs to be specified if wanting to
            mix up targets that are label-enoded. Passing label-encoded targets without specifying 'num_classes'
            inplace: Whether the transform should be performed in-place.
        """
        super().__init__()
        self.lambda_sampler = lambda_sampler
        if not 0 <= p <= 1:
            raise ValueError("'p' must be in the range [0, 1].")
        self.p = p
        if isinstance(mode, str):
            mode = str_to_enum(str_=mode, enum=MixUpMode)
        self.mode = mode
        if (num_classes is not None) and num_classes < 1:
            raise ValueError(f"{ num_classes } must be greater than 1.")
        self.num_classes = num_classes
        self.inplace = inplace

    def _mix(self, tensor_a: Tensor, *, tensor_b: Tensor, lambda_: Tensor) -> Tensor:
        lambda_c = 1 - lambda_
        if self.mode is MixUpMode.linear:
            return lambda_ * tensor_a + lambda_c * tensor_b
        return tensor_a ** lambda_ * tensor_b ** lambda_c

    @overload
    def _transform(
        self, inputs: Tensor, *, targets: Tensor = ..., groups: Tensor | None = ...
    ) -> InputsTargetsPair:
        ...

    @overload
    def _transform(
        self, inputs: Tensor, *, targets: None = ..., groups: Tensor | None = ...
    ) -> Tensor:
        ...

    def _transform(
        self, inputs: Tensor, *, targets: Tensor | None = None, groups: Tensor | None = None
    ) -> Tensor | InputsTargetsPair:
        batch_size = len(inputs)
        if self.p == 0:
            if targets is None:
                return inputs
            return InputsTargetsPair(inputs=inputs, targets=targets)
        elif self.p < 1:
            # Sample a mask determining which samples in the batch are to be transformed
            selected = torch.rand(batch_size, device=inputs.device) < self.p
            num_selected = int(selected.count_nonzero())
            indices = selected.nonzero(as_tuple=False).long().flatten()
        # if p >= 1 then the transform is always applied and we can skip
        # the above step
        else:
            num_selected = batch_size
            indices = torch.arange(batch_size, device=inputs.device, dtype=torch.long)

        if groups is None:
            # Sample the mixup pairs with the guarantee that a given sample will
            # not be paired with itself
            offset = torch.randint(
                low=1, high=batch_size, size=(num_selected,), device=inputs.device, dtype=torch.long
            )
            pair_indices = (indices + offset) % batch_size
        else:
            if groups.numel() != batch_size:
                raise ValueError(
                    "The number of elements in 'groups' should match the size of dimension 0 of 'inputs'."
                )
            groups = groups.view(batch_size, 1)  # [batch_size]
            # Compute the pairwise indicator matrix, indicating whether any two samples
            # belong to the same (0) or a different (1) group
            is_diff_group = groups[indices] != groups.t()  # [num_selected, batch_size]
            # For each sample, compute how many other samples there are that belong
            # to a different group.
            counts = is_diff_group.count_nonzero(dim=1)  # [num_selected]
            # Perform cross-group sampling - samples are paired exclusively with samples
            # from other groups
            rel_pair_indices = (
                torch.randint(
                    low=0,
                    high=int(counts.max()),
                    size=(num_selected,),
                    device=inputs.device,
                    dtype=torch.long,
                )
                % counts
            )
            # Convert the row-wise indices into row-major indices, considering only
            # only the postive entries in the rows
            rel_pair_indices[1:] += counts.cumsum(dim=0)[:-1]
            # Finally, map from group-relative indices to absolute ones
            _, abs_pos_inds = is_diff_group.nonzero(as_tuple=True)
            pair_indices = abs_pos_inds[rel_pair_indices]

        # Sample the mixing weights
        lambdas = self.lambda_sampler.sample(
            sample_shape=(num_selected, *((1,) * (inputs.ndim - 1)))
        ).to(inputs.device)

        if not self.inplace:
            inputs = inputs.clone()
        # Apply mixup to the inputs
        inputs[indices] = self._mix(
            tensor_a=inputs[indices], tensor_b=inputs[pair_indices], lambda_=lambdas
        )
        if targets is None:
            return inputs

        # Targets are label-encoded and need to be one-hot encoded prior to mixup.
        if torch.atleast_1d(targets.squeeze()).ndim == 1:
            if self.num_classes is None:
                raise ValueError(
                    f"{self.__class__.__name__} can only be applied to label-encoded targets if "
                    "'num_classes' is specified."
                )
            targets = cast(Tensor, F.one_hot(targets, num_classes=self.num_classes))
        # Targets need to be floats to be mixed up
        targets = targets.float()
        # Add singular dimensions to lambdas for broadcasting
        lambdas = lambdas.view(num_selected, *((1,) * (targets.ndim - 1)))
        # Apply mixup to the targets
        targets[indices] = self._mix(
            tensor_a=targets[indices], tensor_b=targets[pair_indices], lambda_=lambdas
        )
        return InputsTargetsPair(inputs, targets)

    @overload
    def __call__(
        self, inputs: Tensor, *, targets: Tensor = ..., groups: Tensor | None
    ) -> InputsTargetsPair:
        ...

    @overload
    def __call__(
        self, inputs: Tensor, *, targets: None = ..., groups: Tensor | None = ...
    ) -> Tensor:
        ...

    def __call__(
        self, inputs: Tensor, *, targets: Tensor | None = None, groups: Tensor | None = None
    ) -> Tensor | InputsTargetsPair:
        """
        Args:
            inputs: The samples to be 'mixed up'.
            targets: The corresponding targets to be mixed up. If the targets are label-encoded
            then the 'num_classes' attribute cannot be None.
            groups: Labels indicating which group each sample belongs to. If specified, mixup
            pairs will be sampled in a cross-group fashion (only samples belonging to different groups
            will be mixed up).
        """
        return self._transform(inputs=inputs, targets=targets, groups=groups)


class BetaMixUp:
    def __new__(
        cls,
        alpha: float = 0.2,
        beta: float | None = None,
        mode: MixUpMode | str = MixUpMode.linear,
        p: float = 1.0,
        num_classes: int | None = None,
        inplace: bool = False,
    ) -> RandomMixUp:
        beta = alpha if beta is None else beta
        lambda_sampler = td.Beta(concentration0=alpha, concentration1=beta)
        return RandomMixUp(
            lambda_sampler=lambda_sampler, mode=mode, p=p, num_classes=num_classes, inplace=inplace
        )


class UniformMixUp:
    def __new__(
        cls,
        low: float = 0.0,
        high: float = 1.0,
        mode: MixUpMode | str = MixUpMode.linear,
        p: float = 1.0,
        num_classes: int | None = None,
        inplace: bool = False,
    ) -> RandomMixUp:
        lambda_sampler = td.Uniform(low=low, high=high)
        return RandomMixUp(
            lambda_sampler=lambda_sampler, mode=mode, p=p, num_classes=num_classes, inplace=inplace
        )


class BernoulliMixUp:
    def __new__(
        cls,
        prob_1: float = 0.5,
        mode: MixUpMode | str = MixUpMode.linear,
        p: float = 1.0,
        num_classes: int | None = None,
        inplace: bool = False,
    ) -> RandomMixUp:
        lambda_sampler = td.Bernoulli(probs=prob_1)
        return RandomMixUp(
            lambda_sampler=lambda_sampler, mode=mode, p=p, num_classes=num_classes, inplace=inplace
        )
