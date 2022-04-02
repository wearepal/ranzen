from __future__ import annotations
from enum import Enum, auto
from typing import Generic, NamedTuple, TypeVar, cast, overload

import torch
from torch import Tensor
import torch.distributions as td
import torch.nn.functional as F

from ranzen.misc import str_to_enum
from ranzen.torch.sampling import batched_randint

__all__ = [
    "MixUpMode",
    "RandomMixUp",
]


class MixUpMode(Enum):
    """An enum for the mix-up mode."""

    linear = auto()
    """linear mix-up"""
    geometric = auto()
    """geometric mix-up"""


class InputsTargetsPair(NamedTuple):
    inputs: Tensor
    targets: Tensor


LS = TypeVar("LS", td.Beta, td.Bernoulli, td.Uniform)


class RandomMixUp(Generic[LS]):
    r"""Apply mixup to tensors within a batch with some probability.

    PyTorch implemention of `mixup`_.
    This implementation allows for transformation of the the input in the absence
    of labels (this is relevant, for instance,to contrastive methods that use mixup to generate
    different views of samples to enable instance-discrimination) and additionally allows for
    different lambda-samplers, different methods for mixing up samples (linear vs. geometric)
    based on lambda, and cross-group pair-sampling. Furthermore, unlike the official implementation,
    samples are guaranteed not to be paired with themselves.

    .. _mixup:
        https://arxiv.org/abs/1904.00962v5

    .. note::
        This implementation randomly mixes images within a batch.
    """

    def __init__(
        self,
        lambda_sampler: LS,
        *,
        mode: MixUpMode | str = MixUpMode.linear,
        p: float = 1.0,
        num_classes: int | None = None,
        featurewise: bool = False,
        inplace: bool = False,
    ) -> None:
        """
        :param lambda_sampler: The distribution from which to sample lambda (the mixup interpolation
            parameter).

        :param mode: Which mode to use to mix up samples: geometric or linear.

        .. note::
            The (weighted) geometric mean, enabled by ``mode=geometric``, is only valid for positive
            inputs.

        :param p: The probability with which the transform will be applied to a given sample.
        :param num_classes: The total number of classes in the dataset that needs to be specified if
            wanting to mix up targets that are label-enoded. Passing label-encoded targets without
            specifying ``num_classes`` will result in a RuntimeError.

        :param featurewise: Whether to sample sample feature-wise instead of sample-wise.

        .. note::
            If the ``lambda_sampler`` is a BernoulliDistribution, then featurewise sampling will
            always be enabled.

        :param inplace: Whether the transform should be performed in-place.

        :raises ValueError: if ``p`` is not in the range [0, 1] or ``num_classes < 1``.
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
        self.featurewise = featurewise or isinstance(lambda_sampler, td.Bernoulli)
        self.inplace = inplace

    @classmethod
    def with_beta_dist(
        cls: type[RandomMixUp],
        alpha: float = 0.2,
        *,
        beta: float | None = None,
        mode: MixUpMode | str = MixUpMode.linear,
        p: float = 1.0,
        num_classes: int | None = None,
        inplace: bool = False,
        featurewise: bool = False,
    ) -> RandomMixUp[td.Beta]:
        """
        Instantiate a :class:`RandomMixUp` with a Beta-distribution sampler.

        :param alpha: 1st concentration parameter of the distribution. Must be positive
        :param beta:  2nd concentration parameter of the distribution. Must be positive.
            If ``None``, then the parameter will be set to ``alpha``.

        :param mode: Which mode to use to mix up samples: geometric or linear.

        .. note::
            The (weighted) geometric mean, enabled by ``mode=geometric``, is only valid for positive
            inputs.

        :param p: The probability with which the transform will be applied to a given sample.
        :param num_classes: The total number of classes in the dataset that needs to be specified if
            wanting to mix up targets that are label-enoded. Passing label-encoded targets without
            specifying ``num_classes`` will result in a RuntimeError.
        :param featurewise: Whether to sample sample feature-wise instead of sample-wise.
        :param inplace: Whether the transform should be performed in-place.
        :return: A :class:`RandomMixUp` instance with ``lambda_sampler`` set to a  Beta-distribution
            with ``concentration1=alpha`` and ``concentration0=beta``.
        """
        beta = alpha if beta is None else beta
        lambda_sampler = td.Beta(concentration0=alpha, concentration1=beta)
        return cls(
            lambda_sampler=lambda_sampler,
            mode=mode,
            p=p,
            num_classes=num_classes,
            inplace=inplace,
            featurewise=featurewise,
        )

    @classmethod
    def with_uniform_dist(
        cls: type[RandomMixUp],
        low: float = 0.0,
        *,
        high: float = 1.0,
        mode: MixUpMode | str = MixUpMode.linear,
        p: float = 1.0,
        num_classes: int | None = None,
        inplace: bool = False,
        featurewise: bool = False,
    ) -> RandomMixUp[td.Uniform]:
        """
        Instantiate a :class:`RandomMixUp` with a uniform-distribution sampler.

        :param low: Lower range (inclusive).
        :param high: Upper range (inclusive).
        :param mode: Which mode to use to mix up samples: geometric or linear.

        .. note::
            The (weighted) geometric mean, enabled by ``mode=geometric``, is only valid for positive
            inputs.

        :param p: The probability with which the transform will be applied to a given sample.
        :param num_classes: The total number of classes in the dataset that needs to be specified if
            wanting to mix up targets that are label-enoded. Passing label-encoded targets without
            specifying ``num_classes`` will result in a RuntimeError.
        :param featurewise: Whether to sample sample feature-wise instead of sample-wise.
        :param inplace: Whether the transform should be performed in-place.

        :return: A :class:`RandomMixUp` instance with ``lambda_sampler`` set to a
            Uniform-distribution with ``low=low`` and ``high=high``.
        """
        lambda_sampler = td.Uniform(low=low, high=high)
        return cls(
            lambda_sampler=lambda_sampler,
            mode=mode,
            p=p,
            num_classes=num_classes,
            inplace=inplace,
            featurewise=featurewise,
        )

    @classmethod
    def with_bernoulli_dist(
        cls: type[RandomMixUp],
        prob_1: float = 0.5,
        *,
        mode: MixUpMode | str = MixUpMode.linear,
        p: float = 1.0,
        num_classes: int | None = None,
        inplace: bool = False,
    ) -> RandomMixUp[td.Bernoulli]:
        """
        Instantiate a :class:`RandomMixUp` with a Bernoulli-distribution sampler.

        :param prob_1: The probability of sampling 1.
        :param mode: Which mode to use to mix up samples: geometric or linear.

        .. note::
            The (weighted) geometric mean, enabled by ``mode=geometric``, is only valid for positive
            inputs.

        :param p: The probability with which the transform will be applied to a given sample.
        :param num_classes: The total number of classes in the dataset that needs to be specified if
            wanting to mix up targets that are label-enoded. Passing label-encoded targets without
            specifying ``num_classes`` will result in a RuntimeError.

        :param inplace: Whether the transform should be performed in-place.
        :return: A :class:`RandomMixUp` instance with ``lambda_sampler`` set to a
            Bernoulli-distribution with ``probs=prob_1``.
        """
        lambda_sampler = td.Bernoulli(probs=prob_1)
        return cls(
            lambda_sampler=lambda_sampler, mode=mode, p=p, num_classes=num_classes, inplace=inplace
        )

    def _mix(self, tensor_a: Tensor, *, tensor_b: Tensor, lambda_: Tensor) -> Tensor:
        lambda_c = 1 - lambda_
        if self.mode is MixUpMode.linear:
            return lambda_ * tensor_a + lambda_c * tensor_b
        return tensor_a**lambda_ * tensor_b**lambda_c

    @overload
    def _transform(
        self, inputs: Tensor, *, targets: Tensor, group_labels: Tensor | None = ...
    ) -> InputsTargetsPair:
        ...

    @overload
    def _transform(
        self, inputs: Tensor, *, targets: None = ..., group_labels: Tensor | None = ...
    ) -> Tensor:
        ...

    def _transform(
        self, inputs: Tensor, *, targets: Tensor | None = None, group_labels: Tensor | None = None
    ) -> Tensor | InputsTargetsPair:
        batch_size = len(inputs)
        # If the batch is singular or the sampling probability is 0 there's nothing to do.
        if (batch_size == 1) or (self.p == 0):
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

        if group_labels is None:
            # Sample the mixup pairs with the guarantee that a given sample will
            # not be paired with itself
            offset = torch.randint(
                low=1, high=batch_size, size=(num_selected,), device=inputs.device, dtype=torch.long
            )
            pair_indices = (indices + offset) % batch_size
        else:
            if group_labels.numel() != batch_size:
                raise ValueError(
                    "The number of elements in 'group_labels' should match the size of dimension 0 of 'inputs'."
                )
            group_labels = group_labels.view(batch_size, 1)  # [batch_size]
            # Compute the pairwise indicator matrix, indicating whether any two samples
            # belong to the same group (0) or different groups (1)
            is_diff_group = group_labels[indices] != group_labels.t()  # [num_selected, batch_size]
            # For each sample, compute how many other samples there are that belong
            # to a different group.
            diff_group_counts = is_diff_group.count_nonzero(dim=1)  # [num_selected]
            if torch.any(diff_group_counts == 0):
                raise RuntimeError(
                    f"No samples from different groups to sample as mixup pairs for one or more groups."
                )
            # Sample the mixup pairs via cross-group sampling, meaning samples are paired exclusively
            # with samples from other groups. This can be efficiently done as follows:
            # 1) Sample uniformly from {0, ..., diff_group_count - 1} to obtain the groupwise pair indices.
            # This involves first drawing samples from the standard uniform distribution, rescaling them to
            # [-1/(2*diff_group_count), diff_group_count + (1/(2*diff_group_count)], and then clamping them
            # to [0, 1], making it so that 0 and diff_group_count have the same probability of being drawn
            # as any other value. The uniform samples are then mapped to indices by multiplying by
            # diff_group_counts and rounding. 'randint' is unsuitable here because the groups aren't
            # guaranteed to have equal cardinality (using it to sample from the cyclic group,
            # Z / diff_group_count Z, as above, leads to biased sampling).
            rel_pair_indices = batched_randint(diff_group_counts)
            # 2) Convert the row-wise indices into row-major indices, considering only
            # only the postive entries in the rows.
            rel_pair_indices[1:] += diff_group_counts.cumsum(dim=0)[:-1]
            # 3) Finally, map from group-relative indices to absolute ones.
            _, abs_pos_inds = is_diff_group.nonzero(as_tuple=True)
            pair_indices = abs_pos_inds[rel_pair_indices]

        # Sample the mixing weights
        if self.featurewise:
            sample_shape = (num_selected, *inputs.shape[1:])
        else:
            sample_shape = (num_selected, *((1,) * (inputs.ndim - 1)))
        lambdas = self.lambda_sampler.sample(sample_shape=torch.Size(sample_shape)).to(
            inputs.device
        )

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
                raise RuntimeError(
                    f"{self.__class__.__name__} can only be applied to label-encoded targets if "
                    "'num_classes' is specified."
                )
            targets = cast(Tensor, F.one_hot(targets, num_classes=self.num_classes))
        elif not self.inplace:
            targets = targets.clone()
        # Targets need to be floats to be mixed up
        targets = targets.float()
        # Use the empirical mean of the lambdas for interpolating the targets if the lambdas
        # were sampled feasture-wise, else just use the lambdas as is.
        target_lambdas = lambdas = (
            lambdas.flatten(start_dim=1).mean(1) if self.featurewise else lambdas
        )
        # Add singular dimensions to lambdas for broadcasting
        target_lambdas = lambdas.view(num_selected, *((1,) * (targets.ndim - 1)))
        # Apply mixup to the targets
        targets[indices] = self._mix(
            tensor_a=targets[indices], tensor_b=targets[pair_indices], lambda_=target_lambdas
        )
        return InputsTargetsPair(inputs, targets)

    @overload
    def __call__(
        self, inputs: Tensor, *, targets: Tensor, group_labels: Tensor | None
    ) -> InputsTargetsPair:
        ...

    @overload
    def __call__(
        self, inputs: Tensor, *, targets: None = ..., group_labels: Tensor | None = ...
    ) -> Tensor:
        ...

    def __call__(
        self, inputs: Tensor, *, targets: Tensor | None = None, group_labels: Tensor | None = None
    ) -> Tensor | InputsTargetsPair:
        """
        :param inputs: The samples to apply mixup to.
        :param targets: The corresponding targets to apply mixup to. If the targets are
            label-encoded then the 'num_classes' attribute cannot be None.
        :param group_labels: Labels indicating which group each sample belongs to. If specified,
            mixup pairs will be sampled in a cross-group fashion (only samples belonging to
            different groups will be paired for mixup).

        :return: If target is None, the Tensor of mixup-transformed inputs. If target is not None, a
            namedtuple containing the Tensor of mixup-transformed inputs (inputs) and the
            corresponding Tensor of mixup-transformed targets (targets).
        """
        return self._transform(inputs=inputs, targets=targets, group_labels=group_labels)
