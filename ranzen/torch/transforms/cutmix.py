from __future__ import annotations
from typing import Optional, overload

import torch
from torch import Tensor
import torch.distributions as td
import torch.nn.functional as F

from ranzen.torch.sampling import batched_randint
from ranzen.torch.transforms.mixup import InputsTargetsPair
from ranzen.torch.transforms.utils import sample_paired_indices

__all__ = ["RandomCutMix"]


class RandomCutMix:
    r"""Randomly apply CutMix to a batch of images.
    PyTorch implementation of the the `CutMix`_ image-augmentation strategy.

    This implementation samples the bounding-box coordinates independently for each pair of samples
    being mixed, and, unlike other implementations, does so in a way that is fully-vectorised.

    .. _CutMix:
        https://arxiv.org/abs/1905.04899

    .. note::
        This implementation randomly mixes images within a batch.

    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        p: float = 1.0,
        num_classes: Optional[int] = None,
        inplace: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        :param alpha: hyperparameter of the Beta distribution used for sampling the areas
            of the bounding boxes.

        :param num_classes: The total number of classes in the dataset that needs to be specified if
            wanting to mix up targets that are label-enoded. Passing label-encoded targets without
            specifying ``num_classes`` will result in a RuntimeError.

        :param p: The probability with which the transform will be applied to a given sample.

        :param inplace: Whether the transform should be performed in-place.

        :param generator: Pseudo-random-number generator to use for sampling. Note that
            :class:`torch.distributions.Beta` does not accept such generator object and so
            the sampling procedure is only partially deterministic as a function of it.

        :raises ValueError: if ``p`` is not in the range [0, 1] , if ``num_classes < 1``, or if
            ``alpha`` is not a positive real number.
        """
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("'p' must be in the range [0, 1].")
        self.p = p
        if alpha < 0:
            raise ValueError("'alpha' must be positive.")
        self.alpha = alpha
        if (num_classes is not None) and num_classes < 1:
            raise ValueError(f"{ num_classes } must be greater than 1.")
        self.lambda_sampler = td.Beta(concentration0=alpha, concentration1=alpha)
        self.num_classes = num_classes
        self.inplace = inplace
        self.generator = generator

    def _sample_masks(
        self,
        inputs: Tensor,
        *,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        height, width = inputs.shape[-2:]
        lambdas: Tensor = self.lambda_sampler.sample(sample_shape=torch.Size((num_samples,))).to(
            inputs.device
        )
        side_props = torch.sqrt(1.0 - lambdas)
        box_heights = (side_props * height).round()
        box_widths = (side_props * width).round()
        box_coords_y1 = batched_randint(height - box_heights, generator=generator)
        box_coords_x1 = batched_randint(width - box_widths, generator=generator)
        # Compute the terminal y-coördinates for the bounding boxes
        box_coords_y2 = box_coords_y1 + box_heights
        box_coords_x2 = box_coords_x1 + box_widths
        # Convert the bounding box coordinates into masks.
        y_indices = torch.arange(height, device=inputs.device).unsqueeze(0).expand(num_samples, -1)
        y_masks = (box_coords_y2.unsqueeze(-1) > y_indices) & (
            y_indices >= box_coords_y1.unsqueeze(-1)
        )
        x_indices = torch.arange(width, device=inputs.device).unsqueeze(0).expand(num_samples, -1)
        x_masks = (box_coords_x2.unsqueeze(-1) > x_indices) & (
            x_indices >= box_coords_x1.unsqueeze(-1)
        )
        masks = (
            (y_masks.unsqueeze(-1) * x_masks.unsqueeze(1))
            .unsqueeze(1)
            .expand(-1, inputs.size(1), -1, -1)
        )
        cropped_area_ratios = (box_widths * box_widths) / (width * height)

        return masks, cropped_area_ratios

    @overload
    def _transform(
        self,
        inputs: Tensor,
        *,
        targets: Tensor,
        groups_or_edges: Tensor | None = ...,
        cross_group: bool = ...,
        num_classes: int | None = None,
    ) -> InputsTargetsPair:
        ...

    @overload
    def _transform(
        self,
        inputs: Tensor,
        *,
        targets: None = ...,
        groups_or_edges: Tensor | None = ...,
        cross_group: bool = ...,
        num_classes: int | None = None,
    ) -> Tensor:
        ...

    def _transform(
        self,
        inputs: Tensor,
        *,
        targets: Tensor | None = None,
        groups_or_edges: Tensor | None = None,
        cross_group: bool = False,
        num_classes: int | None = None,
    ) -> Tensor | InputsTargetsPair:
        if inputs.ndim != 4:
            raise ValueError(f"'inputs' must be a batch of image tensors of shape (C, H, W).")
        batch_size = len(inputs)
        if (targets is not None) and (batch_size != len(targets)):
            raise ValueError(f"'inputs' and 'targets' must match in size at dimension 0.")

        index_pairs = sample_paired_indices(
            inputs=inputs,
            p=self.p,
            groups_or_edges=groups_or_edges,
            cross_group=cross_group,
        )
        if index_pairs is None:
            return inputs if targets is None else InputsTargetsPair(inputs=inputs, targets=targets)
        num_selected = len(index_pairs)
        anchor_indices = index_pairs.anchors
        match_indices = index_pairs.matches
        masks, cropped_area_ratios = self._sample_masks(
            inputs=inputs, num_samples=num_selected, generator=self.generator
        )

        if not self.inplace:
            inputs = inputs.clone()
        # Trnasplant patches from the paired images to the anchor images as determined by the masks.
        inputs[anchor_indices] = ~masks * inputs[anchor_indices] + masks * inputs[match_indices]
        # No targets were recevied so we're done.
        if targets is None:
            return inputs

        # Targets are label-encoded and need to be one-hot encoded prior to mixup.
        if torch.atleast_1d(targets.squeeze()).ndim == 1:
            if num_classes is None:
                if self.num_classes is None:
                    raise RuntimeError(
                        f"{self.__class__.__name__} can only be applied to label-encoded targets "
                        "if 'num_classes' is specified."
                    )
                num_classes = self.num_classes
            elif num_classes < 1:
                raise ValueError(f"{ num_classes } must be a positive integer.")
            if num_classes > 2:
                targets = F.one_hot(targets.long(), num_classes=num_classes)
        # Targets need to be floats to be mixed up.
        targets = targets.float()
        target_lambdas = 1.0 - cropped_area_ratios
        if targets.ndim > 1:
            target_lambdas.unsqueeze_(-1)
        targets[anchor_indices] *= target_lambdas
        targets[anchor_indices] += (1.0 - target_lambdas) * targets[match_indices]

        return InputsTargetsPair(inputs=inputs, targets=targets)

    @overload
    def __call__(
        self,
        inputs: Tensor,
        *,
        targets: Tensor,
        groups_or_edges: Tensor | None = ...,
        cross_group: bool = ...,
        num_classes: int | None = ...,
    ) -> InputsTargetsPair:
        ...

    @overload
    def __call__(
        self,
        inputs: Tensor,
        *,
        targets: None = ...,
        groups_or_edges: Tensor | None = ...,
        cross_group: bool = ...,
        num_classes: int | None = ...,
    ) -> Tensor:
        ...

    def __call__(
        self,
        inputs: Tensor,
        *,
        targets: Tensor | None = None,
        groups_or_edges: Tensor | None = None,
        cross_group: bool = True,
        num_classes: int | None = None,
    ) -> Tensor | InputsTargetsPair:
        """
        :param inputs: The samples to apply mixup to.
        :param targets: The corresponding targets to apply mixup to. If the targets are
            label-encoded then the 'num_classes' attribute cannot be None.
        :param groups_or_edges: Labels indicating which group each sample belongs to or
            a boolean connectivity matrix encoding the permissibility of each possible pairing.
            In the case of the former, cutmix pairs will be sampled in a cross-group fashion (only
            samples belonging to different groups will be paired for mixup) if ``cross_group``  is
            ``True`` and sampled in a within-group fashion (only sampled belonging to the same
            groups will be paired for cutmix) otherwise.
        :param cross_group: Whether to sample cutmix pairs in a cross-group (``True``) or
            within-group (``False``) fashion (see ``groups_or_edges``).
        :param num_classes: The total number of classes in the dataset that needs to be specified if
            wanting to mix up targets that are label-enoded. Passing label-encoded targets without
            specifying ``num_classes`` will result in a RuntimeError.

        :return: If target is None, the Tensor of cutmix-transformed inputs. If target is not None, a
            namedtuple containing the Tensor of cutmix-transformed inputs (inputs) and the
            corresponding Tensor of cutmix-transformed targets (targets).
        """
        return self._transform(
            inputs=inputs,
            targets=targets,
            groups_or_edges=groups_or_edges,
            cross_group=cross_group,
            num_classes=num_classes,
        )
