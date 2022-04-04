from __future__ import annotations
from typing import Optional, cast, overload

import torch
from torch import Tensor
import torch.distributions as td
import torch.nn.functional as F

from ranzen.torch.sampling import batched_randint
from ranzen.torch.transforms.mixup import InputsTargetsPair

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
        p: float = 0.5,
        num_classes: int | None = None,
        inplace: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        :param alpha: hyperparameter of the Beta distribution used for sampling the areas
            of the bounding boxes.

        :param num_classes: The total number of classes in the dataset that needs to be specified if
            wanting to mix up targets that are label-enoded. Passing label-encoded targets without
            specifying ``num_classes`` will result in a RuntimeError.

        :param p: The probability with which the transform will be applied to a given sample.

        :param inplace: Whether the transform should be performed in-place.

        :param seed: The PRNG seed to use for sampling pairs and bounding-box coordinates.

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
        self.generator = (
            torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
        )

    def _sample_masks(self, inputs: Tensor, *, num_samples: int):
        height, width = inputs.shape[-2:]
        lambdas = self.lambda_sampler.sample(sample_shape=torch.Size((num_samples,))).to(
            inputs.device
        )
        side_props = (1.0 - lambdas).sqrt()
        box_heights = (side_props * height).round()
        box_widths = (side_props * width).round()
        box_coords_y1 = batched_randint(height - box_heights, generator=self.generator)
        box_coords_x1 = batched_randint(width - box_widths, generator=self.generator)
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
            (y_masks.unsqueeze(1) * x_masks.unsqueeze(-1))
            .unsqueeze(1)
            .expand(-1, inputs.size(1), -1, -1)
        )
        cropped_area_ratios = (box_widths * box_widths) / (width * height)

        return masks, cropped_area_ratios

    @overload
    def _transform(self, inputs: Tensor, *, targets: Tensor) -> InputsTargetsPair:
        ...

    @overload
    def _transform(self, inputs: Tensor, *, targets: None = ...) -> Tensor:
        ...

    def _transform(
        self, inputs: Tensor, *, targets: Tensor | None = None
    ) -> Tensor | InputsTargetsPair:
        if inputs.ndim != 4:
            raise ValueError(f"'inputs' must be a batch of image tensors of shape (C, H, W).")
        batch_size = len(inputs)
        if (targets is not None) and (batch_size != len(targets)):
            raise ValueError(f"'inputs' and 'targets' must match in size at dimension 0.")

        if (batch_size == 1) or (self.p == 0):
            if targets is None:
                return inputs
            return InputsTargetsPair(inputs=inputs, targets=targets)
        elif self.p < 1:
            # Sample a mask determining which samples in the batch are to be transformed
            selected = (
                torch.rand(batch_size, device=inputs.device, generator=self.generator) < self.p
            )
            num_selected = int(selected.count_nonzero())
            indices = selected.nonzero(as_tuple=False).long().flatten()
        # if p >= 1 then the transform is always applied and we can skip
        # the above step
        else:
            num_selected = batch_size
            indices = torch.arange(batch_size, device=inputs.device, dtype=torch.long)

        # Pair each selected sample with another sample that will serve as the 'patch donor'
        pair_indices = torch.arange(batch_size).roll(1, 0)
        masks, cropped_area_ratios = self._sample_masks(inputs=inputs, num_samples=num_selected)

        if not self.inplace:
            inputs = inputs.clone()
        inputs[indices] = ~masks * inputs[indices] + masks * inputs[pair_indices]
        # No targets were recevied so we're done.
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
        # Targets need to be floats to be mixed up.
        targets = targets.float()
        target_lambdas = 1.0 - cropped_area_ratios
        target_lambdas.unsqueeze_(-1)
        targets[indices] *= target_lambdas
        targets[indices] += (1.0 - target_lambdas) * targets[pair_indices]

        return InputsTargetsPair(inputs=inputs, targets=targets)

    @overload
    def __call__(self, inputs: Tensor, *, targets: Tensor) -> InputsTargetsPair:
        ...

    @overload
    def __call__(self, inputs: Tensor, *, targets: None = ...) -> Tensor:
        ...

    def __call__(
        self, inputs: Tensor, *, targets: Tensor | None = None
    ) -> Tensor | InputsTargetsPair:
        """
        :param inputs: The samples to apply mixup to.
        :param targets: The corresponding targets to apply mixup to. If the targets are
            label-encoded then the 'num_classes' attribute cannot be None.

        :return: If target is None, the Tensor of cutmix-transformed inputs. If target is not None, a
            namedtuple containing the Tensor of cutmix-transformed inputs (inputs) and the
            corresponding Tensor of cutmix-transformed targets (targets).
        """
        return self._transform(inputs=inputs, targets=targets)
