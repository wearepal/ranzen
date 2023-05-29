from __future__ import annotations
from enum import Enum, auto
from functools import partial
from typing import Optional, Union

from torch import Tensor, nn
import torch.nn.functional as F

from ranzen import parsable, str_to_enum

__all__ = [
    "CrossEntropyLoss",
    "ReductionType",
    "cross_entropy_loss",
    "reduce",
]


class ReductionType(Enum):
    """An enum for the type of reduction to apply to a batch of losses."""

    mean = auto()
    """compute the mean over all dimensions."""
    none = auto()
    """no reduction."""
    sum = auto()
    """compute the sum over all dimensions."""
    batch_mean = auto()
    """compute the mean over the batch (first) dimension, the sum over the remaining dimensions."""


def reduce(losses: Tensor, reduction_type: ReductionType | str) -> Tensor:
    if isinstance(reduction_type, str):
        reduction_type = str_to_enum(str_=reduction_type, enum=ReductionType)
    if reduction_type is ReductionType.mean:
        return losses.mean()
    elif reduction_type is ReductionType.batch_mean:
        return losses.sum() / losses.size(0)
    elif reduction_type is ReductionType.sum:
        return losses.sum()
    elif reduction_type is ReductionType.none:
        return losses
    raise TypeError(
        f"Received invalid type '{type(reduction_type)}' for argument 'reduction_type'."
    )


def cross_entropy_loss(
    input: Tensor,
    *,
    target: Tensor,
    instance_weight: Tensor | None = None,
    reduction: ReductionType | str = ReductionType.mean,
    ignore_index: int = -100,
    class_weight: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""This criterion computes the cross entropy loss between input and target.

    See :class:`~CrossEntropyLoss` for details.

    :param input: Predicted unnormalized scores (often referred to as logits).
    :param target: Ground truth class indices or class probabilities.

    :param instance_weight: a manual rescaling weight given to each sample. If given, has to be a
        Tensor of 'N'.

    :param class_weight: A manual rescaling weight given to each class. If given, has to be a
        Tensor of size `C`.

    :param ignore_index: Specifies a target value that is ignored and does not contribute to the
        input gradient. Note that :attr:`ignore_index` is only applicable when the target contains
        class indices.

    :param reduction: Specifies the reduction to apply to the output.

    :param label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing
        the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground
        truth and a uniform distribution as described in `Rethinking the Inception Architecture for
        Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    :returns: The (reduced) cross-entropy between ``input`` and ``target``.

    :raises ValueError: If 'input' and 'target' have incompatible sizes.

    :example:
        >>> # Example of target with class indices
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randint(5, (3,), dtype=torch.int64)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
    """
    if isinstance(reduction, str):
        reduction = str_to_enum(str_=reduction, enum=ReductionType)

    input = input.view(input.size(0), -1).squeeze(-1)
    target = target.view(target.size(0), -1).squeeze(-1)

    if input.ndim == 1:  # Binary classification
        if target.ndim == 2:
            if target.size(1) == 2:
                target = target[:, 1]
            else:
                raise ValueError(
                    "'target' must be of size '2' at dimension '1' if not label encoded."
                )
        elif target.ndim > 2:
            raise ValueError(
                "'target' must be a one- or two-dimensional tensor when 'input' is one-dimensional"
                " (excluding dummy dimensions) and corresponds to binary predictions."
            )
        if not target.is_floating_point():
            target = target.float()
        loss_fn = F.binary_cross_entropy_with_logits
    else:  # Multiclass classification
        if (target.ndim == 1) and target.is_floating_point():
            target = target.long()
        elif target.ndim == 2:
            if target.shape != input.shape:
                raise ValueError(
                    "'target' and 'input' must match in size when 'target' is not label encoded."
                )
            elif not target.is_floating_point():
                target = target.to(input.dtype)

        loss_fn = partial(
            F.cross_entropy,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    losses = loss_fn(
        input=input,
        target=target,
        weight=class_weight,
        reduction="none",
    )
    if instance_weight is not None:
        losses *= instance_weight.view_as(losses)
    return reduce(losses=losses, reduction_type=reduction)


class CrossEntropyLoss(nn.Module):
    weight: Tensor | None

    @parsable
    def __init__(
        self,
        *,
        class_weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: Union[ReductionType, str] = ReductionType.mean,
        label_smoothing: float = 0.0,
    ) -> None:
        r"""This criterion computes the cross entropy loss between input and target.

        It is useful when training a classification problem with `C` classes.
        If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
        assigning weight to each of the classes.
        This is particularly useful when you have an unbalanced training set.

        The `input` is expected to contain raw, unnormalized scores for each class.
        `input` has to be a Tensor of size :math:`(C)` for unbatched input,
        :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
        `K`-dimensional case. The last being useful for higher dimension inputs, such
        as computing cross entropy loss per-pixel for 2D images.

        The `target` that this criterion expects should contain either:

        - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
          `ignore_index` is specified, this loss also accepts this class index (this index
          may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
          set to ``'none'``) loss for this case can be described as:

          .. math::
              \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
              l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
              \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

          where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
          :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
          :math:`d_1, ..., d_k` for the `K`-dimensional case. If
          :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

          .. math::
              \ell(x, y) = \begin{cases}
                  \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
                   \text{if reduction} = \text{`mean';}\\
                    \sum_{n=1}^N l_n,  &
                    \text{if reduction} = \text{`sum'.}
                \end{cases}

          Note that this case is equivalent to the combination of :class:`torch.nn.LogSoftmax` and
          :class:`torch.nn.NLLLoss`.

        - Probabilities for each class; useful when labels beyond a single class per minibatch item
          are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
          :attr:`reduction` set to ``'none'``) loss for this case can be described as:

          .. math::
              \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
              l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

          where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
          :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
          :math:`d_1, ..., d_k` for the `K`-dimensional case. If
          :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

          .. math::
              \ell(x, y) = \begin{cases}
                  \frac{\sum_{n=1}^N l_n}{N}, &
                   \text{if reduction} = \text{`mean';}\\
                    \sum_{n=1}^N l_n,  &
                    \text{if reduction} = \text{`sum'.}
                \end{cases}

        .. note::
            The performance of this criterion is generally better when `target` contains class
            indices, as this allows for optimized computation. Consider providing `target` as
            class probabilities only when a single class label per minibatch item is too restrictive.


        :param class_weight: A manual rescaling weight given to each class. If given, has to be a
            Tensor of size `C`.

        :param ignore_index: Specifies a target value that is ignored and does not contribute to the
            input gradient. Note that :attr:`ignore_index` is only applicable when the target contains
            class indices.

        :param reduction: Specifies the reduction to apply to the output.

        :param label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing
            the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground
            truth and a uniform distribution as described in `Rethinking the Inception Architecture for
            Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

        :example:
            >>> # Example of target with class indices
            >>> loss = CrossEntropyLoss()
            >>> input = torch.randn(3, 5, requires_grad=True)
            >>> target = torch.empty(3, dtype=torch.long).random_(5)
            >>> output = loss(input, target)
            >>> output.backward()
            >>>
            >>> # Example of target with class probabilities
            >>> input = torch.randn(3, 5, requires_grad=True)
            >>> target = torch.randn(3, 5).softmax(dim=1)
            >>> output = loss(input, target)
            >>> output.backward()
        """
        super().__init__()
        if isinstance(reduction, str):
            reduction = str_to_enum(str_=reduction, enum=ReductionType)
        self.register_buffer("weight", class_weight)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self._reduction = reduction

    @property
    def reduction(self) -> ReductionType:
        return self._reduction

    @reduction.setter
    def reduction(self, value: ReductionType | str) -> None:
        if isinstance(value, str):
            value = str_to_enum(str_=value, enum=ReductionType)
        self._reduction = value

    def forward(
        self,
        input: Tensor,
        *,
        target: Tensor,
        instance_weight: Tensor | None = None,
        reduction: ReductionType | str | None = None,
    ) -> Tensor:
        """
        Computes the cross entropy loss between ``input`` and ``target``.

        :param input: Predicted unnormalized scores (often referred to as logits).
        :param target: Ground truth class indices or class probabilities.

        :param instance_weight: a manual rescaling weight given to each sample. If given, has to be a
            Tensor of 'N'.

        :param reduction: Overrides :attr:`reduction`.

        :returns: The (reduced) cross-entropy between ``input`` and ``target``.
        """

        reduction = self.reduction if reduction is None else reduction
        return cross_entropy_loss(
            input=input,
            target=target,
            instance_weight=instance_weight,
            reduction=reduction,
            class_weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
