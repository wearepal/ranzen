from __future__ import annotations
import math
from typing import Any, Callable, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ranzen.decorators import implements

__all__ = ["RAdam"]


class RAdam(Optimizer):
    """Implements the Rectified Adam (RAdam) algorithm."""

    def __init__(
        self,
        params: Iterable[Tensor | dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        """
        :param params: iterable of parameters to optimize or dicts defining parameter groups.
        :param lr: learning rate.
        :param betas: coefficients used for computing running averages of gradient and its square.
        :param eps: term added to the denominator to improve numerical stability.
        :param weight_decay: weight decay coefficient.

        :raises ValueError: if any one of ``lr``, ``betas``, ``eps``, or ``weight_decay`` is not in
            its permitted range.
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        buffer = [[None, None, None]] * 10
        if isinstance(params, (list, tuple)) and (len(params) > 0):
            for param in params:
                if isinstance(param, dict):
                    if "betas" in param and (
                        (param["betas"][0] != betas[0]) or (param["betas"][1] != betas[1])
                    ):
                        param["buffer"] = buffer

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=buffer)
        super().__init__(params=params, defaults=defaults)  # type: ignore

    @implements(Optimizer)
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        r"""Performs a single optimization step.

        :param closure: A closure that reevaluates the model and returns the loss.
        :returns: loss returned by the closure if ``closure`` is not ``None`` else ``None``.

        :raises RuntimeError: if gradients are sparse.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state["step"] += 1
                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = n_sma

                    # more conservative since it's an approximated value
                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if n_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
