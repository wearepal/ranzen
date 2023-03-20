from __future__ import annotations
from typing import Any
from typing_extensions import override

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .common import LossClosure

__all__ = ["SAM"]


class SAM(Optimizer):
    """
    Implements the 'Sharpness Aware Minimization' (SAM) algorithm introducued
    in `Sharpness Aware Minimization`_ along with the adaptive variant of it
    introduced in `ASAM`_.

    SAM seeks parameters that lie in neighborhoods having uniformly low loss (rather than
    parameters that only themselves have low loss value). The adaptive variant of the
    algorithm addresses the original algorithm's sensitivity to parameter re-scaling
    that can lead to weakening of the connection between sharpness and generalization gap.

    .. _Sharpness Aware Minimization:
        https://arxiv.org/abs/2010.01412
    .. _ASAM:
        https://arxiv.org/abs/2102.11600
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        *,
        rho: float = 0.05,
        adaptive: bool = True,
    ) -> None:
        """
        :param base_optimizer: Base optimizer for SAM.
        :param rho: Neighborhood size.
        :param adaptive: Whether to use the adaptive variant of the algorithm.

        :raises ValueError: if ``rho`` is negative.

        :example:
            .. code-block:: python

                # Use AdamW as the base optimizer.
                base_optimizer = AdamW(model.parameters())
                # Wrap the base optimizer in SAM.
                optimizer = SAM(base_optimizer)

                # Closure required for recomputing the loss after computing epsilon(w).
                def _closure():
                  return loss_function(logits=model(input), targets=targets)

                loss = _closure()
                loss.backward()

                optimizer.step(closure=_closure)
                optimizer.zero_grad()
        """
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}. (Should be non-negative)")

        defaults = dict(rho=rho, adaptive=adaptive)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        super(SAM, self).__init__(params=base_optimizer.param_groups, defaults=defaults)

    @torch.no_grad()
    def _first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (p.pow(2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    @override
    def step(self, closure: LossClosure) -> Tensor:  # type: ignore
        r"""Performs a single optimization step.

        :param closure: A closure that reevaluates the model and returns the loss.
        :returns: loss returned by the closure if ``closure`` is not ``None`` else ``None``.
        """
        self._first_step(zero_grad=True)
        with torch.enable_grad():
            loss = closure()
            loss.backward()
        self._second_step()
        return loss

    def _grad_norm(self) -> Tensor:
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p="fro",
        )
        return norm

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads the optimizer state.

        :param state_dict: optimizer state. Should be an object returned
            from a call to :meth:`state_dict`.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
