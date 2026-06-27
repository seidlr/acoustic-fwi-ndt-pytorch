"""Waveform misfit, adjoint source, and the autodiff gradient.

L2 misfit J = 0.5 * sum_t |u - u0|^2 * dt over receivers (AdjointMethod.m).
Adjoint source = time-reversed residual flip(u - u0) (AdjointMethod.m:15), no windowing.

`autodiff_gradient` differentiates the FULL misfit through the time-stepping solver
to get dJ/d(alpha2). The manual adjoint kernel (fwi.adjoint) must equal this.
"""

from __future__ import annotations

import torch

from fwi.config import SimConfig
from fwi.forward import forward


def l2_misfit(u: torch.Tensor, u0: torch.Tensor, dt: float) -> torch.Tensor:
    """0.5 * sum |u - u0|^2 * dt."""
    return 0.5 * torch.sum((u - u0) ** 2) * dt


def adjoint_source(u: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
    """Time-reversed residual (u - u0) flipped along the time axis."""
    return torch.flip(u - u0, dims=(-1,))


def autodiff_gradient(
    alpha2_start: torch.Tensor,
    observed: torch.Tensor,
    src_sig: torch.Tensor,
    src_i: torch.Tensor,
    src_j: torch.Tensor,
    rec_i: torch.Tensor,
    rec_j: torch.Tensor,
    cfg: SimConfig,
    *,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """dJ/d(alpha2) via autograd through the forward solver.

    Returns (grad, misfit). The gradient is ghost-masked when active_mask is given.
    """
    alpha2 = alpha2_start.detach().clone().requires_grad_(True)
    res = forward(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg)
    J = l2_misfit(res.traces, observed, cfg.dt)
    J.backward()
    grad = alpha2.grad
    assert grad is not None
    if active_mask is not None:
        grad = grad * active_mask.to(grad.dtype)
    return grad.detach(), float(J.detach())
