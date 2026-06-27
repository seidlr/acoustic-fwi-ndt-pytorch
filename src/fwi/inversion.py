"""Iterative full-waveform inversion: reconstruct alpha2 from observed traces.

Minimizes the L2 waveform misfit by gradient descent, using either the autodiff
gradient or the hand-coded adjoint kernel. Only active (non-ghost) cells are updated;
alpha2 is kept non-negative.

Optimizer note (deviation from the plan's LBFGS default): the gradient magnitude is
extreme (~1e-57 in float64 at the native source scale). We normalize the gradient to
unit max-abs each iteration, so the absolute scale cancels and `lr` is a step in
alpha2 units (~1e7) - a step of ~1e5 changes the model by ~1e5 per iteration at the
strongest-gradient cell. Adam is the default (smooths the normalized direction) and
sgd is selectable; LBFGS is deliberately not offered (its line search needs the
gradient scaled with the loss, which the normalization breaks). (Raw Adam without
normalization also fails: its eps=1e-8 dwarfs a 1e-57 gradient and zeroes the update.)
"""

from __future__ import annotations

from typing import Callable

import torch

from fwi.adjoint import adjoint_gradient
from fwi.config import SimConfig
from fwi.forward import forward
from fwi.misfit import l2_misfit


def invert(
    alpha2_init: torch.Tensor,
    observed: torch.Tensor,
    src_sig: torch.Tensor,
    src_i: torch.Tensor,
    src_j: torch.Tensor,
    rec_i: torch.Tensor,
    rec_j: torch.Tensor,
    cfg: SimConfig,
    *,
    active_mask: torch.Tensor | None = None,
    grad_fn: str = "adjoint",
    optimizer: str = "adam",
    n_iter: int = 60,
    lr: float = 5e4,
    callback: Callable[[int, float, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Run the inversion. Returns (alpha2_hat, misfit_history of length n_iter + 1:
    the misfit before each step plus the misfit of the returned model)."""
    alpha2 = alpha2_init.detach().clone().requires_grad_(True)

    # adam (default) and sgd both work on the unit-normalized gradient direction.
    # LBFGS is intentionally NOT offered: its Wolfe-condition line search needs the
    # gradient scaled consistently with the loss, but we normalize the gradient to
    # unit max-abs (the loss is ~1e-51), so LBFGS's step overshoots and diverges.
    if optimizer == "adam":
        opt: torch.optim.Optimizer = torch.optim.Adam([alpha2], lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD([alpha2], lr=lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r} (use 'adam' or 'sgd')")

    def _normalize(g: torch.Tensor) -> torch.Tensor:
        """Unit max-abs so the optimizer step is independent of the tiny grad scale."""
        if active_mask is not None:
            g = g * active_mask.to(g.dtype)
        return g / (g.abs().max() + 1e-300)

    def compute_grad() -> float:
        """Populate a normalized alpha2.grad and return the current misfit."""
        if grad_fn == "autodiff":
            alpha2.grad = None
            res = forward(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg)
            J = l2_misfit(res.traces, observed, cfg.dt)
            J.backward()
            assert alpha2.grad is not None
            with torch.no_grad():
                alpha2.grad = _normalize(alpha2.grad)
            return float(J.detach())
        # adjoint
        with torch.no_grad():
            kern, Jval = adjoint_gradient(
                alpha2,
                observed,
                src_sig,
                src_i,
                src_j,
                rec_i,
                rec_j,
                cfg,
                active_mask=active_mask,
            )
            alpha2.grad = _normalize(kern)
        return Jval

    history: list[float] = []
    for it in range(n_iter):
        Jval = compute_grad()  # misfit of the model at the start of this iteration
        opt.step()  # type: ignore[call-arg]  # Adam/SGD.step() takes no closure
        with torch.no_grad():
            alpha2.clamp_(min=0.0)
            if active_mask is not None:
                alpha2.mul_(active_mask.to(alpha2.dtype))
        history.append(Jval)
        if callback is not None:
            callback(it, Jval, alpha2.detach())

    # record the misfit of the FINAL (returned) model, so history[-1] matches alpha2_hat
    with torch.no_grad():
        final = forward(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg)
        history.append(float(l2_misfit(final.traces, observed, cfg.dt)))

    return alpha2.detach(), history
