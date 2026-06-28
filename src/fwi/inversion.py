"""Iterative full-waveform inversion: reconstruct alpha2 from observed traces.

Minimizes the L2 waveform misfit, normalized by the initial misfit J0 (so the
reported objective J/J0 starts at 1.0 - matching the thesis LeastSquaresCostFunctional
NORMALIZATION_CONSTANT). Only active (non-ghost) cells are updated; alpha2 >= 0.

Optimizers:
- "lbfgs" (default, the thesis optimizer): inverts a dimensionless model m = alpha2 /
  alpha2_background (m ~ 1) with the J0-normalized misfit and torch.optim.LBFGS
  (strong-Wolfe line search). The J0 normalization cancels the alpha2_bg scale so
  dJ~/dm is O(1), and the standard lr=1.0 initial step works. m is clamped to a
  CFL-safe [0, m_max] inside the closure so the line search never hits a negative or
  unstable alpha2 (which would NaN). The true autograd gradient is used.
- "adam"/"sgd": step on alpha2 directly using the UNIT-MAX-ABS-normalized gradient
  direction (the gradient is otherwise tiny relative to alpha2 ~ 4e7); lr is a step in
  alpha2 units.

`verbose=True` prints a classic training-loop line per iteration.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from fwi.adjoint import adjoint_gradient
from fwi.config import SPEED_ALUMINUM, SimConfig
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
    grad_fn: str = "autodiff",
    optimizer: str = "lbfgs",
    n_iter: int = 20,
    lr: float | None = None,
    alpha2_background: float | None = None,
    verbose: bool = False,
    callback: Callable[[int, float, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Run the inversion. Returns (alpha2_hat, normalized-misfit history J/J0).

    history[k] is J/J0 at iteration k (~1.0 at the start) and history[-1] the
    returned model's J/J0.
    """
    alpha2_bg = alpha2_background or SPEED_ALUMINUM**2
    mask = active_mask

    def misfit_of(alpha2: torch.Tensor) -> torch.Tensor:
        return l2_misfit(
            forward(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg).traces,
            observed,
            cfg.dt,
        )

    with torch.no_grad():
        J0 = float(misfit_of(alpha2_init.detach()))
    # `x or 1.0` would leave a NaN J0 (NaN is truthy), so guard isfinite AND nonzero.
    if not math.isfinite(J0) or J0 == 0.0:
        J0 = 1.0

    def _report(it: int, jn: float, gnorm: float, alpha2: torch.Tensor) -> None:
        if verbose:
            print(
                f"iter {it + 1:03d}/{n_iter} | loss (J/J0) = {jn:.4e} | "
                f"grad_norm = {gnorm:.3e}"
            )
        if callback is not None:
            callback(it, jn, alpha2)

    history: list[float] = []

    if optimizer == "lbfgs":
        # torch.optim.LBFGS needs a closure that calls backward(), so this branch is
        # autograd-only. Fail loud rather than silently ignoring an adjoint request.
        if grad_fn != "autodiff":
            raise ValueError(
                "optimizer='lbfgs' uses autograd; grad_fn='adjoint' is only supported "
                "with optimizer='adam'/'sgd'."
            )
        # dimensionless model m ~ 1 (ghost cells -> 0); invert in m-space.
        m = (alpha2_init.detach() / alpha2_bg).clone().requires_grad_(True)
        if lr is None:
            # The J0 normalization makes dJ~/dm O(1) (it cancels the alpha2_bg scale),
            # so the standard lr=1.0 initial step is correct; strong-Wolfe refines it.
            lr = 1.0
        opt = torch.optim.LBFGS([m], lr=lr, max_iter=1, line_search_fn="strong_wolfe")
        state = {"j": 1.0, "gnorm": 0.0}
        # CFL-safe upper bound on m: dt is stable up to c = cfl*dx_m/dt, i.e.
        # m_max = (c_limit / sqrt(alpha2_bg))^2. Clamp INSIDE the closure (differentiable)
        # so the line search never evaluates a negative/CFL-violating alpha2 (-> NaN).
        c_limit = cfg.cfl * min(cfg.dx_m, cfg.dy_m) / cfg.dt
        m_max = 0.9 * (c_limit**2) / alpha2_bg

        def closure():
            opt.zero_grad()
            jn = misfit_of(m.clamp(0.0, m_max) * alpha2_bg) / J0
            jn.backward()
            if mask is not None and m.grad is not None:
                with torch.no_grad():
                    m.grad.mul_(mask.to(m.grad.dtype))
            state["j"] = float(jn.detach())
            state["gnorm"] = float(m.grad.norm()) if m.grad is not None else 0.0
            return jn

        for it in range(n_iter):
            opt.step(closure)
            with torch.no_grad():
                m.clamp_(0.0, m_max)
                if mask is not None:
                    m.mul_(mask.to(m.dtype))
            history.append(state["j"])
            _report(it, state["j"], state["gnorm"], (m * alpha2_bg).detach())
        alpha2_hat = (m * alpha2_bg).detach()

    elif optimizer in ("adam", "sgd"):
        alpha2 = alpha2_init.detach().clone().requires_grad_(True)
        if lr is None:
            lr = 5e4
        opt = (
            torch.optim.Adam([alpha2], lr=lr)
            if optimizer == "adam"
            else torch.optim.SGD([alpha2], lr=lr)
        )

        def unit_norm(g: torch.Tensor) -> torch.Tensor:
            if mask is not None:
                g = g * mask.to(g.dtype)
            return g / (g.abs().max() + 1e-300)

        for it in range(n_iter):
            if grad_fn == "autodiff":
                alpha2.grad = None
                J = misfit_of(alpha2)
                J.backward()
                grad = alpha2.grad
                assert grad is not None  # set by backward()
                jn = float(J.detach()) / J0
                gn = float(grad.norm())
                with torch.no_grad():
                    alpha2.grad = unit_norm(grad)
            else:
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
                        active_mask=mask,
                    )
                    jn = Jval / J0
                    gn = float(kern.norm())
                    alpha2.grad = unit_norm(kern)
            opt.step()  # type: ignore[call-arg]
            with torch.no_grad():
                alpha2.clamp_(min=0.0)
                if mask is not None:
                    alpha2.mul_(mask.to(alpha2.dtype))
            history.append(jn)
            _report(it, jn, gn, alpha2.detach())
        alpha2_hat = alpha2.detach()

    else:
        raise ValueError(f"unknown optimizer {optimizer!r} (use lbfgs/adam/sgd)")

    with torch.no_grad():
        history.append(float(misfit_of(alpha2_hat)) / J0)
    return alpha2_hat, history
