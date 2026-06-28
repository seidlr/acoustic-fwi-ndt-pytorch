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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

import torch

from fwi.adjoint import adjoint_gradient
from fwi.config import SPEED_ALUMINUM, SimConfig
from fwi.forward import forward, forward_multishot
from fwi.misfit import l2_misfit

if TYPE_CHECKING:
    from fwi.problems import Shot

_Misfit = Callable[[torch.Tensor], torch.Tensor]


def _report(
    it: int,
    n_iter: int,
    jn: float,
    gnorm: float,
    alpha2: torch.Tensor,
    verbose: bool,
    callback: Callable[[int, float, torch.Tensor], None] | None,
) -> None:
    if verbose:
        print(
            f"iter {it + 1:03d}/{n_iter} | loss (J/J0) = {jn:.4e} | "
            f"grad_norm = {gnorm:.3e}"
        )
    if callback is not None:
        callback(it, jn, alpha2)


def _cfl_m_max(cfg: SimConfig, alpha2_bg: float) -> float:
    """CFL-safe upper bound on dimensionless m: dt is stable up to c = cfl*dx_m/dt, so
    m_max = 0.9*(c_limit/sqrt(alpha2_bg))^2. Clamping m to [0, m_max] inside the closure
    keeps every line-search probe at a stable, non-negative alpha2 (else NaN)."""
    c_limit = cfg.cfl * min(cfg.dx_m, cfg.dy_m) / cfg.dt
    return 0.9 * (c_limit**2) / alpha2_bg


def _lbfgs_minimize(
    misfit_of: _Misfit,
    alpha2_init: torch.Tensor,
    *,
    cfg: SimConfig,
    mask: torch.Tensor | None,
    n_iter: int,
    lr: float | None,
    alpha2_bg: float,
    verbose: bool,
    callback: Callable[[int, float, torch.Tensor], None] | None,
) -> tuple[torch.Tensor, list[float]]:
    """L-BFGS in dimensionless m-space (m = alpha2/alpha2_bg) on a J0-normalized
    `misfit_of`, one LBFGS iteration per outer step so the per-iteration training-loop
    progress is reportable. Used by single-shot `invert`."""
    with torch.no_grad():
        J0 = float(misfit_of(alpha2_init.detach()))
    # `x or 1.0` would leave a NaN J0 (NaN is truthy), so guard isfinite AND nonzero.
    if not math.isfinite(J0) or J0 == 0.0:
        J0 = 1.0

    m = (alpha2_init.detach() / alpha2_bg).clone().requires_grad_(True)
    if lr is None:
        # The J0 normalization makes dJ~/dm O(1) (it cancels the alpha2_bg scale),
        # so the standard lr=1.0 initial step is correct; strong-Wolfe refines it.
        lr = 1.0
    opt = torch.optim.LBFGS([m], lr=lr, max_iter=1, line_search_fn="strong_wolfe")
    state = {"j": 1.0, "gnorm": 0.0}
    m_max = _cfl_m_max(cfg, alpha2_bg)

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

    history: list[float] = []
    for it in range(n_iter):
        opt.step(closure)
        with torch.no_grad():
            m.clamp_(0.0, m_max)
            if mask is not None:
                m.mul_(mask.to(m.dtype))
        history.append(state["j"])
        _report(
            it,
            n_iter,
            state["j"],
            state["gnorm"],
            (m * alpha2_bg).detach(),
            verbose,
            callback,
        )
    alpha2_hat = (m * alpha2_bg).detach()
    with torch.no_grad():
        history.append(float(misfit_of(alpha2_hat)) / J0)
    return alpha2_hat, history


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

    if optimizer == "lbfgs":
        # torch.optim.LBFGS needs a closure that calls backward(), so this branch is
        # autograd-only. Fail loud rather than silently ignoring an adjoint request.
        if grad_fn != "autodiff":
            raise ValueError(
                "optimizer='lbfgs' uses autograd; grad_fn='adjoint' is only supported "
                "with optimizer='adam'/'sgd'."
            )
        return _lbfgs_minimize(
            misfit_of,
            alpha2_init,
            cfg=cfg,
            mask=mask,
            n_iter=n_iter,
            lr=lr,
            alpha2_bg=alpha2_bg,
            verbose=verbose,
            callback=callback,
        )

    if optimizer not in ("adam", "sgd"):
        raise ValueError(f"unknown optimizer {optimizer!r} (use lbfgs/adam/sgd)")

    with torch.no_grad():
        J0 = float(misfit_of(alpha2_init.detach()))
    if not math.isfinite(J0) or J0 == 0.0:
        J0 = 1.0

    history: list[float] = []
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
        _report(it, n_iter, jn, gn, alpha2.detach(), verbose, callback)
    alpha2_hat = alpha2.detach()

    with torch.no_grad():
        history.append(float(misfit_of(alpha2_hat)) / J0)
    return alpha2_hat, history


def invert_multishot(
    alpha2_init: torch.Tensor,
    shots: Sequence[Shot],
    src_sig: torch.Tensor,
    cfg: SimConfig,
    *,
    active_mask: torch.Tensor | None = None,
    n_iter: int = 15,
    lr: float | None = None,
    alpha2_background: float | None = None,
    verbose: bool = False,
    callback: Callable[[int, float, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Multi-shot (round-robin) FWI: the misfit/gradient sum over `shots`.

    Each shot is a source at a different sensor (a single source moving from receiver to
    receiver) recorded at that shot's own receivers (the other sensors). Summing the
    J0-normalized misfit over all shots illuminates the medium from many angles,
    conditioning the inversion far better than a single fixed source. Autograd sums the
    per-shot gradients. `shots` is a sequence of `fwi.problems.Shot` (each carrying
    `src_i`, `src_j`, `rec_i`, `rec_j`, `observed`).

    Unlike single-shot `invert`, this runs L-BFGS's own internal iteration loop
    (`max_iter=n_iter` in one `opt.step`): with many shots, the summed-gradient line
    search can fail on the first steepest-descent step, and the single-step-per-call loop
    would then build a degenerate history and stall - the internal loop is robust to it.
    `verbose` reports per function evaluation. Returns (alpha2_hat, J/J0 history).
    """
    alpha2_bg = alpha2_background or SPEED_ALUMINUM**2
    mask = active_mask

    def misfit_of(alpha2: torch.Tensor) -> torch.Tensor:
        total = alpha2.new_zeros(())
        for sh in shots:
            traces = forward(
                alpha2, src_sig, sh.src_i, sh.src_j, sh.rec_i, sh.rec_j, cfg
            ).traces
            total = total + l2_misfit(traces, sh.observed, cfg.dt)
        return total

    with torch.no_grad():
        J0 = float(misfit_of(alpha2_init.detach()))
    if not math.isfinite(J0) or J0 == 0.0:
        J0 = 1.0

    m = (alpha2_init.detach() / alpha2_bg).clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [m],
        lr=lr or 1.0,
        max_iter=n_iter,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    m_max = _cfl_m_max(cfg, alpha2_bg)
    history: list[float] = []
    evals = {"n": 0}

    def closure():
        opt.zero_grad()
        jn = misfit_of(m.clamp(0.0, m_max) * alpha2_bg) / J0
        jn.backward()
        if mask is not None and m.grad is not None:
            with torch.no_grad():
                m.grad.mul_(mask.to(m.grad.dtype))
        jv = float(jn.detach())
        gn = float(m.grad.norm()) if m.grad is not None else 0.0
        history.append(jv)
        if verbose:
            print(
                f"eval {evals['n'] + 1:03d} | loss (J/J0) = {jv:.4e} | "
                f"grad_norm = {gn:.3e}"
            )
        if callback is not None:
            callback(evals["n"], jv, (m * alpha2_bg).detach())
        evals["n"] += 1
        return jn

    opt.step(closure)
    with torch.no_grad():
        m.clamp_(0.0, m_max)
        if mask is not None:
            m.mul_(mask.to(m.dtype))
    alpha2_hat = (m * alpha2_bg).detach()
    with torch.no_grad():
        history.append(float(misfit_of(alpha2_hat)) / J0)
    return alpha2_hat, history


def invert_multishot_batched(
    alpha2_init: torch.Tensor,
    src_sig: torch.Tensor,
    src_i: torch.Tensor,
    src_j: torch.Tensor,
    rec_i: torch.Tensor,
    rec_j: torch.Tensor,
    observed: torch.Tensor,
    cfg: SimConfig,
    *,
    self_mask: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    n_iter: int = 15,
    lr: float | None = None,
    alpha2_background: float | None = None,
    verbose: bool = False,
    callback: Callable[[int, float, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Batched multi-shot FWI - numerically identical to `invert_multishot`, but the S
    shots run as ONE (S, nI, nJ) `forward_multishot` per evaluation instead of S
    sequential solves. On a GPU that turns S tiny kernel launches per timestep into one
    on S x the data, so the device is actually saturated; on CPU it is a smaller win.

    All shots share the wavelet `src_sig` (nt,) and receiver ring `rec_i/rec_j`; shot s
    fires at `(src_i[s], src_j[s])` and `observed` is `(S, R, nt)`. `self_mask` `(S, R)`
    zeroes each shot's own-sensor trace (pitch-catch). Returns (alpha2_hat, J/J0 history).
    """
    alpha2_bg = alpha2_background or SPEED_ALUMINUM**2
    mask = active_mask
    rmask = None if self_mask is None else self_mask[..., None]  # (S, R, 1)

    def misfit_of(alpha2: torch.Tensor) -> torch.Tensor:
        syn = forward_multishot(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg)
        resid = syn - observed
        if rmask is not None:
            resid = resid * rmask
        return 0.5 * torch.sum(resid * resid) * cfg.dt

    with torch.no_grad():
        J0 = float(misfit_of(alpha2_init.detach()))
    if not math.isfinite(J0) or J0 == 0.0:
        J0 = 1.0

    m = (alpha2_init.detach() / alpha2_bg).clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [m],
        lr=lr or 1.0,
        max_iter=n_iter,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    m_max = _cfl_m_max(cfg, alpha2_bg)
    history: list[float] = []
    evals = {"n": 0}

    def closure():
        opt.zero_grad()
        jn = misfit_of(m.clamp(0.0, m_max) * alpha2_bg) / J0
        jn.backward()
        if mask is not None and m.grad is not None:
            with torch.no_grad():
                m.grad.mul_(mask.to(m.grad.dtype))
        jv = float(jn.detach())
        gn = float(m.grad.norm()) if m.grad is not None else 0.0
        history.append(jv)
        if verbose:
            print(
                f"eval {evals['n'] + 1:03d} | loss (J/J0) = {jv:.4e} | "
                f"grad_norm = {gn:.3e}"
            )
        if callback is not None:
            callback(evals["n"], jv, (m * alpha2_bg).detach())
        evals["n"] += 1
        return jn

    opt.step(closure)
    with torch.no_grad():
        m.clamp_(0.0, m_max)
        if mask is not None:
            m.mul_(mask.to(m.dtype))
    alpha2_hat = (m * alpha2_bg).detach()
    with torch.no_grad():
        history.append(float(misfit_of(alpha2_hat)) / J0)
    return alpha2_hat, history
