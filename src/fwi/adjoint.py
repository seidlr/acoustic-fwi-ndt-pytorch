"""Hand-coded adjoint-state gradient dJ/d(alpha2) (no autograd).

Mirrors AdjointMethod.m:196-244:
  1. Re-run the forward solver capturing the BARE Laplacian field nabla2u(t)
     (= d2phi/dx2 + d2phi/dy2, NOT * alpha2).
  2. Build the time-reversed residual as adjoint sources at the receivers.
  3. Adjoint solve (the same leapfrog) -> adjoint field lambda(t).
  4. kernel = -dt * sum_t lambda(t) * nabla2u(T-1-t), skipping the last
     `cutoff` steps. cutoff=0 (default) gives the exact dJ/d(alpha2) that
     equals the autodiff gradient.

Because the adjoint reuses the SAME leapfrog forward() (source injected with the
dt^2 weight, kernel summed with dt), the MATLAB-form kernel turns out to be the
EXACT discrete adjoint of dJ/d(alpha2) up to sign: measured ratio autodiff/kernel =
-1.0 to machine precision (std/mean ~1e-15). So the only reconciliation is a sign
flip (ADJOINT_MATCH_SCALE = -1), and the agreement is exact, not merely ~1e-5.
"""

from __future__ import annotations

import torch

from fwi.config import SimConfig
from fwi.forward import forward
from fwi.misfit import adjoint_source, l2_misfit

# Empirically -1 to machine precision (see tests/test_gradients.py): the kernel is
# the exact discrete adjoint of the forward solver, differing from dJ/d(alpha2) only
# in sign (the MATLAB kernel uses -sum; the gradient is +sum).
ADJOINT_MATCH_SCALE = -1.0


def adjoint_gradient(
    alpha2: torch.Tensor,
    observed: torch.Tensor,
    src_sig: torch.Tensor,
    src_i: torch.Tensor,
    src_j: torch.Tensor,
    rec_i: torch.Tensor,
    rec_j: torch.Tensor,
    cfg: SimConfig,
    *,
    active_mask: torch.Tensor | None = None,
    cutoff: int | None = None,
    raw: bool = False,
) -> tuple[torch.Tensor, float]:
    """Adjoint-state dJ/d(alpha2). Returns (kernel, misfit).

    cutoff defaults to cfg.cutoff_timesteps (0 = exact gradient). raw=True returns
    the unscaled MATLAB-form kernel (for deriving/inspecting ADJOINT_MATCH_SCALE).
    """
    cut = cfg.cutoff_timesteps if cutoff is None else cutoff
    with torch.no_grad():
        fwd = forward(
            alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg, capture_nabla2u=True
        )
        synthetic = fwd.traces
        nabla2u = fwd.nabla2u  # (nt, nI, nJ)
        assert nabla2u is not None

        adj_sig = adjoint_source(synthetic, observed)  # (n_rec, nt), time-reversed
        # adjoint sources sit at the receivers; we only need the adjoint wavefield
        dummy_i = rec_i[:1]
        dummy_j = rec_j[:1]
        adj = forward(
            alpha2, adj_sig, rec_i, rec_j, dummy_i, dummy_j, cfg, capture_wavefield=True
        )
        lam = adj.wavefield  # (nt, nI, nJ)
        assert lam is not None

        T = cfg.nt - cut
        nabla_rev = torch.flip(nabla2u, dims=(0,))  # nabla_rev[t] = nabla2u[nt-1-t]
        kernel = -cfg.dt * (lam[:T] * nabla_rev[:T]).sum(dim=0)

        if not raw:
            kernel = kernel * ADJOINT_MATCH_SCALE
        if active_mask is not None:
            kernel = kernel * active_mask.to(kernel.dtype)

        misfit = l2_misfit(synthetic, observed, cfg.dt)
    return kernel, float(misfit)
