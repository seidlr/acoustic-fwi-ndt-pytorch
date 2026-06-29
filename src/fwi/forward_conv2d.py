"""conv2d-based forward solve (autograd- and GPU-friendly).

`forward.py` builds the Laplacian with `F.pad` + a Python loop over the stencil
coefficients (~12 small ops per timestep). Here the whole Laplacian is a single fixed
`conv2d` kernel, so each timestep launches ONE stencil kernel instead of ~12. It stays
fully differentiable (conv2d has an autograd backward) and is numerically equal to
`fwi.forward.forward` (same zero-padded stencil, same leapfrog, same dt^2 source).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fwi.config import SimConfig
from fwi.forward import _STENCILS, ForwardResult


def laplacian_kernel(cfg: SimConfig, device, dtype) -> torch.Tensor:
    """Fixed (1, 1, K, K) cross-shaped Laplacian kernel = d2/dx2 + d2/dy2."""
    coeffs, denom = _STENCILS[cfg.order]
    k = cfg.order + 1
    c = cfg.order // 2
    ker = torch.zeros((k, k), device=device, dtype=dtype)
    dx2, dy2 = cfg.dx_m**2, cfg.dy_m**2
    for off, coef in coeffs:
        ker[c, c + off] += coef / (denom * dx2)  # x-stencil along columns (j)
        ker[c + off, c] += coef / (denom * dy2)  # y-stencil along rows (i)
    return ker.reshape(1, 1, k, k)


def forward_conv2d(
    alpha2: torch.Tensor,
    src_sig: torch.Tensor,
    src_i: torch.Tensor,
    src_j: torch.Tensor,
    rec_i: torch.Tensor,
    rec_j: torch.Tensor,
    cfg: SimConfig,
    *,
    capture_wavefield: bool = False,
) -> ForwardResult:
    """Forward simulation with a conv2d Laplacian. Returns traces (n_rec, nt)."""
    device, dtype = alpha2.device, alpha2.dtype
    dt2 = cfg.dt * cfg.dt
    p = cfg.order // 2
    ker = laplacian_kernel(cfg, device, dtype)
    src_i = src_i.to(device)
    src_j = src_j.to(device)
    rec_i = rec_i.to(device)
    rec_j = rec_j.to(device)
    src_sig = src_sig.to(device=device, dtype=dtype)
    if src_sig.ndim == 1:
        src_sig = src_sig[None, :]

    phi = torch.zeros_like(alpha2)
    phi_old = torch.zeros_like(alpha2)
    trace_list: list[torch.Tensor] = []
    field_list: list[torch.Tensor] | None = [] if capture_wavefield else None

    for t in range(cfg.nt):
        lap = F.conv2d(phi[None, None], ker, padding=p)[0, 0]
        phi_new = 2.0 * phi - phi_old + alpha2 * lap * dt2
        contrib = torch.zeros_like(phi_new).index_put(
            (src_i, src_j), src_sig[:, t] * dt2, accumulate=True
        )
        phi_new = phi_new + contrib
        phi_old = phi
        phi = phi_new
        trace_list.append(phi[rec_i, rec_j])
        if field_list is not None:
            field_list.append(phi)

    traces = torch.stack(trace_list, dim=1)
    wavefield = torch.stack(field_list, dim=0) if field_list is not None else None
    return ForwardResult(traces=traces, wavefield=wavefield)
