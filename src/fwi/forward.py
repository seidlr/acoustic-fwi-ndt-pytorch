"""Differentiable 2D acoustic forward solver (pure torch).

Solves   d2(phi)/dt2 = alpha2 * (d2phi/dx2 + d2phi/dy2)
with central-difference space (2nd/4th order) and Stoermer/Verlet time stepping,
matching CreateSyntheticSeismogram.m:107-205.

Axis convention: i -> Y (tensor dim 0), j -> X (tensor dim 1). The X second
derivative uses dx_m on dim 1; the Y second derivative uses dy_m on dim 0.

The stored Laplacian (`nabla2u`) is the BARE stencil output d2phi/dx2 + d2phi/dy2,
NOT multiplied by alpha2 (CreateSyntheticSeismogram.m:184) - this is what the
adjoint-state kernel correlates against, so it equals the exact dJ/d(alpha2).

Ghost cells (alpha2 = 0) never evolve: phi stays 0 there by induction, so the
zero-padded stencil naturally supplies the Dirichlet boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fwi.config import SimConfig

# Central second-derivative stencils: (offset -> coefficient, common denominator).
_STENCILS = {
    2: ([(-1, 1.0), (0, -2.0), (1, 1.0)], 1.0),
    4: ([(-2, -1.0), (-1, 16.0), (0, -30.0), (1, 16.0), (2, -1.0)], 12.0),
}


@dataclass
class ForwardResult:
    traces: torch.Tensor  # (n_rec, nt)
    nabla2u: torch.Tensor | None = None  # (nt, nI, nJ) bare Laplacian per step
    wavefield: torch.Tensor | None = None  # (nt, nI, nJ) phi per step


def _second_derivative(
    phi: torch.Tensor, h: float, dim: int, order: int
) -> torch.Tensor:
    """Zero-padded central second derivative of `phi` along `dim` with step `h`."""
    coeffs, denom = _STENCILS[order]
    p = order // 2
    # pad `p` zeros on both sides of `dim` (F.pad pads last dim first)
    pad = [0, 0, 0, 0]
    # index into pad list: dim 1 (X) -> first pair, dim 0 (Y) -> second pair
    if dim == 1:
        pad[0] = pad[1] = p
    else:
        pad[2] = pad[3] = p
    fp = F.pad(phi, pad)
    nI, nJ = phi.shape
    out = torch.zeros_like(phi)
    for off, c in coeffs:
        if dim == 1:  # only X (dim 1) is padded; dim 0 indexed in full
            out = out + c * fp[:, p + off : p + off + nJ]
        else:  # only Y (dim 0) is padded; dim 1 indexed in full
            out = out + c * fp[p + off : p + off + nI, :]
    return out / (denom * h * h)


def laplacian(phi: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Bare Laplacian d2phi/dx2 + d2phi/dy2 (dx_m on dim 1, dy_m on dim 0)."""
    d2x = _second_derivative(phi, cfg.dx_m, dim=1, order=cfg.order)
    d2y = _second_derivative(phi, cfg.dy_m, dim=0, order=cfg.order)
    return d2x + d2y


def forward(
    alpha2: torch.Tensor,
    src_sig: torch.Tensor,
    src_i: torch.Tensor,
    src_j: torch.Tensor,
    rec_i: torch.Tensor,
    rec_j: torch.Tensor,
    cfg: SimConfig,
    *,
    capture_nabla2u: bool = False,
    capture_wavefield: bool = False,
) -> ForwardResult:
    """Run the forward simulation.

    Args:
        alpha2: (nI, nJ) squared-speed model (may require grad for autodiff).
        src_sig: (n_src, nt) source signals.
        src_i, src_j: (n_src,) source grid indices.
        rec_i, rec_j: (n_rec,) receiver grid indices.
        capture_nabla2u: also return the bare Laplacian stack (for the adjoint kernel).
        capture_wavefield: also return phi at every step (for plotting).

    Returns:
        ForwardResult with traces (n_rec, nt) and optional stacks.
    """
    device, dtype = alpha2.device, alpha2.dtype
    dt2 = cfg.dt * cfg.dt
    src_i = src_i.to(device)
    src_j = src_j.to(device)
    rec_i = rec_i.to(device)
    rec_j = rec_j.to(device)
    src_sig = src_sig.to(device=device, dtype=dtype)

    phi = torch.zeros_like(alpha2)
    phi_old = torch.zeros_like(alpha2)

    trace_list: list[torch.Tensor] = []
    nabla_list: list[torch.Tensor] | None = [] if capture_nabla2u else None
    field_list: list[torch.Tensor] | None = [] if capture_wavefield else None

    for t in range(cfg.nt):
        lap = laplacian(phi, cfg)
        phi_new = 2.0 * phi - phi_old + alpha2 * lap * dt2
        # additive source injection (handles coincident sources via accumulate)
        contrib = torch.zeros_like(phi_new)
        contrib = contrib.index_put(
            (src_i, src_j), src_sig[:, t] * dt2, accumulate=True
        )
        phi_new = phi_new + contrib

        phi_old = phi
        phi = phi_new

        trace_list.append(phi[rec_i, rec_j])
        if nabla_list is not None:
            nabla_list.append(lap)
        if field_list is not None:
            field_list.append(phi)

    traces = torch.stack(trace_list, dim=1)  # (n_rec, nt)
    nabla2u = torch.stack(nabla_list, dim=0) if nabla_list is not None else None
    wavefield = torch.stack(field_list, dim=0) if field_list is not None else None
    return ForwardResult(traces=traces, nabla2u=nabla2u, wavefield=wavefield)


def _second_derivative_batched(
    phi: torch.Tensor, h: float, dim: int, order: int
) -> torch.Tensor:
    """Central 2nd derivative on the last two spatial dims of a (S, nI, nJ) tensor.

    `dim` matches the 2D `forward` convention: 1 -> X (last dim), 0 -> Y (2nd-last).
    """
    coeffs, denom = _STENCILS[order]
    p = order // 2
    if dim == 1:  # X: pad the last dim (F.pad pads last dim first)
        fp = F.pad(phi, [p, p, 0, 0])
        n = phi.shape[-1]
        out = sum(
            (c * fp[..., p + off : p + off + n] for off, c in coeffs), phi.new_zeros(())
        )
    else:  # Y: pad the 2nd-last dim
        fp = F.pad(phi, [0, 0, p, p])
        n = phi.shape[-2]
        out = sum(
            (c * fp[..., p + off : p + off + n, :] for off, c in coeffs),
            phi.new_zeros(()),
        )
    return out / (denom * h * h)


def forward_multishot(
    alpha2: torch.Tensor,
    src_sig: torch.Tensor,
    src_i: torch.Tensor,
    src_j: torch.Tensor,
    rec_i: torch.Tensor,
    rec_j: torch.Tensor,
    cfg: SimConfig,
) -> torch.Tensor:
    """Batched forward: S shots (one moving source) solved in ONE (S, nI, nJ) pass.

    All shots share the wavelet `src_sig` (nt,) and the receiver ring `rec_i/rec_j`;
    shot s injects at `(src_i[s], src_j[s])`. Returns traces `(S, R, nt)`, numerically
    equal to stacking S single-shot `forward` solves - but it runs one set of stencil
    kernels per timestep on S x the data, so the GPU's per-launch overhead is amortized
    across all shots instead of paid S times. Differentiable (for batched FWI).

    Args:
        alpha2: (nI, nJ) squared-speed model (shared across shots; may require grad).
        src_sig: (nt,) shared source wavelet.
        src_i, src_j: (S,) per-shot source grid indices.
        rec_i, rec_j: (R,) shared receiver grid indices.
    """
    device, dtype = alpha2.device, alpha2.dtype
    dt2 = cfg.dt * cfg.dt
    src_i = src_i.to(device)
    src_j = src_j.to(device)
    rec_i = rec_i.to(device)
    rec_j = rec_j.to(device)
    src_sig = src_sig.to(device=device, dtype=dtype).reshape(-1)  # (nt,)

    s_count = int(src_i.shape[0])
    n_i, n_j = alpha2.shape
    phi = alpha2.new_zeros((s_count, n_i, n_j))
    phi_old = alpha2.new_zeros((s_count, n_i, n_j))
    sidx = torch.arange(s_count, device=device)

    trace_list: list[torch.Tensor] = []
    for t in range(cfg.nt):
        d2x = _second_derivative_batched(phi, cfg.dx_m, 1, cfg.order)
        d2y = _second_derivative_batched(phi, cfg.dy_m, 0, cfg.order)
        # alpha2 (nI, nJ) broadcasts across the S batch dim
        phi_new = 2.0 * phi - phi_old + alpha2 * (d2x + d2y) * dt2
        # inject each shot's source at its own cell (distinct sidx -> no collision)
        phi_new = phi_new.index_put(
            (sidx, src_i, src_j), (src_sig[t] * dt2).repeat(s_count), accumulate=True
        )
        phi_old = phi
        phi = phi_new
        trace_list.append(phi[sidx[:, None], rec_i[None, :], rec_j[None, :]])  # (S, R)

    return torch.stack(trace_list, dim=2)  # (S, R, nt)
