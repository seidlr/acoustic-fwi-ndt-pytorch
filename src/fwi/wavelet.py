"""Gaussian-derivative source wavelet (real InversionToolbox GaussianDerivativeSourceTerm.m).

    src(t) = -scalingFactor * (t - t0) / (sqrt(2*pi) * deviation^3)
             * exp(-(t - t0)^2 / (2 * deviation^2)) * singularityFactor

with deviation = 1/(2*pi*f0). The deviation^3 normalization (~2e18 at 200 kHz) plus
scalingFactor (~1e7) gives a well-scaled wavefield/misfit. singularityFactor =
1/(dX*dY) in DOMAIN units (dx=dy=1.0 -> 1), matching MATLAB's 1/prod(spacings)
(spacings are the header values, not the metric step). Per-source f0/t0/scaling
arrays are supported (combined sources). Returns shape (n_src, nt). Time axis
t = (1..nt)*dt.
"""

from __future__ import annotations

import math

import torch

from fwi.config import SimConfig


def _as_1d(value, n: int, name: str) -> list[float] | None:
    if value is None:
        return None  # caller substitutes the cfg default
    if isinstance(value, (int, float)):
        return [float(value)] * n
    seq = [float(v) for v in value]
    if len(seq) == 1:  # a 1-element list broadcasts like a scalar
        return seq * n
    if len(seq) != n:
        raise ValueError(f"{name}: length {len(seq)} != n_src {n}")
    return seq


def gaussian_derivative(
    cfg: SimConfig,
    *,
    f0=None,
    t0=None,
    scaling=None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build (n_src, nt) source signals.

    n_src is inferred from whichever of f0/t0/scaling is a sequence (default 1).
    Scalars broadcast across all sources; defaults come from cfg (scaling -> scaling_factor).
    """
    n_src = 1
    for v in (f0, t0, scaling):
        if v is not None and not isinstance(v, (int, float)):
            n_src = max(n_src, len(list(v)))

    f0_l = _as_1d(f0, n_src, "f0") or [cfg.f0] * n_src
    t0_l = _as_1d(t0, n_src, "t0")
    if t0_l is None:
        t0_l = [1.0 / f for f in f0_l]  # default delay 1/f0 per source
    scaling_l = _as_1d(scaling, n_src, "scaling") or [cfg.scaling_factor] * n_src

    t = torch.arange(1, cfg.nt + 1, device=device, dtype=dtype) * cfg.dt  # (nt,)
    sing = 1.0 / (cfg.dx * cfg.dy)  # domain units (=1), matches MATLAB 1/prod(spacings)

    rows = []
    for f, t0_i, sc in zip(f0_l, t0_l, scaling_l):
        deviation = 1.0 / (2.0 * math.pi * f)
        a = t - t0_i
        row = (
            -sc
            * a
            / (math.sqrt(2.0 * math.pi) * deviation**3)
            * torch.exp(-(a**2) / (2.0 * deviation**2))
            * sing
        )
        rows.append(row)
    return torch.stack(rows, dim=0)
