"""Gaussian-derivative source wavelet (CreateSyntheticSeismogram.m:86-97).

Per-source frequency/delay/scaling are supported (arrays) so several sources with
different wavelengths can be combined in one forward run. Returns shape (n_src, nt).

The time axis is t = (1..nt) * dt (starts at dt, matching MATLAB). The singularity
factor 1/(dX*dY) uses the RAW header spacing (cfg.dx, cfg.dy), not the metric dx_m.
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
    Scalars broadcast across all sources; defaults come from cfg.
    """
    # infer number of sources
    n_src = 1
    for v in (f0, t0, scaling):
        if v is not None and not isinstance(v, (int, float)):
            n_src = max(n_src, len(list(v)))

    f0_l = _as_1d(f0, n_src, "f0") or [cfg.f0] * n_src
    t0_l = _as_1d(t0, n_src, "t0")
    if t0_l is None:
        # default delay is 1/f0 per source
        t0_l = [1.0 / f for f in f0_l]
    scaling_l = _as_1d(scaling, n_src, "scaling") or [cfg.scaling] * n_src

    t = torch.arange(1, cfg.nt + 1, device=device, dtype=dtype) * cfg.dt  # (nt,)
    sing = cfg.source_scale / (cfg.dx * cfg.dy)

    rows = []
    for f, t0_i, sc in zip(f0_l, t0_l, scaling_l):
        sigma = sc / (2.0 * math.pi * f)
        a = t - t0_i
        row = (
            -(sigma**2)
            * a
            / (math.sqrt(2.0 * math.pi) * sigma)
            * torch.exp(-(a**2) / (2.0 * sigma**2))
            * sing
        )
        rows.append(row)
    return torch.stack(rows, dim=0)
