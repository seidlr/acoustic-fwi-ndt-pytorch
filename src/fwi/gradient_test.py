"""Taylor / hockey-stick gradient verification.

For a step h along direction dm:
    r1(h) = |J(m + h dm) - J(m)|                 -> O(h)   (slope 1)
    r2(h) = |J(m + h dm) - J(m) - h <grad, dm>|  -> O(h^2) (slope 2)

A correct gradient makes r2 decrease as h^2 until floating-point round-off
dominates, where it turns back up - the "hockey stick". The second-order slope
is fit over the clean (pre-round-off) range only.

Guard: dm must not be (near-)orthogonal to grad, else <grad, dm> ~ 0 and r1 would
also look second-order, hiding the stick.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


def taylor_test(
    J_fn: Callable[[torch.Tensor], float],
    m: torch.Tensor,
    dm: torch.Tensor,
    grad: torch.Tensor,
    hs,
) -> tuple[list[float], list[float], list[float]]:
    """Return (hs, r1, r2). `hs` should be descending (large h first)."""
    J0 = float(J_fn(m))
    gdm = float((grad * dm).sum())
    gnorm = float(grad.norm())
    dnorm = float(dm.norm())
    if abs(gdm) <= 1e-12 * gnorm * dnorm:
        raise ValueError("dm is (near-)orthogonal to grad; Taylor test is ill-posed")
    hs = [float(h) for h in hs]
    r1: list[float] = []
    r2: list[float] = []
    for h in hs:
        Jh = float(J_fn(m + h * dm))
        r1.append(abs(Jh - J0))
        r2.append(abs(Jh - J0 - h * gdm))
    return hs, r1, r2


def clean_slope(
    hs: list[float], r: list[float], band: float = 0.3
) -> tuple[float, int]:
    """Least-squares slope of log r vs log h over the clean power-law plateau.

    The remainder is a power law only in a middle window: at large h higher-order
    terms dominate (slope departs upward/downward), at small h round-off dominates
    (the hockey-stick turn-up). We locate the longest contiguous run of mutually
    consistent local slopes (spread <= `band`) - that run IS the clean plateau -
    and fit over the points it spans.
    """
    n = len(r)
    logs = [math.log(h) for h in hs]
    if any(logs[i + 1] == logs[i] for i in range(n - 1)):
        raise ValueError(
            "step sizes hs must be strictly distinct (no duplicate values)"
        )
    # floor at a tiny positive so exact-zero round-off values stay finite; they
    # become slope outliers and are excluded by the plateau detector below.
    logr = [math.log(max(v, 1e-300)) for v in r]
    local = [(logr[i + 1] - logr[i]) / (logs[i + 1] - logs[i]) for i in range(n - 1)]

    best_start, best_len = 0, 0
    i = 0
    while i < len(local):
        lo = hi = local[i]
        j = i
        while j < len(local) and max(hi, local[j]) - min(lo, local[j]) <= band:
            lo, hi = min(lo, local[j]), max(hi, local[j])
            j += 1
        if j - i > best_len:
            best_start, best_len = i, j - i
        i = max(j, i + 1)

    idx = list(range(best_start, best_start + best_len + 1))  # points spanning the run
    if len(idx) < 3:
        raise ValueError(f"fewer than 3 clean steps (got {len(idx)})")
    xs = [logs[k] for k in idx]
    ys = [logr[k] for k in idx]
    m = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (m * sxy - sx * sy) / (m * sxx - sx * sx)
    return slope, len(idx)
