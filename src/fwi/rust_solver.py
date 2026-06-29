"""Rust-backed forward solve + adjoint gradient, wrapped as a torch.autograd.Function.

The native `fwi_rust` extension (built with `maturin develop --release -m rust/Cargo.toml`)
runs the time-stepping forward solve and the adjoint-state gradient in compiled code. The
`torch.autograd.Function` below makes it a drop-in differentiable op: `loss.backward()`
triggers the Rust adjoint, so the standard L-BFGS machinery (`_lbfgs_minimize`) drives the
inversion with an analytical gradient. CPU-only; falls back via `rust_available()`.

Build the extension (opt-in `speedtest` extra provides maturin):
    uv sync --extra speedtest
    uv run maturin develop --release -m rust/Cargo.toml
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fwi.config import SPEED_ALUMINUM, SimConfig
from fwi.inversion import _lbfgs_minimize
from fwi.misfit import l2_misfit

# The native extension is opt-in and has no type stubs, so it is typed as Any.
fwi_rust: Any
try:  # the rest of the package never imports it
    import fwi_rust as _fwi_rust  # type: ignore[import-not-found]

    fwi_rust = _fwi_rust
    _HAVE_RUST = True
except ImportError:  # pragma: no cover - depends on whether the crate was built
    fwi_rust = None
    _HAVE_RUST = False


def rust_available() -> bool:
    """True if the compiled `fwi_rust` extension is importable."""
    return _HAVE_RUST


def _f64(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float64)


def _i64(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.int64)


def _src2d(src_sig: torch.Tensor) -> np.ndarray:
    ss = _f64(src_sig)
    return ss[None, :] if ss.ndim == 1 else ss


class _RustForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg):  # type: ignore[override]
        assert fwi_rust is not None  # guarded by rust_forward()
        traces = fwi_rust.forward(
            _f64(alpha2), _src2d(src_sig), _i64(src_i), _i64(src_j),
            _i64(rec_i), _i64(rec_j), cfg.dx_m, cfg.dy_m, cfg.dt, cfg.nt, cfg.order,
        )
        ctx.save_for_backward(alpha2, src_sig, src_i, src_j, rec_i, rec_j)
        ctx.cfg = cfg
        return torch.from_numpy(traces).to(device=alpha2.device, dtype=alpha2.dtype)

    @staticmethod
    def backward(ctx, grad_traces):  # type: ignore[override]
        assert fwi_rust is not None  # guarded by rust_forward()
        alpha2, src_sig, src_i, src_j, rec_i, rec_j = ctx.saved_tensors
        cfg = ctx.cfg
        grad = fwi_rust.gradient(
            _f64(alpha2), _src2d(src_sig), _i64(src_i), _i64(src_j),
            _i64(rec_i), _i64(rec_j), _f64(grad_traces),
            cfg.dx_m, cfg.dy_m, cfg.dt, cfg.nt, cfg.order, cfg.cutoff_timesteps,
        )
        g = torch.from_numpy(grad).to(device=alpha2.device, dtype=alpha2.dtype)
        return g, None, None, None, None, None, None


def rust_forward(
    alpha2: torch.Tensor,
    src_sig: torch.Tensor,
    src_i: torch.Tensor,
    src_j: torch.Tensor,
    rec_i: torch.Tensor,
    rec_j: torch.Tensor,
    cfg: SimConfig,
) -> torch.Tensor:
    """Differentiable Rust forward solve; backward runs the Rust adjoint. Returns (R, nt)."""
    if not _HAVE_RUST:
        raise RuntimeError(
            "fwi_rust not built. Run: uv run maturin develop --release -m rust/Cargo.toml"
        )
    return _RustForward.apply(alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg)


def invert_rust(
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
    n_iter: int = 20,
    lr: float | None = None,
    alpha2_background: float | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """L-BFGS inversion using the Rust forward + analytical adjoint gradient.

    Numerically identical to `fwi.inversion.invert` (same J0-normalized objective, same
    L-BFGS), but the misfit's forward/backward run in compiled Rust instead of torch.
    """
    alpha2_bg = alpha2_background or SPEED_ALUMINUM**2

    def misfit_of(a2: torch.Tensor) -> torch.Tensor:
        traces = rust_forward(a2, src_sig, src_i, src_j, rec_i, rec_j, cfg)
        return l2_misfit(traces, observed, cfg.dt)

    return _lbfgs_minimize(
        misfit_of, alpha2_init, cfg=cfg, mask=active_mask, n_iter=n_iter, lr=lr,
        alpha2_bg=alpha2_bg, verbose=verbose, callback=None,
    )
