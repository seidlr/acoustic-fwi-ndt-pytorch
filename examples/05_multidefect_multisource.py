"""Example 5: multi-defect plate with multiple combined sources of different wavelengths.

The more complex PhD variant: the true model has several defects
(Domain2D_Fichtner_3pt_oben) and the forward run fires several sources at once, each
with its own frequency (wavelength) and delay - superimposed in ONE wavefield, exactly
as the commented multi-source block in setupForwardParameters.m. Different wavelengths
illuminate different feature scales. The adjoint structure is unchanged (one residual
-> one adjoint solve -> one kernel), so the same machinery recovers multiple defects.

Run: uv run python examples/05_multidefect_multisource.py [--iters N] [--device ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fwi import geometry, plotting, wavelet
from fwi.adjoint import adjoint_gradient
from fwi.config import SimConfig, default_source_scale, resolve_dtype
from fwi.domain import build_alpha2, read_domain
from fwi.forward import forward
from fwi.inversion import invert

DATA = Path(__file__).resolve().parents[1] / "data" / "domain"
OUT = Path("outputs")

# Four combined sources at different plate locations, each a different wavelength
# (frequency) and delay. f0 in Hz; higher f0 = shorter wavelength = finer detail.
SRC_X = [117.0, 152.0, 173.0, 95.0]
SRC_Y = [47.0, 31.0, 71.0, 60.0]
SRC_F0 = [12000.0, 15000.0, 18000.0, 21000.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(device)
    cfg = SimConfig(source_scale=default_source_scale(dtype))

    dom_start = read_domain(DATA / "Domain2D_model.txt")  # homogeneous start
    dom_true = read_domain(DATA / "Domain2D_Fichtner_3pt_oben.txt")  # 3 defects
    start_alpha2, active = build_alpha2(dom_start, device=device, dtype=dtype)
    true_alpha2, _ = build_alpha2(dom_true, device=device, dtype=dtype)

    # multi-source combined: per-source location + wavelength + staggered delay
    src_i, src_j = geometry.make_sources(dom_start, SRC_X, SRC_Y)
    t0s = [(k + 1) / f for k, f in enumerate(SRC_F0)]  # staggered delays
    src_sig = wavelet.gaussian_derivative(
        cfg, f0=SRC_F0, t0=t0s, device=device, dtype=dtype
    )
    rec_i, rec_j = geometry.make_receivers(dom_start, cfg)
    print(
        f"device={device} dtype={dtype} | {len(SRC_F0)} combined sources, "
        f"wavelengths c/f0 = {[round(cfg.c_max / f * 100, 1) for f in SRC_F0]} cm"
    )

    with torch.no_grad():
        observed = forward(true_alpha2, src_sig, src_i, src_j, rec_i, rec_j, cfg).traces

    kernel, J = adjoint_gradient(
        start_alpha2,
        observed,
        src_sig,
        src_i,
        src_j,
        rec_i,
        rec_j,
        cfg,
        active_mask=active,
    )
    plotting.save_field(
        kernel,
        OUT / "multidefect_kernel.png",
        title="adjoint kernel, 3 defects + 4 combined sources",
    )

    alpha2_hat, history = invert(
        start_alpha2,
        observed,
        src_sig,
        src_i,
        src_j,
        rec_i,
        rec_j,
        cfg,
        active_mask=active,
        grad_fn="adjoint",
        n_iter=args.iters,
        lr=args.lr,
    )
    print(
        f"misfit: {history[0]:.3e} -> {history[-1]:.3e}  "
        f"({history[0] / history[-1]:.1f}x reduction)"
    )

    plotting.save_convergence(
        history,
        OUT / "multidefect_convergence.png",
        title="multi-defect inversion convergence",
    )
    plotting.save_field(
        alpha2_hat - start_alpha2,
        OUT / "multidefect_update.png",
        title="recovered update (3 defects)",
    )
    plotting.save_field(
        true_alpha2 - start_alpha2,
        OUT / "multidefect_true.png",
        title="true anomaly (3 defects)",
    )

    # report how many of the 3 true defect clusters have a recovered peak nearby
    n_recovered = _count_localized(
        alpha2_hat - start_alpha2, true_alpha2 - start_alpha2
    )
    print(f"recovered defects localized: {n_recovered} / 3")
    print(f"saved figures to {OUT}/ (prefix multidefect_*)")


def _count_localized(
    update: torch.Tensor, true_diff: torch.Tensor, radius: int = 5
) -> int:
    """Count true-defect cells that have a sizeable update within `radius`."""
    upd = update.abs()
    thr = 0.15 * float(upd.max())
    ti, tj = torch.where(true_diff.abs() > 0)
    hits = 0
    seen: set[tuple[int, int]] = set()
    for i, j in zip(ti.tolist(), tj.tolist()):
        key = (i // (2 * radius), j // (2 * radius))  # cluster defects coarsely
        if key in seen:
            continue
        i0, i1 = max(0, i - radius), i + radius + 1
        j0, j1 = max(0, j - radius), j + radius + 1
        if float(upd[i0:i1, j0:j1].max()) > thr:
            hits += 1
            seen.add(key)
    return hits


if __name__ == "__main__":
    main()
