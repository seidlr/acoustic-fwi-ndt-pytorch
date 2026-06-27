"""Example 1: forward modeling on the 2D plate.

Regenerates the synthetic (homogeneous start model) and observed (perturbed, true
model) receiver traces from the domain files - replacing the large MATLAB .mat
seismograms - and saves wavefield snapshots.

Run: uv run python examples/01_forward_modeling.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fwi import geometry, plotting, wavelet
from fwi.config import SimConfig, resolve_device, resolve_dtype
from fwi.domain import build_alpha2, read_domain
from fwi.forward import forward

DATA = Path(__file__).resolve().parents[1] / "data" / "domain"
OUT = Path("outputs")


def run_model(domain_file: str, cfg, device, dtype, *, capture_field: bool):
    dom = read_domain(DATA / domain_file)
    alpha2, _ = build_alpha2(dom, device=device, dtype=dtype)
    src_i, src_j = geometry.make_sources(dom, cfg.x_src, cfg.y_src)
    rec_i, rec_j = geometry.make_receivers(dom, cfg)
    sig = wavelet.gaussian_derivative(cfg, device=device, dtype=dtype)
    res = forward(
        alpha2,
        sig,
        src_i,
        src_j,
        rec_i,
        rec_j,
        cfg,
        capture_wavefield=capture_field,
    )
    return dom, res


def main() -> None:
    OUT.mkdir(exist_ok=True)
    device = resolve_device()
    dtype = resolve_dtype(device)
    cfg = SimConfig()
    print(f"device={device} dtype={dtype} nt={cfg.nt} dt={cfg.dt:.3e}s")

    # Synthetic = homogeneous start model (capture wavefield for snapshots).
    _, syn = run_model("Domain2D_model.txt", cfg, device, dtype, capture_field=True)
    # Observed = true model with a gold inclusion.
    _, obs = run_model(
        "Domain2D_Fichtner_1pt_oben.txt", cfg, device, dtype, capture_field=False
    )

    np.savez(
        OUT / "seismograms.npz",
        synthetic=syn.traces.detach().cpu().numpy(),
        observed=obs.traces.detach().cpu().numpy(),
        dt=cfg.dt,
    )

    # snapshot at a step where the wavefront has developed
    snap = syn.wavefield[cfg.nt // 3]
    plotting.save_field(
        snap,
        OUT / "forward_wavefield_snapshot.png",
        title=f"synthetic wavefield, step {cfg.nt // 3}",
    )
    plotting.save_traces(
        syn.traces,
        OUT / "forward_synthetic_traces.png",
        title="synthetic receiver traces",
    )
    plotting.save_traces(
        obs.traces,
        OUT / "forward_observed_traces.png",
        title="observed receiver traces",
    )

    resid = (syn.traces - obs.traces).abs().max().item()
    print(f"saved traces + snapshots to {OUT}/")
    print(
        f"max |synthetic - observed| at receivers: {resid:.3e} (non-zero => defect visible)"
    )


if __name__ == "__main__":
    main()
