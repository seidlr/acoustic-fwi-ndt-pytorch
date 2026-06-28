"""Example 1: forward modeling on the aluminum plate.

Forward-models the homogeneous (start) and cracked (true) 200x100 mm aluminum plates
and saves the wavefield snapshot + receiver traces. Regenerates the data the inversion
examples consume.

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


def run(domain_file: str, cfg, device, dtype, *, capture_field: bool):
    dom = read_domain(DATA / domain_file)
    alpha2, _ = build_alpha2(dom, device=device, dtype=dtype)
    src_i, src_j = geometry.make_sources(dom, cfg.x_src, cfg.y_src)
    rec_i, rec_j = geometry.make_receivers(dom, cfg)
    sig = wavelet.gaussian_derivative(cfg, device=device, dtype=dtype)
    return forward(
        alpha2, sig, src_i, src_j, rec_i, rec_j, cfg, capture_wavefield=capture_field
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    device = resolve_device()
    dtype = resolve_dtype(device)
    cfg = SimConfig()
    print(
        f"device={device} dtype={dtype} | aluminum plate 200x100mm @1mm, "
        f"f0={cfg.f0 / 1e3:.0f}kHz, dt={cfg.dt:.2e}s, nt={cfg.nt}"
    )

    syn = run("homogeneous.txt", cfg, device, dtype, capture_field=True)
    obs = run("cracked.txt", cfg, device, dtype, capture_field=False)

    np.savez(
        OUT / "seismograms.npz",
        synthetic=syn.traces.detach().cpu().numpy(),
        observed=obs.traces.detach().cpu().numpy(),
        dt=cfg.dt,
    )
    plotting.save_field(
        syn.wavefield[cfg.nt // 3],
        OUT / "forward_wavefield.png",
        title=f"aluminum wavefield, step {cfg.nt // 3}",
    )
    plotting.save_traces(
        syn.traces,
        OUT / "forward_synthetic_traces.png",
        title="synthetic (homogeneous)",
    )
    plotting.save_traces(
        obs.traces, OUT / "forward_observed_traces.png", title="observed (cracked)"
    )
    resid = (syn.traces - obs.traces).abs().max().item()
    print(f"saved snapshots + traces to {OUT}/")
    print(f"max |synthetic - observed| = {resid:.3e} (crack signal at sensors)")


if __name__ == "__main__":
    main()
