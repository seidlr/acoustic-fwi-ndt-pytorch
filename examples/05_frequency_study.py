"""Example 5: frequency study (the thesis 'inverted wave speed for different f0' figure).

Multiscale FWI by FREQUENCY CONTINUATION: invert the logo at 50 kHz from the
homogeneous start, then refine at 100 kHz and 200 kHz, each stage seeded by the
previous. This avoids the cycle-skipping that traps high-frequency FWI from a poor
start, and reproduces the thesis result: the recovered model sharpens (correlates
better with the truth) as the frequency rises. Saves the 3-panel figure and asserts
the correlation with the true model increases monotonically with f0.

Run: uv run python examples/05_frequency_study.py [--iters N] [--device cpu|mps|cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fwi import plotting
from fwi.config import SimConfig, resolve_dtype
from fwi.inversion import invert
from fwi.problems import build_problem

OUT = Path("outputs")
FREQS = [50_000.0, 100_000.0, 200_000.0]


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double() - a.flatten().double().mean()
    b = b.flatten().double() - b.flatten().double().mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(device)

    ref = build_problem("logo", device=device, dtype=dtype, cfg=SimConfig(f0=FREQS[-1]))
    true_update = ref.true_alpha2 - ref.start_alpha2

    alpha2 = ref.start_alpha2.clone()  # carried across frequencies (continuation)
    updates, corrs = [], []
    for f0 in FREQS:
        cfg = SimConfig(f0=f0)
        prob = build_problem("logo", device=device, dtype=dtype, cfg=cfg)
        print(
            f"--- f0 = {f0 / 1e3:.0f} kHz (wavelength {cfg.c_max / f0 * 1e3:.0f} mm) ---"
        )
        alpha2, history = invert(
            alpha2,
            prob.observed,
            prob.src_sig,
            prob.src_i,
            prob.src_j,
            prob.rec_i,
            prob.rec_j,
            cfg,
            active_mask=prob.active_mask,
            optimizer="lbfgs",
            n_iter=args.iters,
            verbose=True,
        )
        upd = (alpha2 - ref.start_alpha2).detach()
        updates.append(upd)
        c = _corr(upd, true_update)
        corrs.append(c)
        print(f"    J/J0 {history[0]:.2e} -> {history[-1]:.2e} | corr(true) = {c:.3f}")

    plotting.save_frequency_panel(updates, FREQS, OUT / "frequency_study.png")
    print(f"correlations (50/100/200 kHz) = {[round(c, 3) for c in corrs]}")
    assert corrs[0] < corrs[1] < corrs[2], (
        f"expected monotonically sharper recovery with f0, got corrs {corrs}"
    )
    print(
        f"saved 3-panel frequency study to {OUT}/frequency_study.png "
        f"(recovery sharpens with f0 via continuation)"
    )


if __name__ == "__main__":
    main()
