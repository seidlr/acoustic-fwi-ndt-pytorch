"""Example 6: multi-defect 'L i square' logo inversion (frequency continuation).

Reconstructs the recreated thesis logo (L/square = 5000 m/s, i = 4800 m/s -- all
low-velocity vs the 6420 m/s aluminum) by multiscale FWI: invert at 50 -> 100 -> 200
kHz, each stage seeded by the previous
(continuation avoids high-frequency cycle-skipping). Saves the final reconstruction
next to the true model and checks it is a recognizable multi-defect recovery.

Run: uv run python examples/06_logo_inversion.py [--iters N] [--device cpu|mps|cuda]
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
    alpha2 = ref.start_alpha2.clone()
    history_all: list[float] = []
    for f0 in FREQS:
        cfg = SimConfig(f0=f0)
        prob = build_problem("logo", device=device, dtype=dtype, cfg=cfg)
        print(f"--- f0 = {f0 / 1e3:.0f} kHz ---")
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
        history_all.extend(history)

    update = (alpha2 - ref.start_alpha2).detach()
    corr = _corr(update, true_update)
    print(f"final recovered logo: corr(true) = {corr:.3f}")

    plotting.save_convergence(
        history_all,
        OUT / "logo_convergence.png",
        title="logo inversion convergence (J/J0, 50->100->200 kHz)",
    )
    plotting.save_field(
        update, OUT / "logo_reconstruction.png", title="recovered logo (L i square)"
    )
    plotting.save_field(true_update, OUT / "logo_true.png", title="true logo")
    print(f"saved logo reconstruction to {OUT}/logo_reconstruction.png")
    # All three shapes are LOW-velocity vs the aluminum background, so a faithful
    # recovery is correlated with the truth AND recovers the defect cells as slow
    # (negative alpha2 update on average where the true defects sit).
    defect = true_update < 0
    recovered_in_defect = float(update[defect].mean())
    print(f"mean recovered update in defect cells = {recovered_in_defect:.3e} (expect < 0)")
    assert corr > 0.5, f"logo recovery too poor (corr={corr:.3f})"
    assert recovered_in_defect < 0, "defects not recovered as low-velocity zones"


if __name__ == "__main__":
    main()
