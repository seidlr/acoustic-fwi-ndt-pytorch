"""Example 4: single-crack FWI inversion with L-BFGS.

Reconstructs the crack in the aluminum plate from the observed traces, starting from
the homogeneous model, with the thesis optimizer (L-BFGS, J0-normalized misfit). Prints
classic training-loop progress and saves the convergence curve + reconstruction.

Run: uv run python examples/04_crack_inversion_lbfgs.py [--iters N] [--device cpu|mps|cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fwi import plotting
from fwi.config import resolve_dtype
from fwi.inversion import invert
from fwi.problems import build_problem

OUT = Path("outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(device)
    prob = build_problem("crack", device=device, dtype=dtype)

    alpha2_hat, history = invert(
        prob.start_alpha2,
        prob.observed,
        prob.src_sig,
        prob.src_i,
        prob.src_j,
        prob.rec_i,
        prob.rec_j,
        prob.cfg,
        active_mask=prob.active_mask,
        optimizer="lbfgs",
        n_iter=args.iters,
        verbose=True,
    )
    print(
        f"misfit J/J0: {history[0]:.3e} -> {history[-1]:.3e} "
        f"({history[0] / history[-1]:.0f}x reduction)"
    )

    plotting.save_convergence(
        history,
        OUT / "crack_convergence.png",
        title="single-crack L-BFGS convergence (J/J0)",
    )
    plotting.save_field(
        alpha2_hat - prob.start_alpha2,
        OUT / "crack_reconstruction.png",
        title="recovered alpha2 update (crack)",
    )
    plotting.save_field(
        prob.true_alpha2 - prob.start_alpha2,
        OUT / "crack_true.png",
        title="true crack anomaly",
    )
    print(f"saved convergence + reconstruction to {OUT}/")


if __name__ == "__main__":
    main()
