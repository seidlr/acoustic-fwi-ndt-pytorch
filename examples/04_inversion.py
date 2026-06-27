"""Example 4: iterative full-waveform inversion.

Starting from the homogeneous model and the observed traces, iterate the gradient
(adjoint by default) to reconstruct the perturbation. Saves the misfit-convergence
curve and the reconstructed-model / model-update images.

Run: uv run python examples/04_inversion.py [--grid small|full]
     [--grad adjoint|autodiff] [--iters N] [--lr LR] [--device cpu|mps|cuda]
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
    ap.add_argument("--grid", choices=["small", "full"], default="small")
    ap.add_argument("--grad", choices=["adjoint", "autodiff"], default="adjoint")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(device)
    prob = build_problem(args.grid, device=device, dtype=dtype)

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
        grad_fn=args.grad,
        n_iter=args.iters,
        lr=args.lr,
    )

    drop = history[0] / history[-1]
    print(f"grid={args.grid} grad={args.grad} device={device} dtype={dtype}")
    print(f"misfit: {history[0]:.3e} -> {history[-1]:.3e}  ({drop:.1f}x reduction)")

    suffix = f"{args.grid}_{args.grad}"
    plotting.save_convergence(
        history,
        OUT / f"inversion_convergence_{suffix}.png",
        title=f"inversion convergence ({suffix})",
    )
    plotting.save_field(
        alpha2_hat,
        OUT / f"inversion_reconstruction_{suffix}.png",
        title="reconstructed alpha2",
        cmap="viridis",
        symmetric=False,
    )
    plotting.save_field(
        alpha2_hat - prob.start_alpha2,
        OUT / f"inversion_update_{suffix}.png",
        title="recovered model update (alpha2_hat - start)",
    )
    plotting.save_field(
        prob.true_alpha2 - prob.start_alpha2,
        OUT / f"inversion_true_anomaly_{suffix}.png",
        title="true anomaly (true - start)",
    )
    print(
        f"saved convergence + reconstruction figures to {OUT}/ (prefix inversion_*_{suffix})"
    )


if __name__ == "__main__":
    main()
