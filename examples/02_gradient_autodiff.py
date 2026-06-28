"""Example 2: the misfit gradient dJ/d(alpha2) by automatic differentiation.

Make alpha2 a leaf tensor with requires_grad, run the forward solver + L2 misfit,
call backward(), and read alpha2.grad. This is "FWI gradient for free" - autograd
backprops through the whole time-stepping loop.

On the full aluminum plate (104x204x1000) the autograd graph is sizeable but tractable
on CPU; use --grid small for a quick check.

Run: uv run python examples/02_gradient_autodiff.py [--grid small|crack] [--device cpu|mps|cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fwi import plotting
from fwi.config import resolve_dtype
from fwi.misfit import autodiff_gradient
from fwi.problems import build_problem

OUT = Path("outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["small", "crack"], default="crack")
    ap.add_argument("--device", default="cpu", help="cpu (float64, clean) | mps | cuda")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(device)

    prob = build_problem(args.grid, device=device, dtype=dtype)
    grad, misfit = autodiff_gradient(
        prob.start_alpha2,
        prob.observed,
        prob.src_sig,
        prob.src_i,
        prob.src_j,
        prob.rec_i,
        prob.rec_j,
        prob.cfg,
        active_mask=prob.active_mask,
    )

    print(f"grid={args.grid} device={device} dtype={dtype}")
    print(f"misfit J = {misfit:.6e}")
    print(
        f"grad shape {tuple(grad.shape)} | finite={bool(torch.isfinite(grad).all())} "
        f"| max|grad|={float(grad.abs().max()):.3e}"
    )
    plotting.save_field(
        grad,
        OUT / f"autodiff_kernel_{args.grid}.png",
        title=f"autodiff dJ/d(alpha2) ({args.grid} grid)",
    )
    print(f"saved kernel image to {OUT}/autodiff_kernel_{args.grid}.png")


if __name__ == "__main__":
    main()
