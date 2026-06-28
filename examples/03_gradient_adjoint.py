"""Example 3: the FWI gradient by the hand-coded adjoint-state method.

Re-runs the forward solve capturing the bare Laplacian, solves the adjoint problem
(time-reversed residual as sources at the receivers), and correlates the two into
the sensitivity kernel dJ/d(alpha2). Prints the agreement with the autodiff gradient
and (Task 6) saves the Taylor / hockey-stick verification plot.

Run: uv run python examples/03_gradient_adjoint.py [--grid small|crack] [--device cpu|mps|cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fwi import plotting
from fwi.config import resolve_dtype
from fwi.adjoint import adjoint_gradient
from fwi.misfit import autodiff_gradient, l2_misfit
from fwi.forward import forward
from fwi.gradient_test import taylor_test, clean_slope
from fwi.problems import build_problem

OUT = Path("outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["small", "crack"], default="small")
    ap.add_argument("--device", default="cpu", help="cpu (float64, clean) | mps | cuda")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(device)
    prob = build_problem(args.grid, device=device, dtype=dtype)

    kernel, misfit = adjoint_gradient(
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
    g_ad, _ = autodiff_gradient(
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

    rel = (kernel - g_ad).norm() / g_ad.norm()
    dm = torch.ones_like(kernel) * prob.active_mask.to(kernel.dtype)
    dderiv_rel = abs(float((kernel * dm).sum() - (g_ad * dm).sum())) / abs(
        float((g_ad * dm).sum())
    )
    print(f"grid={args.grid} device={device} dtype={dtype}")
    print(f"misfit J = {misfit:.6e}")
    print(
        f"adjoint vs autodiff:  relative-L2 = {float(rel):.3e}  "
        f"directional-rel = {dderiv_rel:.3e}"
    )

    plotting.save_field(
        kernel,
        OUT / f"adjoint_kernel_{args.grid}.png",
        title=f"adjoint-state dJ/d(alpha2) ({args.grid} grid)",
    )
    print(f"saved adjoint kernel to {OUT}/adjoint_kernel_{args.grid}.png")

    # --- Taylor / hockey-stick verification of the adjoint gradient ---
    def J_fn(m):
        with torch.no_grad():
            tr = forward(
                m,
                prob.src_sig,
                prob.src_i,
                prob.src_j,
                prob.rec_i,
                prob.rec_j,
                prob.cfg,
            ).traces
        return float(l2_misfit(tr, prob.observed, prob.cfg.dt))

    torch.manual_seed(0)
    dm = torch.randn_like(kernel) * prob.active_mask.to(kernel.dtype)
    hs = [10.0**k for k in range(6, -7, -1)]
    hs, r1, r2 = taylor_test(J_fn, prob.start_alpha2, dm, kernel, hs)
    slope2, n2 = clean_slope(hs, r2)
    print(
        f"hockey-stick: second-order slope = {slope2:.3f} over {n2} clean steps "
        f"(expected ~2.0)"
    )
    mps_note = (
        "MPS float32: round-off floor is reached at larger h, so the clean "
        "h^2 range is shorter than CPU float64."
        if dtype == torch.float32
        else ""
    )
    plotting.save_hockey_stick(
        hs,
        r1,
        r2,
        OUT / f"hockey_stick_{args.grid}.png",
        title=f"adjoint gradient Taylor test ({args.grid}, {dtype}) slope2={slope2:.2f}",
        note=mps_note,
    )
    print(f"saved hockey-stick plot to {OUT}/hockey_stick_{args.grid}.png")


if __name__ == "__main__":
    main()
