"""Correctness of the speed-investigation solvers (notebook 05).

The conv2d and Rust forwards must be numerically equal to the reference `forward`, and
the Rust adjoint must equal the autodiff gradient on the active region (ghost cells are
masked out by the inversion). Rust tests skip gracefully when `fwi_rust` is not built.
"""

from __future__ import annotations

import torch

from fwi.forward import forward
from fwi.forward_conv2d import forward_conv2d
from fwi.misfit import autodiff_gradient, l2_misfit
from fwi.problems import build_problem
from fwi.rust_solver import rust_available, rust_forward

F64 = torch.float64
CPU = torch.device("cpu")


def _small():
    p = build_problem("small", device=CPU, dtype=F64)
    return p, (p.start_alpha2, p.src_sig, p.src_i, p.src_j, p.rec_i, p.rec_j, p.cfg)


class TestSpeedtestSolvers:
    def test_conv2d_forward_equals_reference(self):
        _, (a, ss, si, sj, ri, rj, cfg) = _small()
        ref = forward(a, ss, si, sj, ri, rj, cfg).traces
        got = forward_conv2d(a, ss, si, sj, ri, rj, cfg).traces
        assert float((got - ref).norm() / ref.norm()) < 1e-10

    def test_rust_forward_equals_reference(self):
        if not rust_available():
            import pytest

            pytest.skip("fwi_rust not built")
        _, (a, ss, si, sj, ri, rj, cfg) = _small()
        ref = forward(a, ss, si, sj, ri, rj, cfg).traces
        with torch.no_grad():
            got = rust_forward(a, ss, si, sj, ri, rj, cfg)
        assert float((got - ref).norm() / ref.norm()) < 1e-10

    def test_forward_variants_agree(self):
        from fwi.benchmark import forward_variants

        _, args = _small()
        variants = forward_variants()
        assert "naive" in variants and "conv2d" in variants
        ref = variants["naive"](*args)
        for name, fn in variants.items():
            rel = float((fn(*args) - ref).norm() / ref.norm())
            assert rel < 1e-9, f"variant {name} differs by {rel:.1e}"

    def test_rust_gradient_equals_autodiff(self):
        if not rust_available():
            import pytest

            pytest.skip("fwi_rust not built")
        p, (a, ss, si, sj, ri, rj, cfg) = _small()
        m = p.active_mask
        a_req = a.detach().clone().requires_grad_(True)
        J = l2_misfit(rust_forward(a_req, ss, si, sj, ri, rj, cfg), p.observed, cfg.dt)
        J.backward()
        assert a_req.grad is not None
        g_rust = a_req.grad * m  # ghost cells masked, as the inversion does
        g_ad, _ = autodiff_gradient(a, p.observed, ss, si, sj, ri, rj, cfg, active_mask=m)
        assert float((g_rust - g_ad).norm() / g_ad.norm()) < 1e-9
