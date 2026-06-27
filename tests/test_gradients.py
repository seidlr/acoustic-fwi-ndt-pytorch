"""Gradient-correctness tests (Tasks 5 and 6) - the core contract.

All on the small grid, CPU float64. Task 5: the manual adjoint-state kernel equals
the autodiff gradient (relative L2 and directional derivative). Task 6 adds the
Taylor/hockey-stick second-order-slope assertion.
"""

from __future__ import annotations

import torch

from fwi.problems import build_problem
from fwi.misfit import autodiff_gradient, l2_misfit
from fwi.adjoint import adjoint_gradient
from fwi.forward import forward
from fwi.gradient_test import taylor_test, clean_slope

F64 = torch.float64
CPU = torch.device("cpu")


def _grads():
    prob = build_problem("small", device=CPU, dtype=F64)
    g_ad, J_ad = autodiff_gradient(
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
    g_aj, J_aj = adjoint_gradient(
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
    return prob, g_ad, g_aj, J_ad, J_aj


class TestGradientAgreement:
    def test_adjoint_equals_autodiff_relative_l2(self):
        _, g_ad, g_aj, J_ad, J_aj = _grads()
        assert abs(J_ad - J_aj) / abs(J_ad) < 1e-10  # same forward misfit
        rel = (g_aj - g_ad).norm() / g_ad.norm()
        assert float(rel) < 1e-5

    def test_adjoint_equals_autodiff_directional(self):
        prob, g_ad, g_aj, _, _ = _grads()
        torch.manual_seed(0)
        dm = torch.randn_like(g_ad) * prob.active_mask.to(g_ad.dtype)
        d_ad = torch.sum(g_ad * dm)
        d_aj = torch.sum(g_aj * dm)
        assert abs(float(d_aj - d_ad)) / abs(float(d_ad)) < 1e-5


class TestHockeyStick:
    def test_taylor_second_order_slope(self):
        prob = build_problem("small", device=CPU, dtype=F64)
        cfg = prob.cfg
        grad, _ = adjoint_gradient(
            prob.start_alpha2,
            prob.observed,
            prob.src_sig,
            prob.src_i,
            prob.src_j,
            prob.rec_i,
            prob.rec_j,
            cfg,
            active_mask=prob.active_mask,
        )

        def J_fn(m):
            with torch.no_grad():
                tr = forward(
                    m, prob.src_sig, prob.src_i, prob.src_j, prob.rec_i, prob.rec_j, cfg
                ).traces
            return float(l2_misfit(tr, prob.observed, cfg.dt))

        torch.manual_seed(0)
        dm = torch.randn_like(grad) * prob.active_mask.to(grad.dtype)
        hs = [10.0**k for k in range(6, -7, -1)]
        hs, r1, r2 = taylor_test(J_fn, prob.start_alpha2, dm, grad, hs)

        slope1, _ = clean_slope(hs, r1)
        slope2, n2 = clean_slope(hs, r2)
        assert n2 >= 3
        assert abs(slope1 - 1.0) < 0.25  # first-order remainder ~ O(h)
        assert abs(slope2 - 2.0) < 0.25  # second-order remainder ~ O(h^2)
