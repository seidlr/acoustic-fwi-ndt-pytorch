"""L-BFGS inversion test (R3). Small grid, CPU float64.

Asserts: the J0-normalized misfit starts ~1, the strong-Wolfe line search accepts a
step (J decreases on iter 1, no NaN), reduces J/J0 by >=1 order, and the recovered
anomaly localizes near the true defect.
"""

from __future__ import annotations

import math

import torch

from fwi.problems import build_problem, build_multishot_problem
from fwi.inversion import invert, invert_multishot
from fwi.misfit import l2_misfit

F64 = torch.float64
CPU = torch.device("cpu")


class TestInversion:
    def test_start_misfit_is_well_scaled(self):
        # Regression guard: the real source must yield a SANE misfit (not the ~1e-50
        # of the old mis-scaled wavelet) so L-BFGS can work.
        prob = build_problem("crack", device=CPU, dtype=F64)
        from fwi.forward import forward

        syn = forward(
            prob.start_alpha2, prob.src_sig, prob.src_i, prob.src_j,
            prob.rec_i, prob.rec_j, prob.cfg,
        ).traces
        J = float(l2_misfit(syn, prob.observed, prob.cfg.dt))
        assert 1e-6 < J < 1e4, f"start misfit {J:.2e} is not well-scaled"

    def test_lbfgs_reduces_misfit_and_localizes(self):
        prob = build_problem("small", device=CPU, dtype=F64)
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
            n_iter=20,
        )
        # normalized objective starts at ~1.0, never NaN
        assert all(math.isfinite(h) for h in history)
        assert abs(history[0] - 1.0) < 0.5
        # line search accepted a step: J strictly decreases on the first iteration
        assert history[1] < history[0]
        # reduced by >= one order of magnitude
        assert history[-1] < history[0] / 10.0

        # recovered update localizes at the true defect
        update = (alpha2_hat - prob.start_alpha2).abs()
        true_diff = (prob.true_alpha2 - prob.start_alpha2).abs()
        ti, tj = torch.where(true_diff > 0)
        ui = int(update.argmax() // update.shape[1])
        uj = int(update.argmax() % update.shape[1])
        di = min(abs(ui - int(t)) for t in ti)
        dj = min(abs(uj - int(t)) for t in tj)
        assert di <= 3 and dj <= 3

    def test_invert_multishot_reduces_misfit_and_localizes(self):
        # Round-robin acquisition: each sensor fires in turn, the OTHER sensors record;
        # the misfit/gradient sum over shots. Should reduce J/J0 and localize.
        prob = build_multishot_problem("small", device=CPU, dtype=F64)
        n_shots = len(prob.shots)
        assert n_shots >= 6
        # pitch-catch: every shot excludes its own source sensor
        assert all(int(s.rec_i.shape[0]) == n_shots - 1 for s in prob.shots)
        alpha2_hat, history = invert_multishot(
            prob.start_alpha2,
            prob.shots,
            prob.src_sig,
            prob.cfg,
            active_mask=prob.active_mask,
            n_iter=20,
        )
        assert all(math.isfinite(h) for h in history)
        assert abs(history[0] - 1.0) < 0.5
        assert history[1] < history[0]
        assert history[-1] < history[0] / 10.0

        update = (alpha2_hat - prob.start_alpha2).abs()
        true_diff = (prob.true_alpha2 - prob.start_alpha2).abs()
        ti, tj = torch.where(true_diff > 0)
        ui = int(update.argmax() // update.shape[1])
        uj = int(update.argmax() % update.shape[1])
        di = min(abs(ui - int(t)) for t in ti)
        dj = min(abs(uj - int(t)) for t in tj)
        assert di <= 3 and dj <= 3
