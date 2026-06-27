"""Iterative inversion test (Task 7). Small grid, CPU float64.

Asserts the misfit drops by >=1 order of magnitude and the recovered anomaly
localizes (argmax of the model update lands in a box around the true defect).
"""

from __future__ import annotations

import torch

from fwi.problems import build_problem
from fwi.inversion import invert

F64 = torch.float64
CPU = torch.device("cpu")


class TestInversion:
    def test_misfit_drops_and_anomaly_localizes(self):
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
            grad_fn="adjoint",
            n_iter=60,
        )
        assert len(history) == 61  # n_iter pre-step misfits + the final model's misfit
        # misfit reduced by at least one order of magnitude
        assert history[-1] < history[0] / 10.0

        # recovered update localizes at the true defect (where true != start)
        update = (alpha2_hat - prob.start_alpha2).abs()
        true_diff = (prob.true_alpha2 - prob.start_alpha2).abs()
        ti, tj = torch.where(true_diff > 0)
        ui = int(update.argmax() // update.shape[1])
        uj = int(update.argmax() % update.shape[1])
        # argmax of the update should sit within 3 cells of the true defect region
        di = min(abs(ui - int(t)) for t in ti)
        dj = min(abs(uj - int(t)) for t in tj)
        assert di <= 3 and dj <= 3
