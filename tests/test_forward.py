"""Physics sanity tests for the forward solver (Task 3).

Functional tests (the solver cannot be unit-isolated). CPU float64.
Includes an axis-orientation test that an X/Y stencil swap would fail.
"""

from __future__ import annotations

import torch

from fwi.config import SimConfig, SPEED_ALUMINUM
from fwi.forward import forward, forward_multishot

F64 = torch.float64
CPU = torch.device("cpu")


def _uniform_model(n, *, dx=1.0, dy=1.0):
    """Square uniform-aluminum model (no ghost) for clean physics checks."""
    cfg = SimConfig(nt=120, dx=dx, dy=dy)
    alpha2 = torch.full((n, n), SPEED_ALUMINUM**2, device=CPU, dtype=F64)
    return cfg, alpha2


def _point_source(cfg, i, j):
    from fwi import wavelet

    sig = wavelet.gaussian_derivative(cfg, device=CPU, dtype=F64)  # (1, nt)
    src_i = torch.tensor([i])
    src_j = torch.tensor([j])
    return sig, src_i, src_j


class TestForward:
    def test_stable_no_blowup(self):
        cfg, alpha2 = _uniform_model(21)
        sig, si, sj = _point_source(cfg, 10, 10)
        rec_i = torch.tensor([5])
        rec_j = torch.tensor([5])
        res = forward(alpha2, sig, si, sj, rec_i, rec_j, cfg)
        assert torch.isfinite(res.traces).all()
        # stable: bounded well below explosion. With the real source the wavefield is
        # O(1e4); an unstable (CFL-violating) scheme would blow past 1e8 quickly.
        assert float(res.traces.abs().max()) < 1e8

    def test_centered_source_symmetric_wavefield(self):
        n = 21
        cfg, alpha2 = _uniform_model(n)
        c = n // 2
        sig, si, sj = _point_source(cfg, c, c)
        rec_i = torch.tensor([c])
        rec_j = torch.tensor([c])
        res = forward(alpha2, sig, si, sj, rec_i, rec_j, cfg, capture_wavefield=True)
        field = res.wavefield[cfg.nt // 2]  # (n, n) mid-simulation snapshot
        # centered source on isotropic square grid -> symmetric under both flips
        assert torch.allclose(field, field.flip(0), atol=1e-10)
        assert torch.allclose(field, field.flip(1), atol=1e-10)

    def test_multishot_matches_stacked_single_shots(self):
        # The batched S-shot solve must equal stacking S single-shot solves to machine
        # precision: same physics, just one (S, nI, nJ) pass instead of S separate ones.
        from fwi import wavelet

        n = 21
        cfg, alpha2 = _uniform_model(n)
        sig = wavelet.gaussian_derivative(cfg, device=CPU, dtype=F64)  # (1, nt)
        src_i = torch.tensor([5, 10, 15])
        src_j = torch.tensor([6, 10, 14])
        rec_i = torch.tensor([3, 10, 17, 10])  # shared receiver ring
        rec_j = torch.tensor([10, 3, 10, 17])
        batched = forward_multishot(alpha2, sig[0], src_i, src_j, rec_i, rec_j, cfg)
        assert batched.shape == (3, 4, cfg.nt)
        for s in range(3):
            single = forward(
                alpha2, sig, src_i[s : s + 1], src_j[s : s + 1], rec_i, rec_j, cfg
            ).traces  # (R, nt)
            assert torch.allclose(batched[s], single, atol=1e-12, rtol=0.0)

    def test_axis_orientation_anisotropic_spacing(self):
        # dy spacing is coarser than dx -> wave crosses fewer cells per step in Y,
        # so the index-space extent along X (dim=1) must exceed that along Y (dim=0).
        # An X/Y stencil swap flips this and fails the assertion.
        n = 41
        cfg, alpha2 = _uniform_model(n, dx=1.0, dy=2.0)
        c = n // 2
        sig, si, sj = _point_source(cfg, c, c)
        rec_i = torch.tensor([c])
        rec_j = torch.tensor([c])
        res = forward(alpha2, sig, si, sj, rec_i, rec_j, cfg, capture_wavefield=True)
        field = res.wavefield[-1].abs()
        thr = 0.05 * float(field.max())
        rows = torch.where(field.max(dim=1).values > thr)[0]  # extent along Y (dim0)
        cols = torch.where(field.max(dim=0).values > thr)[0]  # extent along X (dim1)
        extent_y = int(rows.max() - rows.min())
        extent_x = int(cols.max() - cols.min())
        assert extent_x > extent_y
