"""Unit tests for domain I/O, geometry, and the source wavelet (Task 2).

One test class covering all three small modules (test parsimony).
All on CPU float64 for exact checks.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from fwi.config import SimConfig, SPEED_LIMESTONE
from fwi.domain import read_domain, build_alpha2
from fwi import geometry, wavelet

DATA = Path(__file__).resolve().parents[1] / "data" / "domain"
F64 = torch.float64
CPU = torch.device("cpu")


class TestIO:
    # ---- domain ----
    def test_read_domain_small_dims_and_labels(self):
        dom = read_domain(DATA / "SmallDomain2D.txt")
        # MATLAB squeeze -> (nI, nJ) = (14, 34)
        assert dom.labels.shape == (14, 34)
        assert dom.dims == (14, 34, 1)
        # 2-cell ghost border (label 0); interior is limestone (label 1)
        assert dom.labels[0, 0] == 0
        assert dom.labels[6, 15] == 1

    def test_build_alpha2_ghost_and_interior(self):
        dom = read_domain(DATA / "SmallDomain2D.txt")
        alpha2, active = build_alpha2(dom, device=CPU, dtype=F64)
        assert alpha2.shape == (14, 34)
        # ghost cell -> 0 and masked out
        assert alpha2[0, 0].item() == 0.0
        assert active[0, 0].item() is False
        # interior cell -> limestone speed^2 and active
        assert alpha2[6, 15].item() == SPEED_LIMESTONE**2
        assert active[6, 15].item() is True

    # ---- geometry ----
    def test_receiver_ring_count_and_inside_active(self):
        cfg = SimConfig()
        dom = read_domain(DATA / "Domain2D_model.txt")
        x_rec, y_rec = geometry.receiver_ring(cfg)
        assert len(x_rec) == 14 and len(y_rec) == 14
        i, j = geometry.coords_to_indices(dom, x_rec, y_rec)
        _, active = build_alpha2(dom, device=CPU, dtype=F64)
        # every receiver lands inside the grid and on a non-ghost cell
        assert int(i.min()) >= 0 and int(i.max()) < dom.dims[0]
        assert int(j.min()) >= 0 and int(j.max()) < dom.dims[1]
        assert bool(active[i, j].all())

    def test_make_sources_accepts_vector(self):
        dom = read_domain(DATA / "Domain2D_model.txt")
        xs = [117.0, 152.0, 173.0, 153.0]
        ys = [47.0, 31.0, 46.0, 71.0]
        i, j = geometry.make_sources(dom, xs, ys)
        assert i.shape == (4,) and j.shape == (4,)

    # ---- wavelet ----
    def test_wavelet_matches_matlab_formula(self):
        cfg = SimConfig()
        src = wavelet.gaussian_derivative(cfg, device=CPU, dtype=F64)
        assert src.shape == (1, cfg.nt)

        # independent scalar recomputation (different code path) at 3 times
        dt, f0, t0, scaling = cfg.dt, cfg.f0, cfg.t0, cfg.scaling
        sigma = scaling / (2.0 * math.pi * f0)
        sing = 1.0 / (cfg.dx * cfg.dy)
        for k in (0, cfg.nt // 2, cfg.nt - 1):
            t = (k + 1) * dt
            a = t - t0
            expected = (
                -(sigma**2)
                * a
                / (math.sqrt(2.0 * math.pi) * sigma)
                * math.exp(-(a**2) / (2.0 * sigma**2))
                * sing
            )
            assert abs(src[0, k].item() - expected) < 1e-9

    def test_wavelet_multi_frequency_distinct_rows(self):
        cfg = SimConfig()
        src = wavelet.gaussian_derivative(
            cfg, f0=[10000.0, 20000.0], device=CPU, dtype=F64
        )
        assert src.shape == (2, cfg.nt)
        # scale-aware: the MATLAB wavelet amplitudes are tiny, so compare relative
        # difference rather than absolute (allclose's atol would call both ~0).
        rel = (src[0] - src[1]).norm() / src[0].norm()
        assert rel > 0.1
        # different f0 -> different default t0 -> different peak location
        assert src[0].abs().argmax() != src[1].abs().argmax()
