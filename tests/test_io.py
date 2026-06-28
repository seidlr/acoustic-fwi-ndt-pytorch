"""Unit tests for domain I/O, geometry, and the source wavelet (Task 2).

One test class covering all three small modules (test parsimony).
All on CPU float64 for exact checks.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from fwi.config import SimConfig, SPEED_ALUMINUM, SPEED_CRACK
from fwi.domain import read_domain, build_alpha2
from fwi import geometry, wavelet

DATA = Path(__file__).resolve().parents[1] / "data" / "domain"
F64 = torch.float64
CPU = torch.device("cpu")


class TestIO:
    # ---- domain ----
    def test_read_domain_plate_dims_and_crack_not_transposed(self):
        dom = read_domain(DATA / "cracked.txt")
        # package convention (nI=Y=104, nJ=X=204) - NOT transposed
        assert dom.labels.shape == (104, 204)
        assert dom.dims == (104, 204, 1)
        assert dom.labels[0, 0] == 0  # ghost border
        # crack at rows Y 48..50, cols X 79..109 (MATLAB data(80:110,49:51) translated)
        import numpy as np

        yi, xj = np.where(dom.labels == 2)
        assert (yi.min(), yi.max()) == (48, 50)
        assert (xj.min(), xj.max()) == (79, 109)

    def test_build_alpha2_ghost_and_materials(self):
        dom = read_domain(DATA / "cracked.txt")
        alpha2, active = build_alpha2(dom, device=CPU, dtype=F64)
        assert alpha2.shape == (104, 204)
        assert alpha2[0, 0].item() == 0.0  # ghost -> 0
        assert active[0, 0].item() is False
        assert alpha2[50, 150].item() == SPEED_ALUMINUM**2  # aluminum interior
        assert alpha2[49, 90].item() == SPEED_CRACK**2  # crack cell

    # ---- geometry ----
    def test_sensor_ring_count_and_inside_active(self):
        cfg = SimConfig()
        dom = read_domain(DATA / "homogeneous.txt")
        x_rec, y_rec = geometry.sensor_ring(cfg)
        assert len(x_rec) == 16 and len(y_rec) == 16  # thesis thin-boundary sensors
        i, j = geometry.coords_to_indices(dom, x_rec, y_rec)
        _, active = build_alpha2(dom, device=CPU, dtype=F64)
        assert int(i.min()) >= 0 and int(i.max()) < dom.dims[0]
        assert int(j.min()) >= 0 and int(j.max()) < dom.dims[1]
        assert bool(active[i, j].all())  # all sensors on non-ghost cells

    def test_make_sources_accepts_vector(self):
        dom = read_domain(DATA / "homogeneous.txt")
        xs = [40.0, 100.0, 160.0, 130.0]
        ys = [30.0, 30.0, 70.0, 70.0]
        i, j = geometry.make_sources(dom, xs, ys)
        assert i.shape == (4,) and j.shape == (4,)

    # ---- wavelet ----
    def test_wavelet_matches_gaussian_derivative_formula(self):
        # Real GaussianDerivativeSourceTerm.m: deviation^3 normalization, scalingFactor.
        cfg = SimConfig()
        src = wavelet.gaussian_derivative(cfg, device=CPU, dtype=F64)
        assert src.shape == (1, cfg.nt)

        # numpy float64 reference with the SAME formula (the ~1e24 prefactor makes
        # hand-arithmetic error-prone), compared at ALL samples.
        t = (np.arange(1, cfg.nt + 1)) * cfg.dt
        deviation = 1.0 / (2.0 * math.pi * cfg.f0)
        sing = 1.0 / (cfg.dx * cfg.dy)
        a = t - cfg.t0
        expected = (
            -cfg.scaling_factor
            * a
            / (math.sqrt(2.0 * math.pi) * deviation**3)
            * np.exp(-(a**2) / (2.0 * deviation**2))
            * sing
        )
        got = src[0].cpu().numpy()
        rel = np.linalg.norm(got - expected) / np.linalg.norm(expected)
        assert rel < 1e-12
        # amplitude is now well-scaled (large), NOT the old ~1e-11
        assert float(np.abs(expected).max()) > 1e10

    def test_wavelet_multi_frequency_distinct_rows(self):
        cfg = SimConfig()
        src = wavelet.gaussian_derivative(
            cfg, f0=[100000.0, 200000.0], device=CPU, dtype=F64
        )
        assert src.shape == (2, cfg.nt)
        rel = (src[0] - src[1]).norm() / src[0].norm()
        assert rel > 0.1
        # different f0 -> different default t0 -> different peak location
        assert src[0].abs().argmax() != src[1].abs().argmax()
