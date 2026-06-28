"""Source and sensor placement on the 200x100 mm plate.

Sensor layout is the thesis "ThinBoundary_16Sensors" (setSensorPositionCase.m case 0)
and the source is at (110, 60) mm (run_inversion.m, actuator-sensor mode). Real-world
coordinates are mapped to 0-based grid indices via the Domain coord maps (y->i, x->j).
"""

from __future__ import annotations

import numpy as np
import torch

from fwi.config import SimConfig
from fwi.domain import Domain

# Thesis 16-sensor "thin complete boundary" (setSensorPositionCase.m case 0) [mm].
_SENSOR_X = np.array(
    [30, 30, 170, 170, 40, 100, 160, 40, 100, 160, 30, 199, 70, 130, 70, 130], float
)
_SENSOR_Y = np.array(
    [35, 70, 35, 70, 30, 30, 30, 70, 70, 70, 52, 52, 30, 30, 70, 70], float
)


def sensor_ring(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Real-world (x, y) sensor coordinates - the thesis 16-sensor thin boundary."""
    return _SENSOR_X.copy(), _SENSOR_Y.copy()


def source_position(cfg: SimConfig) -> tuple[float, float]:
    """Single actuator source location (x, y) [mm]."""
    return cfg.x_src, cfg.y_src


def coords_to_indices(domain: Domain, x, y) -> tuple[torch.Tensor, torch.Tensor]:
    """Map real-world (x, y) to 0-based grid (i, j) index tensors (long).

    Validates that x and y pair up and that every index lands inside the grid -
    otherwise PyTorch fancy-indexing would silently wrap a negative/oversized index
    to the wrong cell, injecting or recording at the wrong location with no error.
    """
    xa = np.atleast_1d(np.asarray(x, dtype=float))
    ya = np.atleast_1d(np.asarray(y, dtype=float))
    if xa.shape != ya.shape:
        raise ValueError(
            f"x and y must have matching shapes, got {xa.shape} vs {ya.shape}"
        )
    i = torch.as_tensor(domain.y_to_i(ya), dtype=torch.long)
    j = torch.as_tensor(domain.x_to_j(xa), dtype=torch.long)
    nI, nJ = domain.ni, domain.nj
    if int(i.min()) < 0 or int(i.max()) >= nI or int(j.min()) < 0 or int(j.max()) >= nJ:
        raise ValueError(
            f"coordinate maps outside the {nI}x{nJ} grid: "
            f"i in [{int(i.min())},{int(i.max())}], j in [{int(j.min())},{int(j.max())}]"
        )
    return i, j


def make_sources(domain: Domain, x_src, y_src) -> tuple[torch.Tensor, torch.Tensor]:
    """Source grid indices (i, j); accepts scalars or vectors (multi-source)."""
    return coords_to_indices(domain, x_src, y_src)


def make_receivers(domain: Domain, cfg: SimConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Sensor grid indices (i, j) for the thesis 16-sensor ring."""
    x_rec, y_rec = sensor_ring(cfg)
    return coords_to_indices(domain, x_rec, y_rec)
