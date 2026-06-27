"""Source and receiver placement on the grid.

Receiver ring follows setupForwardParameters.m:46-74 (sensors on the four plate
edges). For the adjoint solve the receivers become sources (setupAdjointParameters.m).
Real-world coordinates are mapped to 0-based grid indices via the Domain coord maps.
"""

from __future__ import annotations

import numpy as np
import torch

from fwi.config import SimConfig
from fwi.domain import Domain


def receiver_ring(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Real-world (x, y) receiver coordinates on the four plate edges.

    Two vertical edges (x = x_min, x_max) sampled at divisions_y+1 y-values, plus
    two horizontal edges (y = y_min, y_max) sampled at divisions_x+1 x-values.
    Default config -> 6 + 8 = 14 receivers.
    """
    ys = np.linspace(cfg.y_min, cfg.y_max, cfg.divisions_y + 1)  # left/right edges
    xs = np.linspace(cfg.x_min, cfg.x_max, cfg.divisions_x + 1)  # top/bottom edges

    # left + right vertical edges
    x_edge_v = np.concatenate(
        [np.full_like(ys, cfg.x_min), np.full_like(ys, cfg.x_max)]
    )
    y_edge_v = np.concatenate([ys, ys])
    # top + bottom horizontal edges
    x_edge_h = np.concatenate([xs, xs])
    y_edge_h = np.concatenate(
        [np.full_like(xs, cfg.y_min), np.full_like(xs, cfg.y_max)]
    )

    x_rec = np.concatenate([x_edge_v, x_edge_h])
    y_rec = np.concatenate([y_edge_v, y_edge_h])
    return x_rec, y_rec


def coords_to_indices(domain: Domain, x, y) -> tuple[torch.Tensor, torch.Tensor]:
    """Map real-world (x, y) to 0-based grid (i, j) index tensors (long)."""
    i = torch.as_tensor(np.atleast_1d(domain.y_to_i(y)), dtype=torch.long)
    j = torch.as_tensor(np.atleast_1d(domain.x_to_j(x)), dtype=torch.long)
    return i, j


def make_sources(domain: Domain, x_src, y_src) -> tuple[torch.Tensor, torch.Tensor]:
    """Source grid indices (i, j); accepts scalars or vectors (multi-source)."""
    return coords_to_indices(domain, x_src, y_src)


def make_receivers(domain: Domain, cfg: SimConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Receiver grid indices (i, j) for the configured ring."""
    x_rec, y_rec = receiver_ring(cfg)
    return coords_to_indices(domain, x_rec, y_rec)


def setup_adjoint_sources(
    rec_i: torch.Tensor, rec_j: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjoint sources sit at the receiver locations (setupAdjointParameters.m)."""
    return rec_i, rec_j
