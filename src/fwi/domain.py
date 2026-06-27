"""Read the MATLAB plate domain text format and build the alpha2 model.

File format (CreateSyntheticSeismogram.m:13-39):
    line 1: offset  (oX oY oZ)
    line 2: spacing (dX dY dZ)
    line 3: dims    (nI nJ nK)
    then nI*nJ*nK integer voxel labels.

MATLAB does ``reshape(v, [nI nJ nK])`` (column-major) then ``squeeze`` to 2D, so we
reshape the flat label vector with Fortran order into shape (nI, nJ). Axis convention:
i -> Y (dim 0), j -> X (dim 1).

Labels: 0 = ghost layer (alpha2 = 0, masked out of all stencils/sums/updates),
1 = limestone, 2 = gold, 3 = oil.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fwi.config import SPEED_BY_LABEL


@dataclass
class Domain:
    """A parsed 2D domain: integer labels plus geometry."""

    labels: np.ndarray  # int array, shape (nI, nJ); i->Y, j->X
    offset: tuple[float, float, float]  # (oX, oY, oZ)
    spacing: tuple[float, float, float]  # (dX, dY, dZ)
    dims: tuple[int, int, int]  # (nI, nJ, nK)

    @property
    def ni(self) -> int:
        return self.dims[0]

    @property
    def nj(self) -> int:
        return self.dims[1]

    # ---- index <-> real-world coordinate maps (0-based indices) ----
    def y_to_i(self, y):
        oY, dY = self.offset[1], self.spacing[1]
        return np.rint((np.asarray(y, dtype=float) - oY) / dY).astype(np.int64)

    def x_to_j(self, x):
        oX, dX = self.offset[0], self.spacing[0]
        return np.rint((np.asarray(x, dtype=float) - oX) / dX).astype(np.int64)

    def i_to_y(self, i):
        oY, dY = self.offset[1], self.spacing[1]
        return np.asarray(i, dtype=float) * dY + oY

    def j_to_x(self, j):
        oX, dX = self.offset[0], self.spacing[0]
        return np.asarray(j, dtype=float) * dX + oX


def read_domain(path: str | Path) -> Domain:
    tokens = Path(path).read_text().split()
    nums = iter(tokens)
    offset = (float(next(nums)), float(next(nums)), float(next(nums)))
    spacing = (float(next(nums)), float(next(nums)), float(next(nums)))
    nI, nJ, nK = (
        int(float(next(nums))),
        int(float(next(nums))),
        int(float(next(nums))),
    )
    flat = np.array([int(float(t)) for t in nums], dtype=np.int64)
    expected = nI * nJ * nK
    if flat.size != expected:
        raise ValueError(f"{path}: expected {expected} labels, found {flat.size}")
    # column-major (Fortran) reshape mirrors MATLAB reshape([nI nJ nK]); squeeze nK.
    labels = flat.reshape((nI, nJ, nK), order="F")[:, :, 0]
    return Domain(labels=labels, offset=offset, spacing=spacing, dims=(nI, nJ, nK))


def build_alpha2(
    domain: Domain,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map labels -> alpha2 (= c^2) and an active (non-ghost) boolean mask.

    Returns (alpha2, active_mask) where active_mask is True on physical cells.
    Ghost cells (label 0) have alpha2 = 0 and active_mask = False.
    """
    labels = domain.labels
    alpha2 = np.zeros(labels.shape, dtype=np.float64)
    for label, speed in SPEED_BY_LABEL.items():
        alpha2[labels == label] = speed**2
    # ghost layer (label 0) stays 0 (AdjointMethod.m:130)
    active = labels != 0
    alpha2_t = torch.tensor(alpha2, device=device, dtype=dtype)
    active_t = torch.tensor(active, device=device, dtype=torch.bool)
    return alpha2_t, active_t
