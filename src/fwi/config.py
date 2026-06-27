"""Central configuration: physical/numerical parameters and device/dtype resolution.

All values mirror the MATLAB reference
``NonDestructiveTesting/2DCode/PerturbedPlate/{setupGeneralParameters,setupForwardParameters}.m``.

Unit convention (the single most error-prone detail): the MATLAB grid step is
``dX/100`` (centimetre header values converted to metres). We compute
``dx_m = dx / 100`` and ``dy_m = dy / 100`` ONCE here and use the same values for
both the CFL time step and the finite-difference stencils. A raw-vs-/100
mismatch is a 100x error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Speed of compression (P-) wave per material label [m/s] (setupGeneralParameters.m).
# Label 0 = ghost layer (alpha2 = 0), 1 = limestone, 2 = gold, 3 = oil.
SPEED_LIMESTONE = 3600.0
SPEED_GOLD = 3240.0
SPEED_OIL = 3240.0
SPEED_BY_LABEL = {1: SPEED_LIMESTONE, 2: SPEED_GOLD, 3: SPEED_OIL}

# Source-amplitude lift for float32 (MPS): the native MATLAB wavefield (~1e-24) is
# below the float32 subnormal floor (~1e-45 squared in the misfit), so it underflows
# to zero. This factor raises it into float32's well-conditioned range. FWI gradients
# are invariant to a global source scale, so this does not change the recovered model.
FLOAT32_SOURCE_SCALE = 1e20


def default_source_scale(dtype) -> float:
    """1.0 for float64 (exact MATLAB fidelity); a lift for float32 to avoid underflow."""
    return FLOAT32_SOURCE_SCALE if dtype == torch.float32 else 1.0


def cfl_limit(order: int) -> float:
    """CFL constant for the 2D central-difference Laplacian (AdjointMethod.m:137-146)."""
    if order == 2:
        return 1.0 / math.sqrt(2.0)
    if order == 4:
        return math.sqrt(3.0 / 4.0) / math.sqrt(2.0)
    raise ValueError(f"Unsupported spatial order: {order!r} (expected 2 or 4)")


@dataclass
class SimConfig:
    """Simulation parameters for the 2D acoustic plate forward/adjoint solver."""

    nt: int = 800
    order: int = 4  # spatial finite-difference order (2 or 4)

    # Grid spacing in domain (header) units; the metric step is this / 100.
    dx: float = 1.0
    dy: float = 1.0

    # Default single source (real-world coordinates, setupForwardParameters.m).
    x_src: float = 130.0
    y_src: float = 65.0
    f0: float = 15000.0  # dominant frequency [Hz]
    scaling: float = 1.0
    # Global source-amplitude multiplier. The MATLAB amplitude is ~1e-11, giving a
    # wavefield ~1e-24 that underflows float32 (MPS). FWI gradients are invariant to
    # this scale, so float32 runs lift it (see FLOAT32_SOURCE_SCALE). Default 1.0
    # keeps exact MATLAB fidelity in float64.
    source_scale: float = 1.0

    # Receiver-ring geometry (setupForwardParameters.m:46-74).
    y_min: float = 35.0
    y_max: float = 98.0
    x_min: float = 80.0
    x_max: float = 200.0
    divisions_y: int = 2
    divisions_x: int = 3

    # Adjoint kernel truncation. 0 = exact full-time gradient (matches autodiff).
    # The MATLAB value (100) is a cosmetic truncation, used only for the
    # historical kernel image, never for gradient verification or inversion.
    cutoff_timesteps: int = 0

    @property
    def t0(self) -> float:
        """Source delay [s]; defaults to 1 / f0 (derived so it tracks f0 changes)."""
        return 1.0 / self.f0

    @property
    def dx_m(self) -> float:
        """Metric grid step in x [m]."""
        return self.dx / 100.0

    @property
    def dy_m(self) -> float:
        """Metric grid step in y [m]."""
        return self.dy / 100.0

    @property
    def cfl(self) -> float:
        return cfl_limit(self.order)

    @property
    def c_max(self) -> float:
        """Reference max speed used for the CFL step (limestone, per the MATLAB)."""
        return SPEED_LIMESTONE

    @property
    def dt(self) -> float:
        """Time step [s] from the CFL condition (CreateSyntheticSeismogram.m:76)."""
        return self.cfl / 2.0 * min(self.dx_m, self.dy_m) / self.c_max


def resolve_device(prefer: str | None = None) -> torch.device:
    """Pick a compute device: explicit override, else cuda -> mps -> cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(
    device: torch.device | str, prefer: torch.dtype | None = None
) -> torch.dtype:
    """float64 on cpu/cuda for clean gradient tests; float32 on mps (no float64 support)."""
    if prefer is not None:
        return prefer
    dev = torch.device(device)
    if dev.type == "mps":
        return torch.float32
    return torch.float64
