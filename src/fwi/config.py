"""Central configuration: physical/numerical parameters and device/dtype resolution.

Values mirror the real thesis InversionToolbox setup
(``InversionToolbox/ndt/run_inversion.m`` + ``GaussianDerivativeSourceTerm.m`` +
``SolveWaveEquation.m`` + ``TestCaseGenerator.m``).

Unit convention: the domain header spacing is 1.0 (domain units); the metric grid
step is ``dx_m = dx * geometry_scaling_factor`` (1 * 1e-3 = 1 mm). The same dx_m is
used for the CFL time step and the finite-difference stencils.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Speed of compression (P-) wave per material label [m/s] (run_inversion.m).
# Label 0 = ghost layer (alpha2 = 0), 1 = aluminum, 2 = crack/defect, 3 = slow defect.
SPEED_ALUMINUM = 6420.0
SPEED_CRACK = 4800.0
SPEED_SLOW = 5000.0  # slow inclusion used by the multi-defect logo
SPEED_BY_LABEL = {1: SPEED_ALUMINUM, 2: SPEED_CRACK, 3: SPEED_SLOW}


def cfl_limit(order: int) -> float:
    """CFL constant for the 2D central-difference Laplacian (SolveWaveEquation.m)."""
    if order == 2:
        return 1.0 / math.sqrt(2.0)
    if order == 4:
        return math.sqrt(3.0 / 4.0) / math.sqrt(2.0)
    raise ValueError(f"Unsupported spatial order: {order!r} (expected 2 or 4)")


@dataclass
class SimConfig:
    """Simulation parameters for the 2D acoustic plate forward/adjoint solver."""

    nt: int = 1000  # t_end / dt = 48e-6 / 48e-9
    order: int = 4  # spatial finite-difference order (2 or 4)

    # Grid spacing in domain (header) units; metric step = dx * geometry_scaling_factor.
    dx: float = 1.0
    dy: float = 1.0
    geometry_scaling_factor: float = 1e-3  # domain-units -> metres (1 unit = 1 mm)

    # Source (real-world coordinates [mm], actuator-sensor mode; run_inversion.m).
    x_src: float = 110.0
    y_src: float = 60.0
    f0: float = 200000.0  # dominant frequency [Hz]
    # Source amplitude (GaussianDerivativeSourceTerm scalingFactor). With the
    # deviation^3 normalization this yields a well-scaled wavefield/misfit.
    scaling_factor: float = 1e7

    # Explicit time step [s] (run_inversion.m: dt = 48e-9). ~0.5x the CFL limit.
    dt_explicit: float = 48e-9

    # Adjoint kernel truncation. 0 = exact full-time gradient (matches autodiff).
    cutoff_timesteps: int = 0

    @property
    def t0(self) -> float:
        """Source delay [s] = 1 / f0 (derived so it tracks f0)."""
        return 1.0 / self.f0

    @property
    def dx_m(self) -> float:
        """Metric grid step in x [m]."""
        return self.dx * self.geometry_scaling_factor

    @property
    def dy_m(self) -> float:
        """Metric grid step in y [m]."""
        return self.dy * self.geometry_scaling_factor

    @property
    def cfl(self) -> float:
        return cfl_limit(self.order)

    @property
    def c_max(self) -> float:
        """Reference max speed used for the CFL check (aluminum)."""
        return SPEED_ALUMINUM

    @property
    def cfl_limit_dt(self) -> float:
        """The largest CFL-stable dt for these params [s]."""
        return self.cfl * min(self.dx_m, self.dy_m) / self.c_max

    @property
    def dt(self) -> float:
        """Time step [s]. Uses the explicit thesis value; warns if not CFL-stable."""
        if self.dt_explicit > self.cfl_limit_dt:
            import warnings

            warnings.warn(
                f"dt={self.dt_explicit:.2e}s exceeds CFL limit "
                f"{self.cfl_limit_dt:.2e}s - simulation may be unstable",
                stacklevel=2,
            )
        return self.dt_explicit


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
    """float64 on cpu/cuda for clean gradient tests; float32 on mps (no float64)."""
    if prefer is not None:
        return prefer
    dev = torch.device(device)
    if dev.type == "mps":
        return torch.float32
    return torch.float64
