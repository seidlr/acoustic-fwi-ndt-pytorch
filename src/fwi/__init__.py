"""fwi - 2D acoustic full-waveform inversion for non-destructive testing, in PyTorch.

Ported from PhD MATLAB code (NonDestructiveTesting/2DCode/PerturbedPlate). Exposes
the misfit gradient dJ/d(alpha2) two ways - autodiff and a hand-coded adjoint-state
kernel - cross-verified and checked with a Taylor / hockey-stick plot.
"""

from __future__ import annotations

from fwi.config import SimConfig, resolve_device, resolve_dtype

__all__ = ["SimConfig", "resolve_device", "resolve_dtype"]
__version__ = "0.1.0"
