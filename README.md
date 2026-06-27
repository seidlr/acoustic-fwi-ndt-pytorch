# acoustic-fwi-ndt-pytorch

A PyTorch port of **part of my PhD code** ([thesis, TU Munich](https://mediatum.ub.tum.de/doc/1391950/1391950.pdf)):
the 2D acoustic full-waveform inversion (FWI) I used for non-destructive testing of a
plate, reimplemented from the original MATLAB. The misfit gradient `dJ/d(alpha2)` is
computed **two independent ways** - automatic differentiation and a hand-coded
**adjoint-state** kernel - which agree to machine precision, are verified with a Taylor /
hockey-stick plot, and drive a full iterative inversion that reconstructs the defect(s).

This ports the `NonDestructiveTesting/2DCode/PerturbedPlate` example - one piece of the
thesis codebase, not the whole thing.

## What it does

- **Forward solver** - 2D scalar acoustic wave equation `d2(phi)/dt2 = alpha2 * (d2phi/dx2 + d2phi/dy2)`,
  4th-order finite differences, Stoermer/Verlet time stepping, multi-source injection. Pure
  PyTorch, so it is differentiable and runs on CPU / CUDA / Apple MPS.
- **Two gradients** - `loss.backward()` autodiff vs a hand-coded adjoint-state kernel
  (forward solve + adjoint solve + correlation of the bare Laplacian). They match to ~1e-15.
- **Hockey-stick verification** - a Taylor test whose second-order remainder decays as `h^2`
  (slope ~2), confirming the gradient is correct.
- **Iterative inversion** - reconstructs single- and multi-defect perturbations and shows
  misfit convergence.

## Run in Colab

Open a notebook in Google Colab (free GPU) — the badges below point at this repo and the
setup cell installs everything. (If you fork, change `seidlr` to your GitHub account in the
links and in each notebook's first cell.)

| Notebook | What it shows | Colab |
|---|---|---|
| `notebooks/01_autodiff_fwi.ipynb` | Forward modeling, autodiff gradient, autodiff-driven inversion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seidlr/acoustic-fwi-ndt-pytorch/blob/main/notebooks/01_autodiff_fwi.ipynb) |
| `notebooks/02_adjoint_fwi_hockey_stick.ipynb` | Adjoint-state gradient, agreement with autodiff, hockey-stick test, inversion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seidlr/acoustic-fwi-ndt-pytorch/blob/main/notebooks/02_adjoint_fwi_hockey_stick.ipynb) |
| `notebooks/03_multidefect_multisource.ipynb` | 3 defects + 4 combined sources of different wavelengths | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seidlr/acoustic-fwi-ndt-pytorch/blob/main/notebooks/03_multidefect_multisource.ipynb) |

## Quickstart (uv, local)

```bash
uv sync --extra dev                       # install (torch, numpy, matplotlib, pytest)
uv run pytest -q                          # run the test suite

uv run python examples/01_forward_modeling.py        # regenerate seismograms + snapshots
uv run python examples/02_gradient_autodiff.py       # autodiff dJ/d(alpha2)
uv run python examples/03_gradient_adjoint.py        # adjoint kernel + hockey-stick plot
uv run python examples/04_inversion.py               # iterative inversion
uv run python examples/05_multidefect_multisource.py # 3 defects, 4 combined sources
```

Each example writes figures to `outputs/`. Add `--device mps` (Apple) or `--device cuda`
to the gradient/inversion examples; `--grid full` runs the real 104x304 plate.

## Device and precision

`resolve_device()` picks `cuda -> mps -> cpu`; `resolve_dtype()` uses `float64` on
CPU/CUDA and `float32` on MPS (Apple has no float64).

- The **hockey-stick test is cleanest on CPU or CUDA (float64)**. On Apple **MPS (float32)**
  the round-off floor is reached at a much larger step `h`, so the clean `h^2` range is
  shorter and the stick turns up earlier - expected, not a bug.
- The native MATLAB source amplitude is ~1e-11, giving a wavefield ~1e-24 that underflows
  float32. Since FWI gradients are invariant to a global source scale, float32 runs lift it
  (`SimConfig.source_scale`, auto-set in `build_problem`); float64 keeps the exact MATLAB
  scale. The recovered model is unchanged either way.

## MATLAB -> PyTorch map

| MATLAB | PyTorch |
|---|---|
| `CreateSyntheticSeismogram.m` (forward solve) | `fwi/forward.py` |
| `setup*Parameters.m` | `fwi/config.py`, `fwi/geometry.py`, `fwi/wavelet.py` |
| domain `*.txt` reader | `fwi/domain.py` |
| `AdjointMethod.m` (sensitivity kernel) | `fwi/adjoint.py` |
| `GradientChecking.m` (FD gradient test) | `fwi/gradient_test.py` (Taylor / hockey-stick) |
| (autograd - no MATLAB analogue) | `fwi/misfit.py` `autodiff_gradient` |
| `InversionToolbox` optimization | `fwi/inversion.py` |

Key fidelity point: the adjoint kernel correlates the **bare** Laplacian `nabla^2 u`
(not `alpha2 * nabla^2 u`), matching `CreateSyntheticSeismogram.m:184`; this is what makes
the adjoint gradient equal the autodiff gradient exactly.

## Project layout

```
src/fwi/        config, domain, geometry, wavelet, forward, misfit, adjoint,
                gradient_test, inversion, plotting, problems
examples/       01..05 runnable scripts
notebooks/      01..03 Colab notebooks
data/domain/    plate domain files (start, 1/2/3-defect, complex, small)
tests/          io, forward physics, gradient agreement + hockey-stick, inversion
```

## License

MIT. See `LICENSE` and `CITATION.cff`.
