# acoustic-fwi-ndt-pytorch

A PyTorch port of **part of my PhD code** ([thesis, TU Munich](https://mediatum.ub.tum.de/doc/1391950/1391950.pdf)):
the 2D acoustic full-waveform inversion (FWI) I used for non-destructive testing of an
aluminum plate, reimplemented from the original MATLAB `InversionToolbox`. The misfit
gradient `dJ/d(alpha2)` is computed **two independent ways** - automatic differentiation
and a hand-coded **adjoint-state** kernel - which agree to machine precision, are verified
with a Taylor / hockey-stick plot, and drive an **L-BFGS** inversion that reconstructs the
defect(s).

This ports the FWI core of the thesis (`InversionToolbox/ndt`) - the forward solver, the
adjoint gradient, the Gaussian-derivative source, and the optimization - not the whole
codebase.

## What it does

- **Forward solver** - 2D scalar acoustic wave equation `d2(phi)/dt2 = alpha2 * (d2phi/dx2 + d2phi/dy2)`,
  4th-order finite differences, Stoermer/Verlet time stepping, multi-source injection. Pure
  PyTorch, so it is differentiable and runs on CPU / CUDA / Apple MPS.
- **Two gradients** - `loss.backward()` autodiff vs a hand-coded adjoint-state kernel
  (forward solve + adjoint solve + correlation of the bare Laplacian). They match to ~1e-15.
- **Hockey-stick verification** - a Taylor test whose second-order remainder decays as `h^2`
  (slope ~2), confirming the gradient is correct.
- **L-BFGS inversion** - the thesis optimizer (`torch.optim.LBFGS`, strong-Wolfe line
  search). It inverts a dimensionless model `m = alpha2 / alpha2_background` against the
  J0-normalized misfit `J/J0` (starts at 1.0), so the standard `lr=1.0` step works.
- **Three thesis results** - single-crack L-BFGS convergence, a 50/100/200 kHz frequency
  study, and the multi-defect "L i square" logo recovered by frequency continuation.
- **Multi-shot acquisition** - round-robin / full-matrix capture: a single source moving
  from sensor to sensor (each fires in turn, the OTHER sensors record), with the misfit
  and gradient summed over shots. Combining recordings from many source positions
  illuminates the medium from many angles and sharpens the reconstruction over a single
  source position.

## Physical setup (thesis-faithful)

| Parameter | Value | Source |
|---|---|---|
| Plate | 200 x 100 mm aluminum, 1 mm grid (204 x 104 incl. ghost) | `TestCaseGenerator.m` |
| Wave speed (aluminum / crack) | 6420 / 4800 m/s | `run_inversion.m` |
| Source | Gaussian-derivative, `f0 = 200 kHz`, `scalingFactor = 1e7` | `GaussianDerivativeSourceTerm.m` |
| Time stepping | `dt = 48 ns`, `nt = 1000` | `run_inversion.m` |
| Sensors | 16 boundary sensors + 1 source at (110, 60) | `TestCaseGenerator.m` |
| Misfit | L2 waveform, normalized by initial misfit J0 | `LeastSquaresCostFunctional.m` |
| Optimizer | L-BFGS, strong-Wolfe, ~15-30 iters | `InversionToolbox` |

## Run in Colab

Open a notebook in Google Colab (free GPU) - the badges below point at this repo and the
setup cell installs everything. Each inversion prints a **classic per-iteration training
loop** (`iter k/N | loss (J/J0) | grad_norm`) and plots the loss curve. (If you fork, change
`seidlr` to your GitHub account in the links and in each notebook's first cell.)

| Notebook | What it shows | Colab |
|---|---|---|
| `notebooks/01_autodiff_fwi.ipynb` | Forward modeling on the aluminum plate + autodiff gradient | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seidlr/acoustic-fwi-ndt-pytorch/blob/main/notebooks/01_autodiff_fwi.ipynb) |
| `notebooks/02_adjoint_fwi_hockey_stick.ipynb` | Adjoint == autodiff, hockey-stick test, single-crack L-BFGS inversion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seidlr/acoustic-fwi-ndt-pytorch/blob/main/notebooks/02_adjoint_fwi_hockey_stick.ipynb) |
| `notebooks/03_frequency_study_and_logo.ipynb` | 50/100/200 kHz frequency study, multi-defect logo, multi-shot (moving-source) acquisition + CPU/GPU benchmark | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seidlr/acoustic-fwi-ndt-pytorch/blob/main/notebooks/03_frequency_study_and_logo.ipynb) |
| `notebooks/04_fwi_as_nn_training.ipynb` | FWI as classic PyTorch training: the wave solver as an `nn.Module`, the model as `nn.Parameter`, waveform loss, `LBFGS` + `loss.backward()` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seidlr/acoustic-fwi-ndt-pytorch/blob/main/notebooks/04_fwi_as_nn_training.ipynb) |

## Quickstart (uv, local)

```bash
uv sync --extra dev                       # install (torch, numpy, matplotlib, pytest)
uv run pytest -q                          # run the test suite

uv run python examples/01_forward_modeling.py        # forward: homogeneous vs cracked plate
uv run python examples/02_gradient_autodiff.py       # autodiff dJ/d(alpha2)
uv run python examples/03_gradient_adjoint.py        # adjoint kernel + hockey-stick plot
uv run python examples/04_crack_inversion_lbfgs.py   # single-crack L-BFGS convergence
uv run python examples/05_frequency_study.py         # 50/100/200 kHz frequency study (logo)
uv run python examples/06_logo_inversion.py          # multi-defect logo via continuation
uv run python examples/07_multishot_logo.py          # moving source: many recordings vs one
```

Each example writes figures to `outputs/`. Pass `--device mps` (Apple) or `--device cuda`
to override device selection.

## Device and precision

`resolve_device()` picks `cuda -> mps -> cpu`; `resolve_dtype()` uses `float64` on
CPU/CUDA and `float32` on MPS (Apple has no float64). The Gaussian-derivative source
(`scalingFactor = 1e7`) gives a wavefield well within float32 range, so all examples run on
MPS - the **hockey-stick test is cleanest on CPU/CUDA (float64)**, where the clean `h^2`
range is longest; on MPS (float32) the round-off floor is reached at a larger step `h`, so
the stick turns up earlier (expected, not a bug).

For the moving-source acquisition, `forward_multishot` / `invert_multishot_batched` run all
S shots as one `(S, nI, nJ)` batch instead of S sequential solves: the misfit is identical
(bit-for-bit), but each timestep is one stencil kernel doing Sx the work, so the GPU is
saturated rather than launch-bound. Notebook 03's benchmark measures it (~3.5x on Apple MPS
for 8 shots; larger on a CUDA GPU, where the per-launch overhead it removes is higher).

## MATLAB -> PyTorch map

The port follows the thesis `InversionToolbox/ndt` (under `resources/PhD-FWI-MATLAB/`):

| MATLAB (`InversionToolbox`) | PyTorch |
|---|---|
| `SolveWaveEquation.m` (forward solve) | `fwi/forward.py` |
| `GaussianDerivativeSourceTerm.m` | `fwi/wavelet.py` |
| `run_inversion.m`, `TestCaseGenerator.m` | `fwi/config.py`, `fwi/geometry.py` |
| `TestCase_2D_*` domains + logo | `fwi/domain.py`, `fwi/domaingen.py` |
| adjoint sensitivity kernel | `fwi/adjoint.py` |
| gradient check (FD) | `fwi/gradient_test.py` (Taylor / hockey-stick) |
| (autograd - no MATLAB analogue) | `fwi/misfit.py` `autodiff_gradient` |
| `LeastSquaresCostFunctional.m` + L-BFGS | `fwi/inversion.py` (J0-normalized, `torch.optim.LBFGS`) |

Two fidelity points:
- The adjoint kernel correlates the **bare** Laplacian `nabla^2 u` (not `alpha2 * nabla^2 u`),
  which is what makes the adjoint gradient equal the autodiff gradient exactly.
- The source is the real `GaussianDerivativeSourceTerm`
  `src = -scalingFactor * (t - t0) / (sqrt(2pi) * deviation^3) * exp(-(t-t0)^2 / (2 deviation^2))`
  with `deviation = 1 / (2 pi f0)`. Getting this scaling right is what makes the misfit sane
  (the earlier kernel-demo port produced a ~1e-50 misfit that no optimizer could move).

## Project layout

```
src/fwi/        config, domain, domaingen, geometry, wavelet, forward, misfit, adjoint,
                gradient_test, inversion, plotting, problems
examples/       01..06 runnable scripts
notebooks/      01..04 Colab notebooks
data/domain/    plate domain files (homogeneous, cracked, 2/3-crack, logo, small)
tests/          io, forward physics, gradient agreement + hockey-stick, inversion
```

## License

MIT. See `LICENSE` and `CITATION.cff`.
