# 2D Plate FWI — MATLAB→PyTorch Port Implementation Plan

Created: 2026-06-27
Author: robertseidl425@gmail.com
Agent: Claude Code
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Port the MATLAB `NonDestructiveTesting/2DCode/PerturbedPlate` full-waveform-inversion example to a clean open-source PyTorch package that runs on CPU/MPS/CUDA (incl. Colab), exposing dJ/dα² two ways — (1) autodiff via `loss.backward()` and (2) a hand-coded adjoint-state kernel — verified against each other and with a Taylor/"hockey-stick" plot, plus a full iterative inversion, a more complex multi-defect / multi-source / multi-wavelength variant, and Colab-runnable notebooks.

## Approach

**Chosen:** A single differentiable `fwi` package (src layout, uv-managed, pip-installable for Colab) whose `forward.py` solver is pure-torch so the *same* forward run serves both gradient paths — autograd backprops through it, and the manual `adjoint.py` re-runs the forward (capturing the bare Laplacian field) for the adjoint-state correlation. Thin example scripts + three Colab notebooks drive it; an open-source repo shell (LICENSE, CI, README with Run-in-Colab table) wraps it.
**Why:** One physics implementation feeding both gradients makes their agreement a real correctness check (a divergence is a bug, not two different models); pure-torch ops give autodiff for free at the cost of retaining the time-stepping graph (mitigated by small-grid defaults / checkpointing — see Context).

## Context for Implementer

Source MATLAB (read-only reference, symlinked at `resources/PhD-FWI-MATLAB/`):
`NonDestructiveTesting/2DCode/PerturbedPlate/{CreateSyntheticSeismogram,CreateObservedSeismogram,AdjointMethod,setupForwardParameters,setupGeneralParameters,setupAdjointParameters}.m`; gradient-test reference `AdjointMethodOptimization/{GradientChecking,gradientCheckingAdjoint}.m`.

Physics & numerics to replicate exactly (line refs are into the `.m` files above):
- **PDE:** 2D scalar acoustic wave eq. `∂²φ/∂t² = α²·(∂²φ/∂x² + ∂²φ/∂y²)`. Model parameter is `α²` (squared P-wave speed, `alpha2`), one value per cell. Invert for `alpha2` (matches MATLAB), not velocity.
- **Axis convention (must be explicit in code):** MATLAB index `i`→Y (tensor `dim=0`), `j`→X (tensor `dim=1`). The X second derivative uses spacing `dx_m` on `dim=1`; the Y second derivative uses `dy_m` on `dim=0`. Swapping them is numerically stable but wrong and the symmetry test on the non-square 14×34 grid will NOT catch it — assert the mapping directly.
- **Unit fix:** MATLAB uses `dX/100` as the metric grid step (`CreateSyntheticSeismogram.m:158`). Define `dx_m = dX/100.0`, `dy_m = dY/100.0` ONCE in `config.py` and use the same values for `dt` (CFL) and the stencils — a raw-vs-/100 mismatch is a 100× error.
- **Domain format** (`data/domain/*.txt`): line1 offset `(oX oY oZ)`, line2 spacing `(dX dY dZ)`, line3 dims `(nI nJ nK)`, then `nI*nJ*nK` int labels (MATLAB `reshape([nI nJ nK])` then `squeeze` → 2D `nI×nJ`). Labels: `0`=ghost (α²=0, masked from ALL stencils/sums/updates), `1`=limestone c=3600, `2`=gold c=3240, `3`=oil c=3240 (`setupGeneralParameters.m`).
- **Spatial FD:** central stencils, default 4th order (`OrderInSpace=4`): `(-f₋₂+16f₋₁−30f₀+16f₊₁−f₊₂)/(12·δ²)`; 2nd order also supported. Evaluate only on non-ghost cells; ghost cells stay zero (Dirichlet).
- **Time stepping:** Störmer/Verlet `φ_new = 2φ − φ_old + α²·∇²φ·dt²`; source(s) added as `φ_new[i_s,j_s] += src_s(t)·dt²`. `nt=800`. `dt = CFL/2·min(dx_m,dy_m)/c_max` with `CFL=√(3/4)/√2` (4th order), `c_max=limestone` (per the `.m`).
- **Source wavelet** (`CreateSyntheticSeismogram.m:92-97`): Gaussian-derivative; `sigma=scaling/(2π f0)`, `src=−sigma²·(t−t0)/(√(2π)·sigma)·exp(−(t−t0)²/(2 sigma²))·(1/(dX·dY))`. **Per-source arrays**: `f0`, `t0`, `scaling`, location may be vectors → one wavelet per source (different wavelengths/delays). Reproduce values, not just shape.
- **Receiver geometry** (`setupForwardParameters.m:46-74`): a ring of sensors on the plate edges (14 sensors for the default config). For the adjoint solve these receivers become adjoint sources (`setupAdjointParameters.m`).
- **Misfit:** L2 waveform `J = ½·Σₜ |u−u₀|²·dt` over receivers. **Adjoint source = flip(u−u₀, dim=-1)** (time-reversed residual, equivalent to MATLAB `u(:,end:-1:1)-u0(:,end:-1:1)`, `AdjointMethod.m:15`); no windowing applied (`startp=0, endp=nt`).
- **Adjoint-state gradient** (`AdjointMethod.m:196-244`):
  - The forward solver stores `nabla2u(t) = d²φ/dx² + d²φ/dy²` — the **BARE Laplacian, NOT multiplied by α²** (`CreateSyntheticSeismogram.m:184`; line 183 with `alpha2.*` is the commented-out wrong version). This is dResidual/dα², so the kernel is the true dJ/dα². Storing `α²·∇²φ` instead introduces a spatially-varying α²(x) scale error that makes autodiff and adjoint disagree by exactly that factor.
  - `adjoint_gradient()` **re-runs the forward solver internally** with `capture_nabla2u=True` (the trace `.npz` from Task 3 does NOT contain the Laplacian stack — mirrors how `AdjointMethod.m` regenerates it). Then it does the adjoint solve (residual sources at receivers) → `λ(t)`, and `kernel = −Σₜ λ(t)·nabla2u(T−t)·dt`.
  - **`cutoff` is NOT part of the gradient.** The MATLAB `cutoff_timesteps=100` truncates the correlation sum (skips the last 100 adjoint steps), which makes the kernel an *approximation* of dJ/dα², not the exact gradient. Autodiff differentiates the FULL misfit, so to make the autodiff≡adjoint agreement and the Taylor test exact, `adjoint_gradient()` defaults to **`cutoff=0`** (full time integral = the true gradient). The MATLAB `cutoff=100` variant is reproduced ONLY as an optional, clearly-labeled historical kernel image — it is never used in the agreement test, Taylor test, or inversion.
  - **Index bounds:** with `cutoff=0` the sum runs over all steps; for a general `cutoff`, if the adjoint loop runs forward in time `t=0..nt-1` injecting the reversed residual, accumulate while `t < nt - cutoff` and pair adjoint step `t` with `nabla2u[nt-1-t]`. State the exact index pairing in `adjoint.py` for whichever loop direction is implemented.

Cross-cutting constraints every task respects:
- **Ghost mask is sacred:** α²=0 on ghost cells; ghost cells excluded from stencils, misfit, kernels, and inversion updates — otherwise the gradient test will not converge.
- **Multi-source is "combined", not multi-shot:** the multi-source variant fires several sources into ONE wavefield in a single forward run (each with its own `f0`/`t0`); receivers record the superposition. The adjoint structure is unchanged (one residual at the receivers → one adjoint solve → one kernel). So multi-source/multi-defect is a forward-setup + domain change, not a new solver.
- **Memory:** autograd retains the full `nt`-step graph. Fine on the small 14×34 grid (agreement + Taylor tests) and the autodiff example default; on the full 104×304 grid an autograd graph is ~GB-scale on MPS float32, so the **autodiff example/notebook default to the small/medium grid** (or gradient checkpointing), while the manual adjoint (O(grid) memory, recomputes) drives the full plate and multi-defect runs.

Device/precision: auto-select `cuda → mps → cpu`; dtype `float64` on cpu/cuda, **`float32` on mps** (no float64 on Apple MPS). Hockey-stick is cleanest on CPU float64; on MPS float32 the second-order remainder bottoms out at a larger step `h` (round-off floor ~1e-7 vs ~1e-15), so the "stick" turns up earlier — expected, documented in the notebook/README, not a bug.

**Deviation (added in impl, Task 7) — normalized-gradient Adam instead of LBFGS.** The gradient magnitude is ~1e-57 (float64, native source scale), far below Adam's `eps=1e-8`, so raw Adam/LBFGS zero the update. `invert()` normalizes the gradient to unit max-abs each iteration (absolute scale cancels; `lr` becomes a step in alpha2 units) and defaults to Adam over the normalized direction; sgd/lbfgs remain selectable. Verified: small-grid inversion reduces misfit ~275× and localizes the defect exactly.

**Deviation (added in impl, Task 4) — source-amplitude lift for float32.** The native MATLAB source amplitude (~1e-11) yields a wavefield ~1e-24 and misfit ~1e-51, which underflow to ZERO in float32 (MPS) — the gradient would be identically zero, not just less clean. Since FWI gradients are invariant to a global source amplitude, `SimConfig.source_scale` (default 1.0 = exact MATLAB fidelity in float64) is lifted to `FLOAT32_SOURCE_SCALE = 1e20` for float32 runs (`default_source_scale(dtype)`, applied in `build_problem`). Verified: MPS float32 misfit/gradient equal the CPU float64 values times (1e20)², i.e. identical physics, recovered model unchanged. The wavelet unit test keeps the default scale 1.0 and matches MATLAB exactly.

## File Structure

Open-source repo shell:
- `pyproject.toml` (create) — uv/hatchling project `fwi`, src layout, deps `torch`/`numpy`/`matplotlib`, dev `pytest`; pip-installable (`pip install -e .`) for Colab.
- `README.md` (create) — overview, MATLAB→PyTorch map, `uv` quickstart, **Run-in-Colab table**, MPS hockey-stick caveat.
- `LICENSE` (create) — MIT (default; user can change).
- `.gitignore` (create) — Python/uv/`outputs/`/checkpoints.
- `CITATION.cff` (create) — cite the PhD work (research code).
- `.github/workflows/ci.yml` (create) — `uv sync` + `uv run pytest` on push.

Package (`src/fwi/`):
- `config.py` — `SimConfig` (nt, order, speeds, geometry, `dx_m/dy_m`, CFL `dt`) + `resolve_device()/resolve_dtype()`.
- `domain.py` — `read_domain`, `build_alpha2` (+ ghost mask), coord maps.
- `geometry.py` — source & receiver placement (single + multi-source); `setup_adjoint_sources`.
- `wavelet.py` — `gaussian_derivative` with per-source `f0/t0/scaling` arrays.
- `forward.py` — `forward(alpha2, sources, rec_idx, cfg, *, capture_nabla2u=False)` → receiver traces (+ optional `nabla2u` stack, wavefield). Pure torch, autograd-ready, multi-source injection, explicit axis convention.
- `misfit.py` — `l2_misfit`, `adjoint_source` (flip).
- `adjoint.py` — `adjoint_gradient(alpha2, u0, sources, cfg)` (re-runs forward with `capture_nabla2u=True`, adjoint solve, `−Σ λ·nabla2u(T−t)·dt`, cutoff).
- `gradient_test.py` — `taylor_test(J_fn, m, dm, grad, steps)` (dm non-orthogonality guard, auto round-off range).
- `inversion.py` — `invert(...)` (autodiff or adjoint grad, LBFGS/Adam/SD, ghost-masked, misfit history).
- `plotting.py` — wavefield snapshot, kernel, hockey-stick log-log, convergence, reconstructed model.

Examples (`examples/`): `01_forward_modeling.py`, `02_gradient_autodiff.py`, `03_gradient_adjoint.py`, `04_inversion.py`, `05_multidefect_multisource.py`.

Notebooks (`notebooks/`): `01_autodiff_fwi.ipynb` (AD gradient + AD inversion), `02_adjoint_fwi_hockey_stick.ipynb` (manual adjoint + hockey-stick verification + adjoint inversion), `03_multidefect_multisource.ipynb` (complex variant). All start with device auto-detect + `pip install -e .` cell.

Data (`data/domain/*.txt`, copied from `resources/.../PerturbedPlate/data/domain/`): `Domain2D_model.txt` (start), `Domain2D_Fichtner_1pt_oben.txt` (1 defect), `Domain2D_Fichtner_2pt.txt` (2), `Domain2D_Fichtner_3pt_oben.txt` (3), `Domain2D_CiE.txt` (complex), `SmallDomain2D.txt`, `SmallDomain2D_1pt.txt`.

Tests (`tests/`): `test_io.py`, `test_forward.py`, `test_gradients.py`, `test_inversion.py`.

## Assumptions

- The excluded `.mat` seismograms are NOT needed: observed/synthetic data is regenerated from the domain `.txt` files (present locally). — Tasks 3, 8.
- No MATLAB runtime, so port fidelity is established by internal cross-validation (autodiff≡adjoint≡finite-difference), not by diffing MATLAB `.mat`. — Tasks 5, 6.
- Colab links need the repo on GitHub; the `OWNER` in the Colab URLs is set when the repo is pushed (ask the user for the GitHub account if unknown). `git init`/push happen only with the user's go-ahead (git writes need permission).

## Goal Verification

### Truths

1. On the perturbed-plate problem, the hand-coded adjoint-state gradient (cutoff=0, full time window) and the autodiff gradient (full misfit) agree to tolerance (relative L2 ≤ 1e-5 AND directional derivative ⟨g,dm⟩ matching ≤ 1e-5, CPU float64) — they are the same dJ/dα².
2. The Taylor/hockey-stick test for the cutoff=0 adjoint gradient shows second-order convergence (second-order remainder slope ≈ 2 over an auto-detected clean step range with ≥3 clean steps) on CPU float64, with the MPS-float32 round-off floor documented.
3. The iterative inversion substantially reduces the waveform misfit (final misfit ≥1 order of magnitude below initial) and reconstructs an α² whose recovered anomaly localizes (programmatic bounding-box overlap) at the true perturbation region — for both the single-defect and the multi-defect/multi-source cases.

## Progress Tracking

- [x] Task 1: uv scaffold, open-source repo shell, config (dx_m, device/dtype), data copy
- [x] Task 2: Domain I/O, geometry (multi-source), per-wavelength wavelet (+ unit tests)
- [x] Task 3: Differentiable multi-source forward solver (bare nabla2u capture) + forward example (+ physics test)
- [x] Task 4: L2 misfit & autodiff gradient + example (small-grid default)
- [x] Task 5: Manual adjoint-state gradient (re-runs forward) + adjoint≡autodiff agreement test
- [x] Task 6: Taylor/hockey-stick gradient test + plot
- [x] Task 7: Iterative inversion (programmatic anomaly check)
- [x] Task 8: Multi-defect/multi-source example + 3 notebooks + README Colab table + OSS polish

## Implementation Tasks

### Task 1: uv scaffold, open-source repo shell, config & data

**Objective:** Stand up the `fwi` package (src layout, uv-managed, pip-installable) with an open-source repo shell (LICENSE, .gitignore, CI, CITATION) and the central `SimConfig` + device/precision resolution, and copy the domain files in so the repo is self-contained (Colab has no symlink).

**Files:**
- Create: `pyproject.toml`, `src/fwi/__init__.py`, `src/fwi/config.py`, `LICENSE`, `.gitignore`, `CITATION.cff`, `.github/workflows/ci.yml`
- Create (copy): `data/domain/{Domain2D_model,Domain2D_Fichtner_1pt_oben,Domain2D_Fichtner_2pt,Domain2D_Fichtner_3pt_oben,Domain2D_CiE,SmallDomain2D,SmallDomain2D_1pt}.txt` from `resources/PhD-FWI-MATLAB/NonDestructiveTesting/2DCode/PerturbedPlate/data/domain/`

**Key Decisions / Notes:**
- `SimConfig` mirrors `setupGeneralParameters.m`+`setupForwardParameters.m`: `nt=800`, `order=4`, speeds `{limestone:3600,gold:3240,oil:3240}`, geometry (`y_min=35,y_max=98,x_min=80,x_max=200,divisions_y=2,divisions_x=3`, default `x_src=130,y_src=65,f0=15000,t0=1/f0,scaling=1.0`). Compute `dx_m=dX/100`, `dy_m=dY/100` and `dt` from CFL — never hard-code.
- `resolve_device()`: cuda→mps→cpu; `resolve_dtype(device)`: float32 for mps else float64; explicit override allowed.
- hatchling backend; `[project.optional-dependencies] dev=["pytest"]`. CI: `uv sync` + `uv run pytest -q`.

**Definition of Done:**
- [ ] `uv run python -c "import fwi; from fwi.config import SimConfig, resolve_device; c=SimConfig(); print(c.dt>0, c.dx_m, resolve_device())"` prints `True`, a positive `dx_m`, a valid device.
- [ ] All seven domain `.txt` files exist under `data/domain/`; LICENSE, .gitignore, CITATION.cff, CI workflow present.
- [ ] Verify: `uv run python -c "import pathlib; assert all(pathlib.Path('data/domain',f).exists() for f in ['SmallDomain2D.txt','Domain2D_Fichtner_3pt_oben.txt'])"`

### Task 2: Domain I/O, geometry & per-wavelength source wavelet

**Objective:** Parse the MATLAB domain text format into `alpha2` + ghost mask, place single and multiple sources plus the receiver ring on the grid, and generate Gaussian-derivative wavelets supporting per-source frequency/delay (different wavelengths).

**Files:**
- Create: `src/fwi/domain.py`, `src/fwi/geometry.py`, `src/fwi/wavelet.py`
- Test: `tests/test_io.py`

**Key Decisions / Notes:**
- `read_domain` follows `CreateSyntheticSeismogram.m:13-39` (3 header rows, `nI*nJ*nK` ints, reshape `[nI nJ]`). `build_alpha2` sets `c²` by label then `alpha2[ghost]=0` (`AdjointMethod.m:130`); returns `ghost_mask`. Document the i=Y/j=X axis convention here.
- `geometry.py`: receiver ring (`setupForwardParameters.m:46-74`, 14 sensors default); `make_sources` accepts scalar or vector `x_src/y_src`; `setup_adjoint_sources` maps receivers→sources.
- `wavelet.gaussian_derivative` reproduces `:92-97` incl. `1/(dX·dY)`; accepts array `f0/t0/scaling` → `(n_src, nt)` tensor (one row per source, different wavelengths).
- One unit test class (parsimony) covering all three modules.

**Definition of Done:**
- [ ] `read_domain('data/domain/SmallDomain2D.txt')` → dims `(14,34)`; `build_alpha2` gives α²=0 on the 2-cell ghost border, `3600²` interior.
- [ ] Default-config receiver ring has 14 sensors, all indices inside the non-ghost region; `make_sources` with a 4-vector returns 4 source indices.
- [ ] Wavelet matches the MATLAB formula at 3 sample times (hand-computed ref ≤1e-9); a 2-element `f0` yields 2 distinct rows.
- [ ] Verify: `uv run pytest tests/test_io.py -q`

### Task 3: Differentiable multi-source forward solver + forward example

**Objective:** Implement the pure-torch 2D acoustic solver (FD Laplacian with explicit axis convention, Störmer stepping, multi-source injection, receiver recording, optional bare-Laplacian capture) and a runnable example that regenerates synthetic & observed seismograms and wavefield snapshots.

**Files:**
- Create: `src/fwi/forward.py`, `src/fwi/plotting.py`, `examples/01_forward_modeling.py`
- Test: `tests/test_forward.py`

**Key Decisions / Notes:**
- Vectorized torch slicing stencils on the full grid, ghost mask zeroing borders (autograd-friendly, GPU-fast). X-derivative uses `dx_m` on `dim=1`, Y-derivative `dy_m` on `dim=0` — assert this mapping. Replicate `dt²` scaling and per-source injection loop (`CreateSyntheticSeismogram.m:158-184`).
- `capture_nabla2u=True` stores the **bare** `∇²φ` stack (`d2phi_dx2+d2phi_dy2`, NOT ×α²) for the adjoint; default `False`. Preallocate/stack once (hot path is the `nt` loop).
- `01_forward_modeling.py`: synthetic=`Domain2D_model.txt`, observed=`Domain2D_Fichtner_1pt_oben.txt`; save traces (`.npz`) + snapshots to `outputs/`.
- Functional test only (solver behaviour can't be unit-isolated).

**Definition of Done:**
- [ ] Forward on `SmallDomain2D` is CFL-stable (no NaN/Inf, bounded max|φ|) over all `nt` steps.
- [ ] Symmetric domain + centered source → left-right symmetric wavefield (≤1e-10 float64); a deliberately swapped-axis stencil fails an explicit axis-orientation assertion.
- [ ] `examples/01_forward_modeling.py` writes synthetic+observed trace files and ≥1 snapshot to `outputs/`.
- [ ] Verify: `uv run pytest tests/test_forward.py -q && uv run python examples/01_forward_modeling.py`

### Task 4: L2 misfit & autodiff gradient

**Objective:** Add the L2 waveform misfit and time-reversed adjoint source, then the autodiff gradient example: `alpha2` leaf with `requires_grad`, forward+misfit, `backward()`, read `alpha2.grad` as dJ/dα².

**Files:**
- Create: `src/fwi/misfit.py`, `examples/02_gradient_autodiff.py`
- Test: covered by `tests/test_gradients.py` (Task 5)

**Key Decisions / Notes:**
- `l2_misfit=0.5·Σ(u−u0)²·dt`; `adjoint_source(u,u0)=flip(u−u0, dim=-1)` (no windowing) — used by Task 5.
- `02_gradient_autodiff.py` **defaults to the small/medium grid** (DoD-enforced): a full 104×304×800 autograd graph is ~GB-scale on MPS float32. Ghost-mask `alpha2.grad` before plotting; save kernel image.
- No separate test class here — exercised by Tasks 5–6 (avoid redundant same-path assertions).

**Definition of Done:**
- [ ] `examples/02_gradient_autodiff.py` runs on the small grid by default, produces a finite ghost-masked `alpha2.grad` of domain shape, saves a kernel figure; running with `--grid full` is gated behind an explicit flag with a memory note.
- [ ] Verify: `uv run python examples/02_gradient_autodiff.py`

### Task 5: Manual adjoint-state gradient + agreement test

**Objective:** Hand-code the adjoint-state gradient (re-run forward capturing the bare Laplacian, adjoint solve with time-reversed residual sources at receivers, correlate to the kernel) and prove it equals the autodiff gradient on the small grid — the core correctness contract.

**Files:**
- Create: `src/fwi/adjoint.py`, `examples/03_gradient_adjoint.py`
- Test: `tests/test_gradients.py`

**Key Decisions / Notes:**
- `adjoint_gradient(..., cutoff=0)` **re-runs `forward(..., capture_nabla2u=True)` internally** (does not read the Task 3 `.npz`). Build adjoint sources from `adjoint_source`; adjoint solve → `λ(t)`; `kernel=−Σₜ λ(t)·nabla2u(T−t)·dt`. **Default `cutoff=0` (full integral) = the exact dJ/dα²** that matches autodiff; `cutoff` is a parameter, and the MATLAB `cutoff=100` truncation (`AdjointMethod.m:236-239`) is exposed only for the optional historical kernel image, never in the agreement/Taylor tests or inversion. Ghost-mask the kernel. Pin sign/scaling so it matches `alpha2.grad`.
- `tests/test_gradients.py` = the one functional gradient-correctness class: (a) adjoint≡autodiff relative-L2 ≤1e-5 AND directional-derivative ⟨g,dm⟩ match ≤1e-5 on `SmallDomain2D` float64; (b) Taylor-slope (Task 6).
- `03_gradient_adjoint.py` prints both agreement numbers and saves the adjoint kernel image.

**Definition of Done:**
- [ ] On `SmallDomain2D` (CPU float64): relative-L2(adjoint, autodiff) ≤1e-5 and |⟨g_adj,dm⟩−⟨g_ad,dm⟩|/|⟨g_ad,dm⟩| ≤1e-5.
- [ ] `examples/03_gradient_adjoint.py` prints agreement metrics and saves the kernel figure.
- [ ] Verify: `uv run pytest tests/test_gradients.py -q`

### Task 6: Taylor / hockey-stick gradient test + plot

**Objective:** Implement the Taylor test (first/second-order remainders vs step size) with a non-orthogonality guard and automatic round-off-range detection, producing the hockey-stick log-log plot wired into the adjoint example.

**Files:**
- Create: `src/fwi/gradient_test.py`
- Modify: `src/fwi/plotting.py` (hockey-stick), `examples/03_gradient_adjoint.py` (call it), `tests/test_gradients.py` (slope assertion)

**Key Decisions / Notes:**
- `taylor_test(J_fn, m, dm, grad, steps)`: per `h`, `r1=|J(m+h·dm)−J(m)|` (O(h)), `r2=|J(m+h·dm)−J(m)−h·⟨grad,dm⟩|` (O(h²)). `dm` smooth ghost-masked random (fixed seed). **Guard: assert `|⟨grad,dm⟩| > 1e-12·||grad||·||dm||`** so r1 truly shows slope 1 (orthogonal dm hides the stick).
- Clean-range fit: fit `log r2` vs `log h` over the pre-`argmin(r2)` portion only (excludes the round-off tail); require ≥3 clean steps; assert slope ≈2 (±0.25). Run against the **cutoff=0** adjoint gradient (the exact dJ/dα²); reusable for the autodiff grad too.
- Plot r1 (ref slope 1) and r2 (ref slope 2) vs h log-log; annotate the round-off upturn. Document MPS float32 stick turning up near `h~1e-3…1e-2` vs `~1e-7` CPU float64.

**Definition of Done:**
- [ ] Second-order slope ≈2 (±0.25) over ≥3 clean steps for the adjoint gradient on `SmallDomain2D` float64; non-orthogonality guard passes.
- [ ] `examples/03_gradient_adjoint.py` saves `outputs/hockey_stick.png` showing r1 (≈1) and r2 (≈2).
- [ ] Verify: `uv run pytest tests/test_gradients.py -q && uv run python examples/03_gradient_adjoint.py`

### Task 7: Iterative inversion

**Objective:** Add the iterative FWI loop that reconstructs the perturbation from observed data, with a programmatic anomaly-localization check (not visual-only).

**Files:**
- Create: `src/fwi/inversion.py`, `examples/04_inversion.py`
- Test: `tests/test_inversion.py`

**Key Decisions / Notes:**
- `invert(alpha2_init, u0, cfg, grad_fn, optimizer, n_iter)`: gradient via autodiff or adjoint (default adjoint for the full plate — memory); `torch.optim.LBFGS` default, Adam/SD selectable; ghost-masked updates; misfit history. Start=`Domain2D_model`, observed=`Domain2D_Fichtner_1pt_oben`.
- `04_inversion.py` saves misfit-convergence + reconstructed-α² figures.
- `tests/test_inversion.py` (one functional class): on the small grid (start=`SmallDomain2D`, true=`SmallDomain2D_1pt`), assert final misfit < initial by ≥1 order, AND the argmax of |α̂²−α²_start| falls within a bounding box around the true 2-cell anomaly.

**Definition of Done:**
- [ ] Small-grid inversion reduces misfit ≥1 order of magnitude; recovered-anomaly argmax inside the true-perturbation bounding box.
- [ ] `examples/04_inversion.py` saves convergence + reconstructed-model figures.
- [ ] Verify: `uv run pytest tests/test_inversion.py -q && uv run python examples/04_inversion.py`

### Task 8: Multi-defect / multi-source example + notebooks + README Colab table + OSS polish

**Objective:** Add the more complex variant (multiple defects, multiple combined sources with different wavelengths), the three Colab notebooks, and the README with a Run-in-Colab table — completing the open-source deliverable.

**Files:**
- Create: `examples/05_multidefect_multisource.py`, `notebooks/01_autodiff_fwi.ipynb`, `notebooks/02_adjoint_fwi_hockey_stick.ipynb`, `notebooks/03_multidefect_multisource.ipynb`, `README.md`

**Key Decisions / Notes:**
- `05_multidefect_multisource.py`: true model = `Domain2D_Fichtner_3pt_oben.txt` (3 defects) or `Domain2D_CiE.txt` (complex), start = `Domain2D_model.txt`; multiple combined sources with per-source `f0` (different wavelengths) + `t0` delays (the `setupForwardParameters.m:31-35` commented block). Show combined-source forward, adjoint kernel, and inversion recovering multiple defects.
- Notebooks each open with a Colab setup cell (`pip install -e .` / clone) + `resolve_device()`; `01` runs forward→AD gradient→AD inversion, `02` runs forward→manual adjoint→hockey-stick→adjoint inversion, `03` runs the multi-defect/multi-source variant. A markdown cell documents the MPS float32 hockey-stick caveat.
- README Colab table: rows for the three notebooks with badge links of the form `https://colab.research.google.com/github/OWNER/phd-fwi-pytorch/blob/main/notebooks/<nb>.ipynb`. Set `OWNER` to the GitHub account when the repo is pushed (ask the user if unknown); note this in the README.

**Definition of Done:**
- [ ] `examples/05_multidefect_multisource.py` runs a multi-source combined forward on a multi-defect domain and saves kernel + reconstruction figures localizing ≥2 defects.
- [ ] All three notebooks execute top-to-bottom on CPU; README shows the Run-in-Colab table and the `uv`/Colab/MPS instructions.
- [ ] Verify: `uv run python examples/05_multidefect_multisource.py && for nb in notebooks/01_autodiff_fwi notebooks/02_adjoint_fwi_hockey_stick notebooks/03_multidefect_multisource; do uv run jupyter nbconvert --to notebook --execute "$nb.ipynb" --output "/tmp/_nbcheck_$(basename $nb).ipynb" || exit 1; done`

## Out of Scope

- Porting `SourceLocating`, `SimulationOfAnkleJoint`, or the OO `InversionToolbox` — only `PerturbedPlate` (incl. its multi-defect/multi-source variants) is in scope.
- Bit-exact reproduction of the MATLAB `.mat` seismograms (no MATLAB runtime; correctness via autodiff≡adjoint≡FD).
- Absorbing boundaries / Cerjan tapers (`init_absbound.m` is unused by the plate scripts — Dirichlet ghost layer only).
- `git init` + GitHub push and final Colab `OWNER` substitution — done with the user's go-ahead, not autonomously.
- MATLAB figure/colormap styling, video export, German-labeled plot variants.
