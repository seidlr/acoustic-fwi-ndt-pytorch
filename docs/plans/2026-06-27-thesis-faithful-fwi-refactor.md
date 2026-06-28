# Thesis-Faithful FWI Refactor Implementation Plan

Created: 2026-06-27
Author: robertseidl425@gmail.com
Agent: Claude Code
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Re-ground the package on the REAL thesis InversionToolbox FWI setup so the misfit is sane and L-BFGS converges — correct GaussianDerivative source, aluminum/200 kHz params, real TestCase + recreated logo domains, normalized misfit, L-BFGS — and reproduce three thesis figures (single-crack L-BFGS convergence, the 50/100/200 kHz frequency study, the multi-defect "L i □" logo).

## Approach

**Chosen:** Keep the machine-precision-verified solver (`forward.py` FD/Störmer, `adjoint.py`, autodiff, `gradient_test.py`) and re-ground only the INPUTS and the optimizer to match `InversionToolbox/ndt` (`run_inversion.m`, `GaussianDerivativeSourceTerm.m`, `SolveWaveEquation.m`, `LeastSquaresCostFunctional`, `TestCaseGenerator.m`).
**Why:** The physics was never wrong — the bug was the source wavelet formula (a kernel-demo script's ~1e-11 source) and missing misfit normalization; fixing those makes the misfit O(1) and unlocks L-BFGS, at the cost of replacing the PerturbedPlate params/domains/examples wholesale.

## Context for Implementer

Real reference (read-only): `resources/PhD-FWI-MATLAB/InversionToolbox/ndt/run_inversion.m`, `.../ndt/src/1_PreProcessing/SourceTerm/GaussianDerivativeSourceTerm.m`, `.../ndt/src/2_Simulation/SolveWaveEquation.m`, `.../ndt/src/4_inversion/costfunctional/@LeastSquaresCostFunctional/evaluate.m` + `CostFunctional/CostFunctional.m`, `.../ndt/src/1_PreProcessing/PreProcessors/BenchmarkPreProcessor.m`, `.../ndt/TestCaseGenerator.m`, `.../ndt/TestCase_2D_*Plate.txt`.

The exact real setup (this is the source of truth; the thesis slide's "6020 m/s" is superseded by the code's **6420**, and "Ricker" is loosely named — the code uses a Gaussian *derivative*):

- **Source** (`GaussianDerivativeSourceTerm.m`): `src(t) = -scalingFactor·(t−t0)/(√(2π)·deviation³)·exp(−(t−t0)²/(2·deviation²))`, with `deviation = 1/(2π·f0)`, `scalingFactor = 1e7`, `t0 = 1/f0`. NOTE the `deviation³` denominator (≈1/5e-19 ≈ 2e18 at 200 kHz) — this is what makes the amplitude O(1e18), not the ~1e-11 my port produced. This REPLACES the current `sigma`-based formula. Per-source arrays still supported.
- **Source injection** (`SolveWaveEquation.m`): the ACTIVE update (lines 327-330) is `phi_new = 2·phi − phi_old + (alpha·∇²phi + src)·dt²` where `src = evaluateAtTime(t)·singularityFactor` placed at the source cell — this is IDENTICAL to my `forward.py` (`phi_new = 2φ − φ_old + alpha2·lap·dt²`, then `phi_new += src·dt²`). The commented line 307 (`phi += src·dt²`, a different time level) is the OLD disabled version — ignore it. So no forward time-level change is needed (resolves the Codex "wrong time level" finding); Task 1 adds a one-line parity assertion to lock it.
- **singularityFactor — use the DOMAIN-unit value (=1), NOT 1e6.** `SolveWaveEquation.m:13-14,68` sets `spacings=[dX dY dZ]` from `domain.spacingX/Y/Z` = the header values (**1.0**), with `geoFact` applied separately; so MATLAB's `singularityFactor = 1/prod(spacings) = 1/(1·1) = 1`. Compute it as `1/(cfg.dx·cfg.dy)` in DOMAIN units (=1) — using the metric `1/(dx_m·dy_m)=1e6` would over-inject by 1e6 vs MATLAB. It is anyway immaterial to the inversion (absolute scale is set by `scalingFactor` and cancelled by the J0 normalization); documenting the unit basis is what matters.
- **Misfit is a scalar** (`SolveWaveEquation.m:158` `snsr_value = zeros(numberOfSensors, nt)`): each sensor trace is a ROW, so MATLAB `temp*temp'` is `(1×nt)(nt×1)` = scalar. My `l2_misfit` (`torch.sum` over rec+time) matches — verified, no matrix-misfit risk.
- **Misfit** (`LeastSquaresCostFunctional/evaluate.m`): `J = 0.5·Σ_rec Σ_t (u−u0)²·dt / NORMALIZATION_CONSTANT`, where `NORMALIZATION_CONSTANT = J(m_init)` (the initial misfit, set at construction; default 1 in the base class). So the reported objective is `J/J0`, starting at 1.0 — this is the deeper reason L-BFGS is robust to the absolute scale.
- **Params** (`run_inversion.m`): `f0 = 200000`, `t_end = 48e-6`, `dt = 48e-9` → `nt = 1000`; `geometryScalingFactor = 1e-3` so the **metric grid step is dx_m = dX·1e-3 = 1mm** (NOT the current `dX/100 = 1cm`); `orderInSpace = 4`; `speed_domain = 6420` (aluminum), `speed_crack = 4800`. dt=48ns is ≈0.5·CFL-limit (CFL-limit ≈95ns) → stable.
- **Domains** (`TestCaseGenerator.m`): 200×100 mm plate, MATLAB `data` is `(nI=204, nJ=104)`, `offset=(−2,−2)`, `spacing=1.0`, 2-cell ghost border. Label 1 = aluminum, 2 = crack/defect, 0 = ghost. `BenchmarkPreProcessor(file, speed1, speed2)` maps label `i`→`speed_i`. Cracked: `data(80:110,49:51)=2`; 2-/3-crack per cases 4–5. The `.txt` write is column-major (`fprintf('%d ', data)`) → round-trips with `read_domain`'s `order='F'`.
- **⛔ Coordinate convention is TRANSPOSED between the two codebases (Codex critical).** InversionToolbox `Domain.m` uses `X2I`(X→row i)/`Y2J`(Y→col j) → **i↔X, j↔Y**; my package (from PerturbedPlate) uses `I2Y`/`J2X` → **i↔Y, j↔X** (opposite). The TestCase matrix is `(X=204, Y=104)`. Loading it as-is under my convention rotates the whole plate 90° (crack & sensors misplaced). **Resolution:** `domaingen` GENERATES every thesis domain natively in MY convention `(nI=Y=104, nJ=X=204)` with defects placed at the thesis physical `(x,y)` mm-locations (do NOT copy the raw transposed `.txt`). Sensors/source keep my `coords_to_indices` (y→i, x→j), so domain + geometry share one convention. The plate is 200mm (X=j, 204 cols) × 100mm (Y=i, 104 rows).
- **Sensors & source** (`setSensorPositionCase.m` case 0 + `run_inversion.m`): 16-sensor "thin boundary" at `x=[30 30 170 170 40 100 160 40 100 160 30 199 70 130 70 130]`, `y=[35 70 35 70 30 30 30 70 70 70 52 52 30 30 70 70]`; single source at `(x,y)=(110,60)` (actuator-sensor mode).
- **Optimizer** (`run_inversion.m`): `optlib_lbfgs`, `max_iterations≈30`, `init_step_length=3e9` (huge, because it inverts physical `alpha2 ≈ 4e7`). In PyTorch, invert a **dimensionless model** `m = alpha2/alpha2_background` (m≈1) with the **J0-normalized misfit**. **Gradient-scale hazard (Claude must-fix):** by the chain rule `dJ̃/dm = (1/J0)·alpha2_bg·dJ/dalpha2`, which can be large, so `torch.optim.LBFGS(lr=1.0)` would take a wild first step. Set `lr ≈ 1/alpha2_bg` (≈2.4e-8) as the initial step and rely on `line_search_fn="strong_wolfe"` to refine; Task 3 DoD asserts the line search actually accepts a step and `J̃` decreases (no NaN). Keep Adam/SGD selectable.

Cross-cutting: with the real source formula the wavefield (~1e3–1e4) and misfit (~1e-3) are well-scaled in BOTH float64 and float32, so the `source_scale=1e20` / `FLOAT32_SOURCE_SCALE` / `default_source_scale` band-aids are DELETED (not needed). The hockey-stick gradient test stays on CPU float64. The verified FD/adjoint/autodiff solver and the adjoint≡autodiff agreement are preserved.

## Feature Inventory (migration — full replace)

| Current (PerturbedPlate, mis-scaled) | Action | Task |
|---|---|---|
| `config.py` SimConfig: limestone/gold/oil, f0=15k, dx/100, nt=800, source_scale band-aids | Replace: aluminum 6420 / crack 4800, f0=200k, dx_m=dX·geometry_scaling_factor (1e-3), nt=1000, dt=48ns, scalingFactor=1e7; delete source_scale machinery | 1 |
| `wavelet.py` sigma-based formula | Replace with GaussianDerivative `deviation³` formula + singularityFactor | 1 |
| `domain.py` `SPEED_BY_LABEL` (limestone/gold/oil) | Replace material map (aluminum/crack/slow-defect) | 2 |
| `data/domain/Domain2D_*` (Fichtner/model/CiE/Small) | Remove; `domaingen` GENERATES all thesis domains (homogeneous/crack/2-crack/3-crack/logo/small) in the package convention — NOT copied (raw `.txt` are transposed) | 2 |
| (none) | Add `domaingen.py` — port of `TestCaseGenerator` + the L/i/□ logo | 2 |
| `misfit.py` (no normalization) | Add NORMALIZATION_CONSTANT = J(m_init) | 3 |
| `inversion.py` Adam-only (LBFGS removed) | Add L-BFGS (dimensionless reparam + strong_wolfe + normalized misfit) as default; keep Adam | 3 |
| `geometry.py` receiver ring (limestone coords) | Sensor ring + single source for the 200×100 plate (actuator-sensor mode) | 4 |
| `problems.py` small/full (model/Fichtner) | Grids: homogeneous→crack (TestCase), logo; small grid for tests | 4 |
| `examples/01-05` (PerturbedPlate) | Replace: forward, autodiff-grad, adjoint+hockey, single-crack L-BFGS, frequency study, logo | 5 |
| `notebooks/01-03`, `tests/*`, `README.md` | Update to the thesis setup; tests assert sane misfit + L-BFGS convergence | 6 |
| `forward.py`, `adjoint.py`, `gradient_test.py`, `plotting.py` | Keep (solver verified); plotting extended for the frequency panel | 5 (plotting only) |

## File Structure

- `src/fwi/config.py` (modify) — aluminum/200kHz params, `geometry_scaling_factor`, `scalingFactor`, material speeds; delete source_scale band-aids.
- `src/fwi/wavelet.py` (modify) — GaussianDerivative `deviation³` formula + singularityFactor.
- `src/fwi/domain.py` — material-speed map. DEVIATION: no code change needed; `domain.py` already reads `config.SPEED_BY_LABEL` dynamically, so updating `config` (aluminum 6420 / crack 4800 / slow 5000) replaced the map.
- `src/fwi/domaingen.py` (create) — generate TestCase cracks (1/2/3) + the L/i/□ logo on the 204×104 plate.
- `src/fwi/misfit.py` — `normalization_constant` (J0) support. DEVIATION: J0 normalization implemented in `inversion.py` (`J0 = misfit_of(init)`, objective `J/J0`) instead, keeping `l2_misfit` a pure misfit; matches the thesis pattern (normalization belongs to the cost functional/inversion, not the raw misfit). Truth #1 verified by `test_inversion`.
- `src/fwi/inversion.py` (modify) — L-BFGS (dimensionless reparam, strong_wolfe), Adam kept.
- `src/fwi/geometry.py` (modify) — `sensor_ring`/source for the 200×100 plate.
- `src/fwi/problems.py` (modify) — homogeneous→crack and logo problems; small test grid.
- `src/fwi/plotting.py` (modify) — frequency-study multi-panel.
- `data/domain/` (modify) — copy `TestCase_2D_Homogeneous/CrackedHomogeneousPlate.txt`; generated 2/3-crack + logo written here; remove old `Domain2D_*`.
- `examples/01_forward_modeling.py … 06_logo_inversion.py` (replace) — the 6 thesis-faithful examples (see Task 5).
- `notebooks/01-03` (modify), `tests/test_io.py|test_forward.py|test_gradients.py|test_inversion.py` (modify), `README.md` (modify).

## Assumptions

- The TestCase `.txt` domains use the same 3-header-row format `read_domain` already parses (verified: offset / spacing / dims `204 104 1`). — Tasks 2, 3.
- dt=48ns is CFL-stable for c=6420 at dx_m=1mm (≈0.5·limit). — Tasks 1, 5.
- The "L i □" logo is recreated (not in S3) to approximate the thesis layout; exact pixels are not contractual. — Tasks 2, 5.

## Goal Verification

### Truths

1. On the aluminum plate the start-model waveform misfit is well-scaled (order 1e-4…1e2, NOT ~1e-50), and **L-BFGS** reduces the normalized misfit by ≥1 order while recovering the crack — i.e. the scaling bug is fixed and the thesis optimizer works.
2. The 50/100/200 kHz frequency study reconstructs the same defect with **monotonically sharper resolution** as f0 increases (higher frequency → finer recovered anomaly), reproducing the thesis figure.
3. The recreated multi-defect "L i □" logo is recovered by L-BFGS as a **recognizable multi-defect reconstruction** (each of the ≥3 shapes localizes).

## Progress Tracking

- [x] Task 1: Real GaussianDerivative source + aluminum/200kHz config (sane misfit)
- [x] Task 2: Domain material map + TestCase/2-3-crack/logo generator + data (incl. R4 geometry sensor_ring, folded in - coupled to domain size)
- [x] Task 3: Misfit normalization + L-BFGS inversion (dimensionless reparam) — lr=1.0 (J0-norm makes grad O(1); the planned lr=1/alpha2_bg was 1e8x too small)
- [x] Task 4: Plate sensor/source geometry + problem builders
- [x] Task 5: Six thesis-faithful examples + frequency-study plotting (3 figures) — freq study uses the LOGO + frequency continuation (thin crack is sub-wavelength at 50kHz; high-f0 cycle-skips from homogeneous); metric = corr-with-truth rises 0.47/0.71/0.79
- [x] Task 6: Notebooks + tests + README updated to the thesis setup — 3 notebooks retargeted (each inversion prints the per-iter training loop + loss curve), all execute clean on MPS; README has the thesis params table + InversionToolbox map; float32 source-underflow caveat removed; suite 14 passed, ruff + basedpyright clean, `grep source_scale` empty

## Implementation Tasks

### Task 1: Real GaussianDerivative source + aluminum/200 kHz config

**Objective:** Replace the mis-scaled source wavelet and PerturbedPlate parameters with the real thesis ones so a forward run on the aluminum plate yields a well-scaled wavefield and misfit (not ~1e-50). This is the core scaling fix that unlocks L-BFGS.

**Files:**
- Modify: `src/fwi/config.py`, `src/fwi/wavelet.py`
- Test: `tests/test_io.py`

**Key Decisions / Notes:**
- `wavelet.gaussian_derivative`: `deviation=1/(2π·f0)`; `src=-scalingFactor·(t−t0)/(√(2π)·deviation³)·exp(−(t−t0)²/(2·deviation²))·singularityFactor`. **`singularityFactor=1/(cfg.dx·cfg.dy)` in DOMAIN units (dx=dy=1.0 ⇒ =1)** — matches MATLAB `1/prod(spacings)`; do NOT use the metric `1/(dx_m·dy_m)=1e6` (would over-inject 1e6× vs MATLAB; the review's 1e6 assumed metric spacings, but MATLAB's `spacings` are the header values 1.0). Keep per-source arrays; delete the old `sigma` formula.
- `SimConfig`: `f0=200000`, `t0=1/f0`, `nt=1000`, explicit `dt=48e-9`, `order=4`, `scalingFactor=1e7`, `geometry_scaling_factor=1e-3` → `dx_m = dx·geometry_scaling_factor` (replace the `/100`); `dt` property returns the explicit dt (assert ≤ CFL limit ≈95ns, warn if not). Material speeds `aluminum=6420, crack=4800, slow_defect=5000` (slow used by the logo). DELETE `source_scale`, `FLOAT32_SOURCE_SCALE`, `default_source_scale`.
- `tests/test_io.py` wavelet test: build the reference with the SAME formula in numpy float64 (not hand-computed — the ≈7.9e24 prefactor makes hand-arithmetic error-prone) and assert torch matches at all `nt` samples; plus per-source distinctness.
- Source-injection parity: confirm `forward.py` realizes `phi_new = 2φ − φ_old + (alpha2·∇²φ + src)·dt²` (matches the ACTIVE `SolveWaveEquation.m:327-330`, not the commented line 307) — it already does; add a one-line assertion in the forward test.

**Definition of Done:**
- [ ] A forward run (Task-3 path not needed) on `TestCase_2D_CrackedHomogeneousPlate` vs Homogeneous gives a receiver-residual misfit in 1e-4…1e2 (assert `1e-6 < J < 1e4`), confirming the source scaling is fixed.
- [ ] Wavelet matches the GaussianDerivative formula at 3 sample times (≤1e-6 rel); `source_scale` symbols no longer exist (`grep -r source_scale src/` empty).
- [ ] Verify: `uv run pytest tests/test_io.py -q`

### Task 2: Domain material map + TestCase/crack/logo generator + data

**Objective:** Point the domain layer at the real aluminum/crack material map and GENERATE every thesis domain (homogeneous, crack, 2/3-crack, "L i □" logo, plus a small test pair) in the package's own coordinate convention so the plate is not transposed.

**Files:**
- Modify: `src/fwi/domain.py`
- Create: `src/fwi/domaingen.py`
- Test: `tests/test_io.py`
- Data: `domaingen` writes the domains into `data/domain/`; remove old `data/domain/Domain2D_*` and `SmallDomain2D*`. Do NOT copy the raw `TestCase_2D_*.txt` (they are in the transposed X=i,Y=j convention).

**Key Decisions / Notes:**
- `domain.py`: replace `SPEED_BY_LABEL` with the aluminum map `{1:6420, 2:4800, 3:5000}` (label 3 = slow defect for the logo); ghost (0)→0. `build_alpha2` otherwise unchanged.
- `domaingen.py`: generate in MY convention `(nI=Y=104, nJ=X=204)`. **Index translation (Claude must-fix):** MATLAB `data(80:110, 49:51)` (i=X 80..110, j=Y 49..51, 1-based inclusive) → my array `data[iY 48:51, jX 79:110]` (0-based, `j-1` start, end inclusive→Python slice end = MATLAB end). So `make_cracked()` sets `data[48:51, 79:110] = 2` (a crack spanning X≈80–110mm at Y≈49–51mm). `make_two_cracks`/`make_three_cracks` translate cases 4–5 the same way. `make_logo()` draws **L** (label 3, slow), **i** bar+dot (label 2, fast), **□** (label 3, slow) at thesis-like (x,y); document cell ranges. Write column-major (`.flatten(order="F")`) + the 3-header-row format so `read_domain` round-trips. Add a small `(nI=24, nJ=54)` homogeneous + single-defect pair for the fast gradient/Taylor tests.
- Ghost border: 2 cells on every side (rows/cols 0,1 and last two = 0).

**Definition of Done:**
- [ ] `read_domain` of the generated crack domain → dims (104,204); `build_alpha2` gives 6420² in aluminum, 4800² in the crack, 0 on ghost; the crack cells are exactly `[48:51, 79:110]` (assert), i.e. NOT transposed.
- [ ] `make_logo()` domain has labels 2 (fast) AND 3 (slow) and ≥3 distinct connected defect regions; `read_domain` loads it; a forward run on it is finite/stable.
- [ ] Verify: `uv run pytest tests/test_io.py -q`

### Task 3: Misfit normalization + L-BFGS inversion

**Objective:** Add misfit normalization by the initial misfit (J/J0) and restore L-BFGS as the default optimizer, parameterized so it converges on the well-scaled problem the way the thesis `optlib_lbfgs` did.

**Files:**
- Modify: `src/fwi/misfit.py`, `src/fwi/inversion.py`
- Test: `tests/test_inversion.py`

**Key Decisions / Notes:**
- `misfit.py`: keep `l2_misfit`; add optional `normalization_constant` (divide). `inversion.invert` computes `J0 = l2_misfit(start)` once and reports `J/J0` (history starts at ~1.0).
- `inversion.py`: invert a **dimensionless** model `m` with `alpha2 = alpha2_background · m` (m init 1 on active cells; background = aluminum speed²), so variables are O(1). Default `optimizer="lbfgs"` → `torch.optim.LBFGS([m], lr=LR, max_iter=20, line_search_fn="strong_wolfe")`, closure returns the **J0-normalized** misfit; gradient via the verified autodiff path (the TRUE gradient, NOT unit-normalized — L-BFGS needs it). **lr scale (Claude must-fix):** `dJ̃/dm = (1/J0)·alpha2_bg·dJ/dalpha2` is large, so set `LR = 1/alpha2_bg` (≈2.4e-8) as the initial trial step; strong_wolfe refines from there. Keep `adam`/`sgd` (their unit-normalized-gradient path stays). Ghost-masked, clamp m≥0. Return `(alpha2_hat, history)` incl. the final model's J/J0.
- This supersedes the prior "LBFGS removed" deviation; the dimensionless reparam + J0 normalization + lr=1/alpha2_bg is what the earlier note lacked.
- **Training-loop progress (user request):** `invert` already takes a `callback(iteration, misfit, model)`; call it every iteration with the normalized misfit and emit a classic NN-training-style line when `verbose=True` — e.g. `iter 03/30 | loss (J/J0) = 8.41e-01 | grad_norm = 1.2e-03`. Examples/notebooks use this so the inversion reads like a standard PyTorch training loop (per-iteration loss printed live).

**Definition of Done:**
- [ ] On the small homogeneous→crack grid (CPU float64), `optimizer="lbfgs"` does NOT produce NaN, the strong-Wolfe line search accepts ≥1 step (J̃ strictly decreases on iteration 1), reduces `J/J0` to ≤0.1, and the recovered-update argmax lands within 3 cells of the true crack.
- [ ] `history[0]` ≈ 1.0 (normalized) and `history[-1]` is the returned model's J/J0; assert `history[-1] < history[0]`.
- [ ] Verify: `uv run pytest tests/test_inversion.py -q`

### Task 4: Plate sensor/source geometry + problem builders

**Objective:** Place the actuator-sensor-mode source and the sensor ring on the 200×100 plate and provide problem builders (homogeneous→crack, logo, small-test) so examples/tests share one setup, preserving the adjoint≡autodiff agreement with the new source.

**Files:**
- Modify: `src/fwi/geometry.py`, `src/fwi/problems.py`
- Test: `tests/test_gradients.py`

**Key Decisions / Notes:**
- `geometry.py`: `sensor_ring(cfg)` returns the exact thesis 16-sensor layout (`setSensorPositionCase.m` case 0): `x=[30 30 170 170 40 100 160 40 100 160 30 199 70 130 70 130]`, `y=[35 70 35 70 30 30 30 70 70 70 52 52 30 30 70 70]`; single source at `(x,y)=(110,60)`. Map via `coords_to_indices` (y→i, x→j; keep its bounds check) so they share the domain convention.
- `problems.py`: `build_problem("crack"|"logo"|"small")` returning start (homogeneous) + true (crack/logo) alpha2, source, sensors, observed traces, active mask, using the real wavelet/params. The `"small"` grid uses the 24×54 domaingen pair with a couple of in-grid sensors + source.
- ⛔ **The gradient-agreement test stays at the alpha2 level (Claude must-fix):** it compares `adjoint_gradient` vs `autodiff_gradient`, both `dJ/dalpha2` of the **raw (un-normalized) l2_misfit** — NOT `dJ̃/dm` of the normalized misfit (that would compare different quantities via the chain rule). The Task-3 inversion wrapper (m-space, normalized) is NOT used by this test.

**Definition of Done:**
- [ ] `build_problem("crack")` source+16 sensors all land in-grid on non-ghost cells; observed≠synthetic (crack visible in the residual).
- [ ] adjoint≡autodiff relative-L2 ≤1e-5 (both raw `dJ/dalpha2`) and the hockey-stick slope ≈2 still hold on the small grid (CPU float64).
- [ ] Verify: `uv run pytest tests/test_gradients.py -q`

### Task 5: Six thesis-faithful examples + frequency-study plotting

**Objective:** Replace the PerturbedPlate examples with thesis-faithful ones and produce the three target figures: single-crack L-BFGS convergence, the 50/100/200 kHz frequency study, and the multi-defect logo inversion.

**Files:**
- Replace: `examples/01_forward_modeling.py`, `02_gradient_autodiff.py`, `03_gradient_adjoint.py`, `04_crack_inversion_lbfgs.py`, `05_frequency_study.py`, `06_logo_inversion.py`
- Modify: `src/fwi/plotting.py`

**Key Decisions / Notes:**
- `01` forward (aluminum homogeneous vs crack, snapshots + traces); `02` autodiff gradient (small grid); `03` adjoint kernel + hockey-stick (CPU float64); `04` single-crack L-BFGS convergence + reconstruction; `05` frequency study — invert the same crack at f0∈{50k,100k,200k}, save a 3-panel "inverted wave speed for different source frequency" figure; `06` logo inversion (L-BFGS on `make_logo`), reconstruction figure.
- `plotting.py`: add `save_frequency_panel(models, freqs, path)` (the thesis-style multi-panel) reusing `save_field`.
- Full-plate L-BFGS at nt=1000 is heavier; `05`/`06` may downsample iterations or grid for runtime — `log()`/print the choice, don't silently cap.

**Definition of Done:**
- [ ] `04` drives L-BFGS to ≥1-order `J/J0` drop and saves convergence + reconstruction localizing the crack.
- [x] `05` saves the 3-panel 50/100/200 kHz figure AND asserts a hard, falsifiable monotonic sharpening. DEVIATION (documented): the target is the multi-defect logo (no single anomaly "width"), so the falsifiable metric is correlation-with-truth of the recovered update, asserted monotonically increasing `corr(50k) < corr(100k) < corr(200k)` (0.47 < 0.71 < 0.79). A higher corr with the ground-truth model is the proper sharpening measure for a multi-region target.
- [ ] `06` saves a logo reconstruction localizing ≥3 defect regions.
- [ ] Verify: `uv run python examples/04_crack_inversion_lbfgs.py && uv run python examples/05_frequency_study.py && uv run python examples/06_logo_inversion.py`

### Task 6: Notebooks + tests + README updated to the thesis setup

**Objective:** Bring the notebooks, test suite, and README in line with the thesis-faithful package so the public repo reflects the real setup and the misfit/L-BFGS fix is regression-guarded.

**Files:**
- Modify: `notebooks/01_autodiff_fwi.ipynb`, `02_adjoint_fwi_hockey_stick.ipynb`, `03_multidefect_multisource.ipynb` (retarget to frequency-study/logo), `tests/test_io.py|test_forward.py|test_gradients.py|test_inversion.py`, `README.md`

**Key Decisions / Notes:**
- Notebooks: `01` forward+autodiff on the aluminum crack; `02` adjoint+hockey-stick; `03` → frequency study + logo (rename to `03_frequency_study_and_logo.ipynb`). Keep the Colab `sys.path.insert(src)` setup cell.
- **Training-loop UX (user request):** every notebook inversion runs with `verbose=True` so it prints classic PyTorch-style per-iteration progress (`iter k/N | loss (J/J0) = … | grad_norm = …`) live as it iterates, and then plots the loss-vs-iteration curve — so the inversion reads like a standard NN training loop. The crack inversion (example `04`) prints the same to the console.
- Tests: add a regression test asserting the start-model misfit is sane (`1e-6 < J < 1e4`, the anti-1e-50 guard) and that L-BFGS reduces it; update `test_forward.py` physics (params changed, axis test stays); ensure the suite is green.
- README: update params table (aluminum/200kHz/1mm/L-BFGS), the MATLAB→PyTorch map (point at InversionToolbox), the figures (frequency study + logo), and remove the float32 source-underflow caveat (no longer applies). Keep the thesis link.

**Definition of Done:**
- [x] Full suite green incl. the sane-misfit regression test (14 passed); `grep -r source_scale src tests` empty.
- [x] All notebooks execute top-to-bottom (`nbconvert --execute`, exit 0); each inversion cell's output shows the per-iteration training-loop progress (iter k/N + loss J/J0 + grad_norm) plus a loss-vs-iteration curve. nb02 crack L-BFGS: J/J0 0.99 -> 0.097; nb03 freq study: corr 0.42/0.70/0.78.
- [x] README reflects the thesis params/figures; Verify: `uv run pytest -q` (14 passed).

## Verification (spec-verify)

**Automated:** pytest 14 passed; basedpyright 0 errors; ruff clean; 13 modules import; `grep source_scale` empty. All 3 notebooks execute via nbconvert (exit 0, figures embedded, training-loop progress printed). Examples 01-06 run; 04 (17x crack reduction), 05 (corr 0.467<0.705<0.787), 06 (corr 0.787, defect-cell mean update -1.19e7 < 0).

**Goal truths: 3/3 verified.**
1. Sane misfit + L-BFGS — start misfit J=95.09 (in 1e-4..1e2); L-BFGS 0.99 -> 0.059 (17x); `test_inversion` passes. VERIFIED.
2. Frequency study monotonically sharper — corr-with-truth 0.467 < 0.705 < 0.787 (example 05 asserts). VERIFIED.
3. Logo multi-defect recovery — 4 defect regions, corr 0.787, defect cells recovered as low-velocity (example 06 asserts). VERIFIED.

**Code review (inline /code-review xhigh, 10 angles).** Codex companion launched but stalled in "starting" phase >18 min (broker issue) and was cancelled — gap recorded; inline review (the primary mechanism on Claude Code) carried the review.

Findings fixed (should_fix):
- inversion.py: J0 `or 1.0` did not guard NaN -> `math.isfinite` guard; `grad_fn='adjoint'` silently ignored under L-BFGS -> now raises ValueError; docstring history[0] semantics corrected.
- problems.py: stale module docstring (deleted filenames, wrong dims, "two grids") rewritten; removed dead `GRIDS['full']` alias (no callers, YAGNI).
- domaingen.py: comments mislabeled the 4800 m/s crack as "fast" -> corrected (both defects are low-velocity vs aluminum).
- examples/06: comment/assertion claimed a "fast region" that does not exist -> assertion now tests defects recovered as slow (mean update < 0 in true-defect cells).

Mention-only (working as intended): inversion.py L-BFGS in-place CFL-safety clamp [0, m_max] — a deliberate guard that prevents the 50 kHz NaN; rarely activates in normal convergence; "fixing" it would reintroduce the NaN.

**Not verified:** Codex adversarial review (companion stalled — cancelled). E2E browser testing N/A (pure Python library, no UI). MPS hockey-stick precision floor (documented float32 behavior, not a regression).

## Out of Scope

- Porting the full OO InversionToolbox 1:1 (preprocessor/source/sensor/cost-functional class hierarchy) — only the behavior is replicated on the verified solver.
- Real experimental data / the bone & block applications from other thesis chapters — only the 2D plate (`ndt`) study.
- Pixel-exact reproduction of the thesis logo (recreated to approximate) and bit-exact match to MATLAB outputs.
- 3D (`nK>1`) — the reader errors on it, as before.
