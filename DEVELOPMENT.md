# Development notes

EGM.jl was translated from the MATLAB package [EGM-v2](https://github.com/yulanvanoppen/EGM-v2) (commit `3c83609`) in phases. Every component is tested against reference ("golden") files exported from the MATLAB implementation. This document records how the translation maps onto the MATLAB code, the design decisions, and how to reproduce the comparisons.

&nbsp;

## Source layout

| Julia file | MATLAB counterpart | Content |
|---|---|---|
| `src/model.jl` | `System.m`, auxiliary model files | `ODEModel`, g/h decomposition, Jacobians, integration, variational and sensitivity systems, `LinearInterpolant` |
| `src/bspline.jl` | `BSpline.m`, knot handling in `EGM.m`, `Smoother.place_knots` | B-spline bases, knot normalization and placement, `SplineBases` |
| `src/smoother.jl` | `Smoother.m`, `Optimization.noise_parameters`, `tryinv.m` | FWLS spline smoothing with the noise model σ² + τ²ŷ² |
| `src/kalman.jl` | `Filter.m`, `EGM.augment_covariances` | EKF/RTS smoother with joint covariances |
| `src/gm.jl` | `GM.m`, `Optimization.QPGLS`, `nearestSPD.m` | Gradient-matching step, delta-method uncertainties |
| `src/estimate.jl` | `EGM.estimate`, `EGM.init_ICmean` | `egm`, `egm_inference`, `egm_cells`, `ic_candidates` |
| `src/tm.jl` | `TM.m`, `EGM.estimate_TM` | Trajectory matching |
| `src/generate.jl` | `Generator.m` | Synthetic data |
| `src/iqm.jl` | IQM Tools parsing in `System.m` | `load_iqm`, `parse_iqm` |

&nbsp;

## Design decisions

- **Models are plain Julia functions** `f(x, β, t)`. Because f is affine in β, the decomposition is exact without symbolic algebra: h(x) = f(x, 0) and gₚ(x) = f(x, eₚ) − h(x). Jacobians come from ForwardDiff, and the variational equation dΦ/dt = J(x(t))Φ is generated automatically, so no auxiliary model files are needed.
- **Process noise is L(x) = M·diag(σx)** with a user-supplied M (default I). In the default mode, M enters only the process-noise covariance Q, and the prediction and smoothing steps use the true propagator. `mode = :compat` reproduces EGM-v2, where the variational system starts at Φ = M, so M also enters the covariance propagation. For M = I both modes are identical.
- **No model-name special cases.** The branches in `init_ICmean` and `System.auxiliary` became the options `ic_scan`, `ic_ties`, `ic_constraint` and `M`.
- **Q keeps the MATLAB form** ∫ Φ(s, t₋) L Lᵀ Φ(s, t₋)ᵀ ds. It agrees with the textbook form ∫ Φ(t₊, s) L Lᵀ Φ(t₊, s)ᵀ ds to second order in Δt, and exactly for linear time-invariant systems.
- **Optimizers are replaced by exact or more accurate methods:**

  | MATLAB | Julia |
  |---|---|
  | `lsqnonneg` (two columns) | `nnls2`: exact, by enumerating the four supports |
  | `fmincon` for the noise likelihood | `noise_mle`: projected Newton. It reaches bounds exactly, where `fmincon` stops slightly inside the feasible region, and its likelihood is always at least as good. |
  | `mpcActiveSetSolver` | `box_qp`: primal active-set method, exact for strictly convex problems |
  | `fmincon` in log space for trajectory matching | `lm_bounded`: bounded Levenberg–Marquardt with exact forward sensitivities |

- **Trajectory matching always refits with the estimated noise model** (`min_iter = 2`). Because the initial-condition search in Julia already optimizes tightly, the first refit barely moves β. With the MATLAB stopping rule, Julia TM therefore stopped after one iteration, still using the initial weights. MATLAB's loose search (`StepTolerance` 1e-4, at most 50 iterations) usually avoids this by accident.

- **Performance.** The variational and sensitivity systems use an in-place right-hand side with a reusable ForwardDiff configuration. A generic out-of-place path is kept for non-Float64 states, which arise in implicit solvers with automatic differentiation.

&nbsp;

## Findings from the translation

- **The covariance bookkeeping of `Filter.m` is exact in the linear-Gaussian case.** For a linear-Gaussian model, the recursions in `update_covariance_EKF/EKS` reproduce the exact joint covariance of smoothed errors and measurement errors, compared with brute-force Gaussian conditioning (`test/test_phase4.jl`).
- **IQM Tools' default absolute tolerance limits the MATLAB trajectories.** IQM receives only `reltol`, so even at reltol 1e-10 the CVODES trajectories are accurate to about 1e-5. This sets the integration tolerances of the Phase 1 tests.
- **`fmincon` leaves noise parameters slightly inside their bounds.** On the reference cases, one of σ², τ² has its optimum at 0, and `fmincon` stops slightly inside. Smoothed values are unaffected (about 1e-5), but variances differ by up to about 2%.

&nbsp;

## Reference files and tests

### Generating the reference files (MATLAB)

1. Clone EGM-v2 at commit `3c83609` and make sure IQM Tools works.
2. Set `REPO` in `matlab/export_reference.m` and run it.
3. It writes `test/reference/{HS_P, HS_P_sub, GLV_X3, STAT5, STAT5_patched}.mat`. The array layouts are documented in `test/reference/README.md`.

The export replays `EGM.estimate` step by step through the public methods of the MATLAB classes, without modifying EGM-v2. For each unpatched case it checks the replay against `EGM.estimate` itself. `STAT5_patched` uses `matlab/FilterPatched.m` (M only in the noise) as the reference for the Julia default mode.

### Running the tests

```julia
pkg> activate .
pkg> test
```

Comparisons with MATLAB are skipped (with a warning) when the reference files are absent. Set `ENV["EGM_REFDIR"]` to use another folder.

| Test file | Content |
|---|---|
| `test_phase0.jl` | Completeness and consistency of the reference files |
| `test_phase1.jl` | Model layer |
| `test_phase2.jl` | B-splines and knots |
| `test_phase3.jl` | Spline smoother |
| `test_phase4.jl` | Kalman filter/smoother |
| `test_phase5.jl` | Gradient-matching step |
| `test_phase6.jl` | Complete estimation |
| `test_phase7.jl` | Sensitivities, trajectory matching, uncertainties, data generator, multiple cells, IQM parser |

Each module is tested on MATLAB's inputs to that module, not on upstream Julia output, so a discrepancy points to one component. Pure linear algebra (B-splines, Savitzky–Golay, covariance bookkeeping, GLS) agrees to near machine precision. Steps involving ODE solvers or iterative optimizers agree to roughly their tolerances.

### Benchmarks against the MATLAB simulation studies

`benchmarks/` ports `accuracy_*.m` and `test_*.m`. `matlab/export_datasets.m` exports the datasets those MATLAB scripts generate, so the Julia ports can run on identical data. Results are written as JSON and CSV; see `benchmarks/README.md` for the format.

### Diagnostic reports

`test/diagnostics/phaseN_report.jl` prints every MATLAB comparison of phase N with relative errors and the locations of the largest deviations. To run one:

```
julia --project=. test/diagnostics/phase6_report.jl
```

The reports need MAT.jl, so add it once to your default environment (`] add MAT`).
