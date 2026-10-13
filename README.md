# EGM.jl: Extended Gradient Matching
_Julia package for parameter inference of ODE models with uncertain initial conditions and unobserved states, using the extended gradient matching (EGM) method presented in Chapter 4 of [1]_

&nbsp;

## Quickstart
Install Julia 1.10 or later, then add the package:
```julia
julia> ]
pkg> add https://github.com/yulanvanoppen/EGM-julia
```

Run `examples/example.jl` (environment: `examples/`) to simulate measurements of one state of the Higgins–Selkov oscillator and recover its parameters and the initial condition of the unobserved state with EGM and with trajectory matching.

The package is a Julia translation of the MATLAB® package [EGM-v2](https://github.com/yulanvanoppen/EGM-v2). It needs no MATLAB, IQM Tools or Monolix installation. IQM Tools model files can still be read with `load_iqm`.

`ODEModel` defines an ODE system that is linear in its unknown parameters, either as a Julia function or from an IQM Tools model file.

`egm` estimates the parameters from measurements of some or all states.

`tm` estimates the parameters by trajectory matching, for comparison.

`generate_data` simulates measurements from a model.

`smoothing_app` smooths measurements interactively, also on its own, without estimation (see [SMOOTHING_APP.md](SMOOTHING_APP.md)).

&nbsp;


## Method
EGM infers the parameters `β` of an ODE system that is linear in its parameters,
```
    dx(t)/dt = g(x(t), t) β + h(x(t), t),        x(t_1) = x_0 (uncertain)
```
from noisy measurements `y_j` of a subset of the states at time points `t_j`. Gradient matching [2] avoids repeated numerical integration. It estimates the states and their derivatives first, and then obtains `β` by generalized least squares on the equation above. EGM extends this to partially observed systems:

1. **Smoothing.** The observed states are smoothed with B-splines. The measurement noise variances `σ² + τ² y²` (additive and multiplicative) are estimated jointly by feasible weighted least squares.
2. **Kalman smoothing.** The hidden states are reconstructed by extended Kalman smoothing, using the current parameter estimate and the smoothed observed states. The joint covariance of all state estimates and measurement errors is tracked along the way.
3. **Gradient matching.** `β` is updated by generalized least squares, weighted by the covariance of the gradient-matching residuals and subject to box constraints and an optional Gaussian prior. Steps 2 and 3 alternate in an approximate expectation–maximization (E-M) loop until `β` converges.

The prior mean of the uncertain initial conditions is chosen beforehand by a grid search: the candidate with the smallest weighted residual sum of squares after one iteration is kept.

&nbsp;


## Usage
### `egm()` function
`result = egm(model, t, traces, observed, β_init; ...)` estimates the parameters of `model` from the measurements `traces`, starting from the guess `β_init`. Here `traces` is a `T×L` matrix with measurements at the `T` time points `t` of the `L` states with indices `observed`. The estimate is `result.β`. The other fields of `result` are described below.

`result = egm(_; x_init = x, ...)` uses `x` as the guess for the initial conditions. By default, `x_init = model.x0`. Observed initial conditions are replaced by the smoothed measurements, and the hidden ones are scanned around `x`.

`result = egm(_; knots = knots, ...)` smooths the measurements with B-splines with the given knots (in time units), either one vector for all observed states or one vector per state. By default, knots are placed automatically.

`results = egm_cells(model, t, traces, observed, β_init; ...)` estimates cell-specific parameters for each trajectory in the `T×L×N` array `traces`, in parallel threads (start Julia with `--threads=auto`).

`result = egm_inference(model, smoothing, traces, observed, β_init; ...)` skips the smoothing stage and starts from an existing `SmoothingResult` (see the components below).

&nbsp;

#### Keyword arguments
`x_init` - Initial condition guess  
`model.x0 (default) | numeric vector`

`knots` - Knot locations for B-spline smoothing (time units)  
`nothing (default) | numeric vector | vector of vectors`  
(one vector for all observed states, or one vector per observed state)

`autoknots` - Use the automatic knot placement heuristic when `knots` is not given  
`true (default) | false`  
(if false, `round((T−1)/2) + 1` equidistant knots are used)

`t_gm` - Time points used for Kalman smoothing and gradient matching  
`t (default) | numeric vector`  
(must be a subset of `t`)

`lb`, `ub` - Parameter bounds  
`0.25 model.β0`, `4 model.β0 + 1e-4 mean(model.β0) (default) | numeric vectors`  
(must bracket `model.β0`)

`prior_mean`, `prior_prec` - Gaussian parameter prior  
`0, 0 (default) | numeric vector, P×P matrix`  
(alternatively `prior_sd`, a vector of standard deviations, sets `prior_prec = inv(Diagonal(prior_sd.^2))`)

`positive` - Force positive smoothed and predicted states  
`true (default) | false`

`σ` - Process noise scale of the extended Kalman smoother  
`0.05 (default) | positive scalar`  
(the process noise of state `k` has standard deviation proportional to `σ x_k`; small values trust the model, large values trust the data)

`P_CV` - Coefficient of variation of the initial condition prior  
`0.25 (default) | positive scalar`

`mode` - Use of the noise mixing matrix `model.M`  
`:default (default) | :compat`  
(`:default` uses `M` only in the process noise; `:compat` reproduces EGM-v2, where `M` also enters the covariance propagation)

`ic_scan` - States whose initial conditions are scanned  
`hidden states (default) | integer vector`

`ic_multipliers` - Multiples of `x_init` scanned for each state in `ic_scan`  
`4 .^ (-1:0.5:1) (default) | numeric vector`  
(all combinations are tried, so the number of candidates grows exponentially with `length(ic_scan)`)

`ic_ties` - Initial conditions set equal to others  
`[] (default) | vector of pairs`  
(e.g. `[2 => 1]` sets the initial condition of state 2 equal to that of state 1)

`ic_constraint` - Function applied to each candidate initial condition vector  
`identity (default) | function`  
(e.g. `x -> (x[4] = min(x[3], x[4]); x)` enforces `x₄(0) ≤ x₃(0)`)

`interactive` - Choose the smoothing knots in the interactive smoothing app (requires `using GLMakie`)  
`false (default) | true`

`niter_sm`, `tol_sm` - Maximum number of smoothing iterations and convergence tolerance  
`20, 1e-3 (default) | positive integer, positive scalar`

`niter_gm`, `tol_gm` - Maximum number of E-M iterations and convergence tolerance on the relative change in `β`  
`20, 1e-4 (default) | positive integer, positive scalar`

`alg` - ODE solver  
`Tsit5() (default) | any SciML ODE algorithm`  
(for stiff models, e.g. `Rodas5P()` from OrdinaryDiffEqRosenbrock or `CVODE_BDF()` from Sundials)

`verbose` - Print progress  
`false (default) | true`

&nbsp;

#### Output
`result` is an `EGMResult` with the following fields:

| Field | Content |
|---|---|
| `β` | Parameter estimate |
| `β_cov` | Approximate covariance of `β` (delta method as in [2]; ignores bounds and prior) |
| `β_history`, `SS_history` | Parameter estimates and weighted residual sums of squares per E-M iteration |
| `iterations` | Number of E-M iterations |
| `x_init` | Selected prior mean of the initial conditions |
| `ic_candidates`, `ic_betas`, `ic_errors`, `ic_index` | Initial-condition search: candidates, their one-step estimates and residual sums of squares, and the selected candidate |
| `init` | Smoothed initial state used for the final fit |
| `fitted`, `dfitted` | Model fit and its right-hand side on `t` (all states) |
| `fitted_fine`, `dfitted_fine`, `t_fine` | The same on an 81-point grid, for plotting |
| `smoothing` | `SmoothingResult`: B-spline smoothing of the observed states, noise model, variances |
| `kalman` | `KalmanResult` of the last iteration: smoothed means `xbar`, covariances `P`, standard deviations `sd` |
| `gm` | `GMStepResult` of the last iteration: state and derivative estimates `X`, `dX`, residual covariance `V` |
| `time` | Seconds spent on smoothing and on inference |

&nbsp;

### Interactive smoothing app
The smoothing app has its own page: [SMOOTHING_APP.md](SMOOTHING_APP.md). In brief:

`out = smoothing_app(t, traces; state_names = names, ...)` opens a window for choosing the B-spline knots by hand, with immediate visual feedback. `traces` is `T×L` (one trajectory) or `T×L×N` (`N` cells). The knots start from `knots` (time units) or from the automatic knot placement. Julia waits until you press Finish or close the window. Then `out.knots` (normalized, per state), `out.knots_time` (time units, usable as `egm(_; knots = out.knots_time)`) and `out.smoothing` (the smoothing result) are returned. `out.action` records how the app was ended (`:finish`, `:estimate` or `:closed`).

`egm(_; interactive = true)` opens the app as part of the estimation. The app then has two buttons: [Finish and run estimation] continues with EGM, and [Finish without estimation] stops after smoothing. In the latter case, and when the window is closed, `egm` returns the app's output instead of an `EGMResult`.

The app needs GLMakie, which is not installed with EGM.jl. Add it to your environment (`] add GLMakie`; the first installation takes a few minutes) and load it with `using GLMakie`. The folder `examples/` has an environment that includes it; see `examples/smoothing_app.jl`.

The window contains the following controls:
- **[State(s)] menu:** Edit all states together ([All]), or one state.
- **[Automatic knot placement] button:** Replace the knots of the selected state(s) using the knot placement heuristic.
- **[#Knots] slider:** Replace the knots of the selected state(s) by equidistant knots.
- **[Manual knot selector] plot:** Shows the B-spline basis and the knots (dashed lines) of the selected state, or of the first state if [All] is selected. Left click adds a knot, right click removes the nearest one. With [All], the result is applied to all states.
- **[#Cells] slider:** Number of trajectories shown (several cells only). The most typical cell is shown first, followed by the others from most to least atypical.
- **[Smooth] button and [Live] toggle:** Smooth with the current knots, or smooth after every change.
- **[Finish] button:** Return the knots and smoothing result. States changed without smoothing are smoothed first. From `egm`, there are two buttons instead: [Finish and run estimation] and [Finish without estimation].

The right side shows the measurements (solid) and smoothed curves (dashed) of the selected state(s), one colour per cell.

&nbsp;

### Defining models
#### Julia functions
`model = ODEModel(f; states, parameters, x0, β0)` defines the system `dx/dt = f(x, β, t)` from a Julia function `f` that returns a vector. `f` must be linear in the parameters `β` (affine, to be precise); this is checked when the model is created. Fixed parameters and inputs are simply captured by `f`. Write `f` without type annotations, so that it can be differentiated automatically:
```julia
c = 0.95                                                      # fixed parameter
f(x, β, t) = [β[1] - c * x[1] * x[2]^2,                       # dS/dt = r − c S P²
              c * x[1] * x[2]^2 - β[2] * x[2]]                # dP/dt = c S P² − k P
model = ODEModel(f; states = ["S", "P"], parameters = ["r", "k"],
                 x0 = [1.0, 1.0], β0 = [1.0, 1.0])
```

The decomposition into `g` and `h`, and all derivatives needed by EGM, are computed automatically.

`ODEModel(f; ..., tstops = ts)` makes the ODE solver step exactly onto the times `ts`, where `f` is not smooth in `t`. A piecewise linear input is defined with `u = LinearInterpolant(ts, ys)`, which is held constant outside `[ts[1], ts[end]]`. Pass `tstops = breakpoints(u)` to step onto its breakpoints.

`ODEModel(f; ..., M = M)` sets the `K×K` noise mixing matrix of the Kalman smoother. The process noise is then `M diag(σx)`, i.e. independent noise on linear combinations of the states. This is useful when the states are lumped quantities, such as total protein amounts in the STAT5 model.

#### IQM Tools model files
`model = load_iqm(model_file; fixed = names)` reads an IQM Tools model file, as used by EGM-v2 and GMGTS. The parameters in `names` are fixed at their nominal values, or at `fixed_values` if given. The remaining parameters, in file order, are the free parameters `β`. Model variables, reactions and functions are supported. `interp1IQM` inputs with literal breakpoints become `LinearInterpolant`s, and their breakpoints become `tstops`. Model events are not supported. The generated model is slightly slower than a hand-written function; `model = compile_iqm(load_iqm(...))` removes the overhead for models used in later top-level statements (see `?compile_iqm`). A model file looks as follows:
```
********** MODEL NAME
Simple model
********** MODEL STATES
d/dt(A) = k1 - k2*A
d/dt(B) = k2*A
A(0) = 1
B(0) = 0
********** MODEL PARAMETERS
k1 = 0.1
k2 = 0.5
```

&nbsp;

### `tm()` trajectory matching
`result = tm(model, t, traces, observed, β_init; ...)` estimates the parameters by fitting the integrated model directly to the measurements (weighted least squares), alternating with re-estimation of the noise model. By default, the initial conditions are estimated too (`estimate_x0 = true`), after an initial-condition search like that of `egm`. It accepts `x_init`, `lb`, `ub`, `positive`, `niter`, `tol`, `min_iter`, the `ic_*` options, `reltol` and `alg`. By default (`min_iter = 2`), the parameters are always refitted at least once with the estimated noise model. The result contains `β`, `x0`, the noise model and the fits.

&nbsp;

### `generate_data()` synthetic measurements
`data = generate_data(model, t; ...)` integrates `model` and perturbs the observed states with noise `y = ε_add + x (1 + ε_mul)`. It returns `traces` (`T×L×N`), `truth` (`T×K×N`), the cell parameters `β` (`N×P`) and the noise variances `noisevar`.

`N` - Number of cells  
`1 (default) | positive integer`

`β`, `x0` - Parameters (mean, for multiple cells) and initial conditions  
`model.β0, model.x0 (default) | numeric vectors`

`D_mult`, `D` - Cell-to-cell parameter variability: common coefficient of variation, or covariance matrix  
`0, nothing (default) | non-negative scalar, P×P matrix`

`lognormal` - Draw log-normal cell parameters with the same mean and covariance  
`false (default) | true`

`observed` - Observed states  
`1:K (default) | integer vector`

`error_std`, `error_const` - Multiplicative and additive noise standard deviations  
`0.05, 0 (default) | non-negative scalars`

`rng` - Random number generator  
`Random.default_rng() (default) | AbstractRNG`

&nbsp;

### Components
The stages of `egm` can also be used separately, for example to inspect intermediate results:

| Function | Purpose |
|---|---|
| `resolve_knots`, `place_knots`, `SplineBases` | Knot selection and B-spline bases on the data, inference and plotting grids |
| `Smoother`, `smoothing_result` | Mutable smoother for one or more trajectories, as used by the app |
| `SmoothingSession` and its actions | The smoothing app without a window (`select_states!`, `click_knot!`, `equidistant_knots!`, `automatic_knots!`, `smooth!`, `finish!`) |
| `smooth` | B-spline smoothing with noise model estimation (`SmoothingResult`) |
| `KalmanSmoother`, `eks` | Extended Kalman smoothing of all states for given `β` (`KalmanResult`) |
| `augment_covariances` | Extend the Kalman covariances to all measurement time points |
| `GradientMatching`, `gm_step` | One gradient-matching update (`GMStepResult`) |
| `parameter_covariance` | Delta-method covariance of a gradient-matching estimate |
| `ic_candidates` | Candidate initial conditions for the grid search |
| `integrate`, `integrate_variational`, `integrate_sensitivities` | Model trajectories with propagators and parameter sensitivities |

All functions have docstrings, which are shown with `?name` in the Julia REPL.

&nbsp;

### Benchmarks
The folder `benchmarks/` contains Julia ports of the EGM-v2 simulation studies (`accuracy_HS`, `accuracy_GLV`, `test_STAT5`, `test_TCS`, `test_repressilator`). They export all estimates, fits and computing times as JSON and CSV for comparison with the MATLAB results, optionally on exactly the datasets generated by MATLAB. See [benchmarks/README.md](benchmarks/README.md).

&nbsp;

### Differences from EGM-v2
The Julia implementation follows EGM-v2 closely and has been tested against it component by component (see [DEVELOPMENT.md](DEVELOPMENT.md)). The deliberate differences are:
- Models are Julia functions, and all derivatives are computed by automatic differentiation. Auxiliary model files are not needed.
- Model-specific settings (initial-condition scanning, noise mixing matrix) are options instead of special cases in the code.
- The noise mixing matrix `M` enters only the process noise by default. `mode = :compat` restores the EGM-v2 behaviour.
- The optimizers are exact or more accurate. Noise parameters on their bounds are found exactly, whereas `fmincon` stops slightly inside the feasible region. As a result, estimated noise variances can differ by a few percent.

&nbsp;


## References
[1] van Oppen, Yulan B. (2025). _Beyond the average: Computational Approaches for Analyzing and Modeling Cellular Variability._ [Doctoral dissertation, University of Groningen].

[2] van Oppen, Yulan B. and Milias-Argeitis, Andreas. Gradient matching accelerates mixed-effects inference for biochemical networks. _Bioinformatics_, https://doi.org/10.1093/bioinformatics/btaf154.

&nbsp;

## Citation
If you use this software, please cite [1] and the software itself (see [CITATION.cff](CITATION.cff)).

&nbsp;

## DISCLAIMER
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
