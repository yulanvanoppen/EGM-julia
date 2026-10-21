# Interactive smoothing app
_Interactive B-spline smoothing of time series with an additive and multiplicative noise model, for one trajectory or many cells. Part of [EGM.jl](README.md); a Julia port of the smoothing app of EGM-v2 and GMGTS_

&nbsp;

## Quickstart
Download EGM.jl and link it to your environment together with GLMakie (Julia 1.11 or later):
```
git clone https://github.com/yulanvanoppen/EGM-julia
```
```julia
julia> using Pkg
julia> Pkg.activate("path/to/MyProject")
julia> Pkg.develop(path = "path/to/EGM-julia")
julia> Pkg.add("GLMakie")
```
The first installation and use of GLMakie compile for several minutes. Later sessions start much faster. See [INSTALL.md](INSTALL.md) for installing Julia, setting up an environment, and troubleshooting.

Then smooth your data:
```julia
using GLMakie, EGM

out = smoothing_app(t, traces; state_names = ["A", "B"])
```
A window opens. Adjust the knots until the smoothed curves follow the data well, and press Finish. Julia waits until then and returns the knots and the smoothing result.

Run `examples/smoothing_app.jl` (simulated Higgins–Selkov data) or `examples/smoothing_app_TCS.jl` (two-component system) for complete examples.

&nbsp;


## Method
Each measured state is described by a cubic B-spline, whose shape is set by its knots. The measurements are assumed to have variance
```
    Var(y) = σ² + τ² ŷ²
```
with an additive (`σ²`) and a multiplicative (`τ²`) component per state. The spline coefficients and the noise parameters are estimated together by feasible weighted least squares. This alternates two steps until the coefficients converge:
- weighted least-squares spline coefficients, given the current variances;
- maximum-likelihood noise parameters, given the current fit.

With several cells, every cell has its own coefficients, and the noise parameters of a state are shared by all cells.

The knots determine how flexible the smoothed curves are: more knots follow faster changes, fewer knots give smoother curves. The app starts from knots that you provide or from an automatic placement heuristic. The heuristic places knots at peaks and troughs and at the end of initial fast dynamics; with several cells, it uses the mean slopes and curvatures over the cells. You then refine the knots by hand with immediate visual feedback.

&nbsp;


## Usage
### `smoothing_app()` function
`out = smoothing_app(t, traces; ...)` opens the app for measurements `traces` at the time points `t`:
- `traces` is a `T×L` matrix for one trajectory of `L` states, or a `T×L×N` array for `N` cells;
- all trajectories share the time points `t`;
- missing values are not supported.

`out = smoothing_app(_; knots = knots, ...)` starts from the given knots (time units), one vector for all states or one vector per state.

`result = egm(model, t, traces, observed, β_init; interactive = true, ...)` opens the app as the first step of an EGM estimation. See [Using the app with EGM](#using-the-app-with-egm).

&nbsp;

#### Keyword arguments
`state_names` - Names of the states shown in the app  
`["state 1", …] (default) | vector of strings`

`knots` - Initial knot locations (time units)  
`nothing (default) | numeric vector | vector of vectors`  
(one vector for all states, or one vector per state; the end points are added automatically)

`autoknots` - Use the automatic knot placement heuristic when `knots` is not given  
`true (default) | false`  
(if false, `round((T−1)/2) + 1` equidistant knots are used)

`ncells` - Number of cells shown initially  
`20 (default) | positive integer`

`live` - Smooth after every change  
`true (default) | false`

`niter`, `tol` - Maximum number of smoothing iterations and convergence tolerance  
`20, 1e-3 (default) | positive integer, positive scalar`

`positive` - Clip smoothed values at 1e-12 in the result  
`true (default) | false`

`order` - Order of the B-splines  
`4 (default, cubic) | positive integer`

`t_gm` - Time points at which the result is evaluated (as for EGM)  
`t (default) | numeric vector`

`figure_size` - Window size in pixels  
`(1500, 850) (default) | tuple`

&nbsp;

#### Output
`out` is a named tuple:

| Field | Content |
|---|---|
| `knots` | Knots per state, normalized to [0, 1] (including the end points) |
| `knots_time` | Knots per state in time units; can be passed to `egm(_; knots = out.knots_time)` or to `smoothing_app` to start from them again |
| `smoothing` | Smoothing result (below) |
| `action` | How the app was ended: `:finish`, `:estimate` (from EGM) or `:closed` |

`out.smoothing` is a `SmoothingResult` for one trajectory, or a `CellSmoothingResult` for several cells. The latter has one more array dimension for the cells:

| Field | One trajectory | Several cells |
|---|---|---|
| `smoothed`, `dsmoothed` | `Tg×L` smoothed values and time derivatives at `t_gm` | `Tg×L×N` |
| `smoothed_fine`, `dsmoothed_fine` | `81×L` the same on a fine grid, for plotting | `81×L×N` |
| `sigma2`, `tau2` | Noise parameters per state | shared by all cells |
| `variances_sm`, `variances_gm` | Measurement variances at `t` and at `t_gm` | with a cell dimension |
| `delta` | Spline coefficients per state | `card×N` per state |
| `var_delta`, `lincomb` | Covariance of the coefficients, and the linear map from data to coefficients | per state and cell |
| `bases` | B-spline bases and time grids: `bases.t`, `bases.t_gm`, `bases.t_fine` | |

For example, `lines(out.smoothing.bases.t_fine, out.smoothing.smoothed_fine[:, 1])` plots the smoothed first state of a single trajectory.

&nbsp;

### The window
<!-- Screenshots can be added here, e.g. ![startup](docs/images/smoothing_app_startup.png) -->

**Left panel**
- **[State(s)] menu:** Edit all states together ([All]), or one state.
- **[Automatic knot placement] button:** Replace the knots of the selected state(s) using the knot placement heuristic.
- **[#Knots] slider:** Replace the knots of the selected state(s) by equally spaced knots.
- **[Manual knot selector] plot:** Shows the B-spline basis and the knots (dashed lines) of the selected state, or of the first state if [All] is selected. **Left click** adds a knot, **right click** removes the nearest one. With [All] selected, the result is applied to all states.
- **[#Cells] slider:** Number of trajectories shown (several cells only). The most typical cell is shown first, followed by the others from most to least atypical. More cells are revealed as you move the slider.
- **[Smooth] button and [Live] toggle:** Smooth with the current knots now, or after every change.
- **[Finish] button:** Close the app and return the result. States changed without smoothing are smoothed first. When the app is opened by EGM, there are two buttons instead: [Finish and run estimation] and [Finish without estimation].

**Right panel**

Measurements (solid lines) and smoothed curves (dashed lines) of the selected state(s), with one colour per cell.

&nbsp;

### Typical workflow
1. **Get an overview.** With [All] selected, look for states where the smoothed curves miss features of the data (too smooth) or follow the noise (too wiggly).
2. **Select one problem state** in the [State(s)] menu. Changes now apply to this state only.
3. **Adjust its knots.** Add knots where the data change quickly (sharp rises, peaks), and remove knots in flat or noisy regions. Wiggles near the end of the time range often disappear when the last knot is removed. [#Knots] and [Automatic knot placement] reset the knots if needed.
4. **With many cells, check the atypical cells.** Start with few cells to judge the fit, then increase [#Cells] to see whether the knots also suit the atypical cells.
5. **If smoothing slows the app down** (many cells or states), switch [Live] off, make several changes, and press [Smooth].
6. **Check all states again** with [All], then press [Finish].
7. **Keep `out.knots_time`** to reproduce the smoothing later without the app.

&nbsp;

### Using the app with EGM
`egm(_; interactive = true)` smooths the measured states in the app before estimating the parameters:
```julia
using GLMakie, EGM
result = egm(model, t, y, observed, β_init; interactive = true)
```
The app then has two buttons:
- **[Finish and run estimation]** continues with EGM; `result` is an `EGMResult`.
- **[Finish without estimation]** stops after smoothing; `result` is the app's output described above.

Closing the window also stops after smoothing. For scripts, check which one you got:
```julia
if result isa EGMResult
    println("estimate: ", result.β)
else
    println("knots: ", result.knots_time)
end
```
To estimate later with the same smoothing, pass the knots: `egm(_; knots = out.knots_time)`.

&nbsp;

### Smoothing without a window
Everything the app does is also available as functions, for scripts and reproducible analyses. `GLMakie` is not needed for these.

`smooth(SplineBases(t, t_gm, resolve_knots(t, y; knots = knots)), y)` smooths one trajectory with given knots, as EGM does.

`s = Smoother(t, traces; knots = knots)` creates a smoother for one or more trajectories. Then:
- `EGM.optimize!(s)` smooths all states;
- `smoothing_result(s)` returns the result;
- `EGM.set_knots!(s, k, knots)` changes the knots of state `k` (normalized to [0, 1]).

`sess = SmoothingSession(t, traces; state_names = names)` is the app without its window. Each control has a function:

| Control | Function |
|---|---|
| [State(s)] menu | `select_states!(sess, "B")`, `select_states!(sess, :all)` |
| Knot selector clicks | `click_knot!(sess, x)`, `click_knot!(sess, x; remove = true)` |
| [#Knots] slider | `equidistant_knots!(sess, n)` |
| [Automatic knot placement] | `automatic_knots!(sess)` |
| [Smooth] | `smooth!(sess)` |
| [Finish] | `out = finish!(sess)` |

`place_knots(t, traces)` returns the knots of the automatic placement heuristic (normalized), for one or more trajectories.

&nbsp;

### Loading data
Any `T×L` or `T×L×N` array of numbers works. Two common sources:
```julia
using DelimitedFiles                                  # CSV: time in column 1, one column per state
A = readdlm("data.csv", ','; skipstart = 1)
t, traces = A[:, 1], A[:, 2:end]

using MAT                                             # MATLAB file, e.g. a T×L×N array 'traces'
d = matread("data.mat")
traces = d["traces"]
```

&nbsp;


## Notes
- **Graphics.** GLMakie needs OpenGL, which normal Windows, macOS and Linux desktops provide. Remote desktops and virtual machines may not.
- **VS Code / VSCodium.** The app always opens in its own window, also when the editor's plot pane is enabled.
- **Errors in the window.** If an action fails, the error is printed in the REPL and the window stays open.
- **Differences from the MATLAB app** (EGM-v2/GMGTS `smoothing_new.mlapp`):
  - The smoothed curves are plotted at the correct times when the first time point is not 0.
  - Finishing with [Live] off smooths states changed since their last smoothing.
  - Changing the state selection keeps the current smoothing.
  - Duplicate knots are merged.
  - When opened by EGM, the app can stop without estimation.

&nbsp;

## References
See [README.md](README.md#references).
