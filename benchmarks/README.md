# Benchmarks

Julia ports of the EGM-v2 simulation scripts. Each script estimates parameters with EGM and with trajectory matching (TM) over many simulated datasets, as the MATLAB script does, and exports all estimates, fits and computing times.

| Julia script | MATLAB script | Cases |
|---|---|---|
| `accuracy_HS.jl` | `accuracy_HS.m` | 3 sampling intervals × 4 noise levels × 5 initial-condition errors × 50 seeds |
| `accuracy_GLV.jl` | `accuracy_GLV.m` | the same, for the generalized Lotka–Volterra model |
| `test_STAT5.jl` | `test_STAT5.m` | 100 seeds; also the sum of squares and generalized sum of squares |
| `test_TCS.jl` | `test_TCS.m` | 100 seeds |
| `test_repressilator.jl` | `test_repressilator.m` | 100 seeds |

&nbsp;

## Running
Set up the benchmark environment once (Julia 1.11 or later). It uses the EGM.jl package in the parent folder:
```
julia --project=benchmarks -e "using Pkg; Pkg.instantiate()"
```

Then run a script from the package folder:
```
julia --project=benchmarks benchmarks/test_STAT5.jl
```

From a REPL in an editor (VS Code / VSCodium with the Julia extension), with the repository folder as the working directory:
```julia
using Pkg
Pkg.activate("benchmarks")
Pkg.instantiate()                       # first time only
ENV["EGM_SEEDS"] = "1:2"                # optional settings, see below
include("benchmarks/test_STAT5.jl")
```
Scripts can be included one after another in the same session.

Options are set with environment variables:

`EGM_DATA` - Source of the datasets  
`"auto" (default) | "matlab" | "julia"`  
(`"auto"` uses `datasets/<script>.mat` when it exists, and simulates in Julia otherwise)

`EGM_SEEDS` - Subset of seeds  
`all (default) | "1:5" | "3"`  
(`"3"` means seeds 1–3)

`EGM_THREADED` - Run cases in parallel threads (start Julia with `--threads=auto`)  
`"0" (default) | "1"`  
(per-case timings are less reliable when threads share the machine)

`EGM_WARMUP` - Run the first case once, untimed, before the timed runs  
`"1" (default) | "0"`  
(otherwise the first case's timing includes Julia's compilation)

For example, on Linux or macOS:
```
EGM_SEEDS=1:5 julia --project=benchmarks benchmarks/accuracy_HS.jl
```
On Windows (PowerShell):
```
$env:EGM_SEEDS = "1:5"; julia --project=benchmarks benchmarks/accuracy_HS.jl
```

&nbsp;

## Same datasets as MATLAB
Julia cannot reproduce MATLAB's random numbers. For a case-by-case comparison, run `matlab/export_datasets.m` (set `REPO` first). It repeats only the data generation of the five MATLAB scripts, with identical `rng` calls, and writes `benchmarks/datasets/<script>.mat`. The Julia scripts then use exactly these datasets, including the initial-condition errors of the accuracy scripts. With `EGM_DATA = "julia"`, Julia simulates its own datasets with the same design, which is suitable for comparing distributions and timings, but not individual cases.

&nbsp;

## Output
Each run writes two files to `benchmarks/results/`, named `<script>_<data source>_<date>_<time>`.

### JSON (everything)
```
{
  "script", "julia_version", "egm_version", "threads", "threaded", "date",
  "model":    { "name", "states", "parameters", "k0", "x0" },
  "settings": { … script settings … },
  "runs": [
    {
      "case":        "dt=0.25 noise=0.005 init_error=0.5 seed=1",
      "indices":     { "dt_idx", "noise_idx", "init_idx", "seed" }      (accuracy)  |  { "seed" }  (test)
      "settings":    { "dt", "noise", "init_error" }                    (accuracy)  |  { "noise" } (test)
      "data_source": "matlab" | "julia",
      "ic_multiplier": [ … ]                                            (accuracy only)
      "t": [ … ],  "traces": [[ … ], …],                                (data used, rows = time points)
      "EGM": { … },  "TM": { … }
    }, …
  ]
}
```

Fields of `"EGM"` and `"TM"`:

| Field | Content | MATLAB counterpart |
|---|---|---|
| `status`, `message` | `"ok"` or `"error"` with the error message | try/catch |
| `beta` | Parameter estimate (order of `model.parameters`) | `out.beta` |
| `beta_sd` | Delta-method standard errors (EGM only) | — |
| `init` | Initial conditions: smoothed initial state (EGM) or estimated x0 (TM) | `out.init` |
| `time_smoothing`, `time_inference` | Computing times in seconds; TM: `[0, total]`; 99 on failure | `out.time` |
| `time_wall` | Wall time of the whole call | — |
| `iterations` | Number of E-M (EGM) or least-squares/noise (TM) iterations | — |
| `accuracy` | `‖k0 − β‖ / ‖k0‖`; 99 on failure | `accuracies_*` |
| `fitted` | Model fit on `t`, all states (rows = time points) | `out.fitted` |
| `fit` | Mean of `(y − ŷ)² / x_true` over the observed data (test scripts) | `fits_*` |
| `SS`, `GSS` | Sum of squares and generalized sum of squares (test_STAT5) | `SS_*`, `GSS_*` |

Missing or non-finite values are written as `null`. In MATLAB, `jsondecode(fileread(file))` returns the runs as a struct array.

### CSV (one row per case and method)
Columns:
- `case`, then the indices and settings;
- `data_source`, `method`, `status`;
- `time_smoothing`, `time_inference`, `time_wall`, `iterations`, `accuracy`;
- `fit`, `SS`, `GSS` (where computed);
- `beta_<parameter>`, `beta_sd_<parameter>`, `init_<state>`;
- `message`.

In MATLAB, read it with `readtable(file)`.

&nbsp;

## Notes
- Trajectory matching refits at least once with the estimated noise model (`min_iter = 2`, see `?tm`). Results from before this change stopped after one iteration and used the initial noise weights.
- `test_STAT5.jl` runs in `mode = :compat` (constant `MODE`) to match the current EGM-v2 code.
- `accuracy_GLV.m` refers to `auxiliary_GLV_legacyConvention.txt`, which is not in the EGM-v2 repository. The Julia port uses the standard variational equation, like the other models.
- Models are read from the EGM-v2 model files in `benchmarks/models` (`load_iqm` and `compile_iqm`).
