# Shared infrastructure for the benchmark scripts (ports of EGM-v2's accuracy_*.m and
# test_*.m): data sources, method runners with timing, and JSON/CSV export.
#
# Environment variables (all optional):
#   EGM_DATA      "auto" (default): use MATLAB-exported datasets if present, else simulate
#                 "matlab": require benchmarks/datasets/<script>.mat
#                 "julia":  simulate the data in Julia (different random numbers than MATLAB)
#   EGM_SEEDS     subset of seeds, e.g. "1:5" or "3" (default: all, as in MATLAB)
#   EGM_THREADED  "1" to run the cases in parallel threads (timings are then less reliable)
#   EGM_WARMUP    "0" to skip the untimed warm-up run (default: "1")

using EGM, LinearAlgebra, Random, Printf, Dates
using MAT

const BENCH_DIR = @__DIR__
const MODEL_DIR = joinpath(BENCH_DIR, "models")
const DATA_DIR = joinpath(BENCH_DIR, "datasets")
const OUT_DIR = joinpath(BENCH_DIR, "results")

"Relative error as EGM-v2's eucl_rel(x, y) = ‖x − y‖ / ‖x‖."
eucl_rel(x, y) = norm(x .- y) / norm(x)

"Load an IQM model file from benchmarks/models (call at top level, see compile_iqm)."
load_model(file; fixed = String[], M = I) = compile_iqm(load_iqm(joinpath(MODEL_DIR, file); fixed = fixed, M = M))

"Seeds to run: `default`, or the subset given in ENV[\"EGM_SEEDS\"]."
function seed_selection(default)
    s = strip(get(ENV, "EGM_SEEDS", ""))
    isempty(s) && return collect(default)
    sel = occursin(':', s) ? (parse(Int, split(s, ':')[1]):parse(Int, split(s, ':')[2])) : (1:parse(Int, s))
    return [x for x in default if x in sel]
end

# --- Data sources -----------------------------------------------------------------------------

"""
    dataset_source(name) -> (source, data)

`(:matlab, Dict)` with the datasets exported by `matlab/export_datasets.m`, or
`(:julia, nothing)` to simulate in Julia (see EGM_DATA).
"""
function dataset_source(name)
    choice = lowercase(get(ENV, "EGM_DATA", "auto"))
    file = joinpath(DATA_DIR, name * ".mat")
    if choice == "matlab" || (choice == "auto" && isfile(file))
        isfile(file) || error("EGM_DATA=matlab but $file does not exist (run matlab/export_datasets.m)")
        return :matlab, matread(file)
    end
    return :julia, nothing
end

_vec(x) = x isa AbstractArray ? vec(Float64.(x)) : [Float64(x)]
_mat(x, ncol) = reshape(Float64.(x isa AbstractArray ? x : fill(x, 1, 1)), :, ncol)

"""
    matlab_run(data, row, K, L) -> NamedTuple

Dataset `row` of a MATLAB export: `t`, `traces` (T×L), `truth` (T×K), `noisevar` (T×K),
`init` (initial conditions used for the simulation) and `ic_mult` (accuracy scripts).
"""
function matlab_run(data, row, K, L)
    cell(key) = data[key][row]
    return (; t = _vec(cell("t")), traces = _mat(cell("traces"), L), truth = _mat(cell("original"), K),
            noisevar = _mat(cell("noisevar"), K), init = vec(data["init"][row, :]),
            ic_mult = haskey(data, "ic_mult") ? vec(data["ic_mult"][row, :]) : Float64[])
end

"Find the row of a MATLAB export with the given index tuple (first columns of data[\"index\"])."
function matlab_row(data, idx...)
    index = Int.(reshape(data["index"], :, size(data["index"], 2)))
    r = findfirst(i -> all(index[i, j] == idx[j] for j in eachindex(idx)), axes(index, 1))
    r === nothing && error("no dataset with index $idx in the MATLAB export")
    return r
end

"""
    julia_run(model, t, observed, noise, rng) -> NamedTuple

Simulated dataset as EGM-v2's Generator (N = 1, init = x0 + 1e-8, multiplicative noise).
"""
function julia_run(model, t, observed, noise, rng)
    init = model.x0 .+ 1e-8
    d = generate_data(model, t; x0 = init, observed = observed, error_std = noise, rng = rng)
    noisevar = noise^2 .* d.truth[:, :, 1] .^ 2
    return (; t = collect(Float64, t), traces = d.traces[:, :, 1], truth = d.truth[:, :, 1], noisevar = noisevar,
            init = init, ic_mult = Float64[])
end

# --- Method runners ------------------------------------------------------------------------------

"""
    run_egm(model, t, y, observed, β_init; kwargs...) -> NamedTuple

Run `egm` and collect what the MATLAB scripts record; on failure, record the error
(times 99, as in MATLAB).
"""
function run_egm(model, t, y, observed, β_init; kwargs...)
    t0 = time_ns()
    try
        r = egm(model, t, y, observed, β_init; kwargs...)
        wall = (time_ns() - t0) / 1e9
        return (; status = "ok", message = "", beta = r.β, init = r.init, time = collect(r.time), wall = wall,
                iterations = r.iterations, fitted = r.fitted, beta_sd = sqrt.(max.(diag(r.β_cov), 0)),
                result = r)
    catch err
        return (; status = "error", message = sprint(showerror, err), beta = fill(NaN, model.P),
                init = fill(NaN, model.K), time = [99.0, 99.0], wall = (time_ns() - t0) / 1e9, iterations = 0,
                fitted = nothing, beta_sd = fill(NaN, model.P), result = nothing)
    end
end

"Run `tm` (trajectory matching); `time = [0, inference time]` as in EGM-v2's estimate_TM."
function run_tm(model, t, y, observed, β_init; kwargs...)
    t0 = time_ns()
    try
        r = tm(model, t, y, observed, β_init; kwargs...)
        wall = (time_ns() - t0) / 1e9
        return (; status = "ok", message = "", beta = r.β, init = r.x0, time = [0.0, wall], wall = wall,
                iterations = r.iterations, fitted = r.fitted, beta_sd = fill(NaN, model.P), result = r)
    catch err
        return (; status = "error", message = sprint(showerror, err), beta = fill(NaN, model.P),
                init = fill(NaN, model.K), time = [99.0, 99.0], wall = (time_ns() - t0) / 1e9, iterations = 0,
                fitted = nothing, beta_sd = fill(NaN, model.P), result = nothing)
    end
end

"Method record for the JSON output (plus any extra metrics)."
method_record(res, k0; extra...) =
    merge((; status = res.status, message = res.message, beta = res.beta, beta_sd = res.beta_sd, init = res.init,
           time_smoothing = res.time[1], time_inference = res.time[2], time_wall = res.wall,
           iterations = res.iterations, accuracy = res.status == "ok" ? eucl_rel(k0, res.beta) : 99.0,
           fitted = res.fitted), values(extra))

"""
    run_cases(f, cases; warmup = true)

Run `f(case)` for every case (serially, or threaded if EGM_THREADED=1), printing
progress. With `warmup`, `f(first(cases))` is run once beforehand, untimed, so that
compilation does not count towards the first timing.
"""
function run_cases(f, cases; warmup = get(ENV, "EGM_WARMUP", "1") != "0")
    if warmup && !isempty(cases)
        print("warm-up run (compilation) … ")
        t0 = time_ns()
        f(first(cases))
        @printf("%.1f s\n", (time_ns() - t0) / 1e9)
    end
    records = Vector{Any}(undef, length(cases))
    if get(ENV, "EGM_THREADED", "0") == "1"
        Threads.@threads for i in eachindex(cases)
            records[i] = f(cases[i])
        end
    else
        for (i, c) in enumerate(cases)
            records[i] = f(c)
            @printf("[%d/%d] %s\n", i, length(cases), summary_line(records[i]))
        end
    end
    return records
end

function summary_line(rec)
    parts = String[]
    for key in (:EGM, :TM)
        haskey(rec, key) || continue
        m = rec[key]
        push!(parts, @sprintf("%s: %s acc %.3g, %.2f s", key, m.status, m.accuracy, m.time_smoothing + m.time_inference))
    end
    return string(rec.case, " ", join(parts, " | "))
end

# --- Export ----------------------------------------------------------------------------------------

function _json(io::IO, x)
    if x === nothing || x === missing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
    elseif x isa AbstractString || x isa Symbol
        s = string(x)
        print(io, '"')
        for c in s
            if c == '"' || c == '\\'
                print(io, '\\', c)
            elseif c == '\n'
                print(io, "\\n")
            elseif c < ' '
                @printf(io, "\\u%04x", UInt32(c))
            else
                print(io, c)
            end
        end
        print(io, '"')
    elseif x isa AbstractMatrix                                       # array of rows
        print(io, '[')
        for i in axes(x, 1)
            i > first(axes(x, 1)) && print(io, ',')
            _json(io, x[i, :])
        end
        print(io, ']')
    elseif x isa AbstractVector || x isa Tuple
        print(io, '[')
        for (i, v) in enumerate(x)
            i > 1 && print(io, ',')
            _json(io, v)
        end
        print(io, ']')
    elseif x isa NamedTuple || x isa AbstractDict
        print(io, '{')
        for (i, (k, v)) in enumerate(pairs(x))
            i > 1 && print(io, ',')
            _json(io, string(k))
            print(io, ':')
            _json(io, v)
        end
        print(io, '}')
    else
        _json(io, string(x))
    end
end

"Write `x` as JSON (matrices as arrays of rows; NaN/Inf as null)."
write_json(path, x) = open(io -> _json(io, x), path, "w")

_csvfield(v::AbstractString) = occursin(r"[\",\n]", v) ? "\"" * replace(v, "\"" => "\"\"") * "\"" : v
_csvfield(v::Nothing) = ""
_csvfield(v::AbstractFloat) = isfinite(v) ? repr(Float64(v)) : (isnan(v) ? "NaN" : (v > 0 ? "Inf" : "-Inf"))
_csvfield(v) = string(v)

"Write rows (NamedTuples) as CSV; columns are the union of keys in order of appearance."
function write_csv(path, rows)
    cols = Symbol[]
    for r in rows, k in keys(r)
        k in cols || push!(cols, k)
    end
    open(path, "w") do io
        println(io, join(string.(cols), ","))
        for r in rows
            println(io, join([_csvfield(get(r, c, nothing)) for c in cols], ","))
        end
    end
end

"NamedTuple from a vector of `Symbol => value` pairs."
_nt(ps) = NamedTuple{Tuple(Symbol(first(p)) for p in ps)}(Tuple(last(p) for p in ps))

"""
    flat_rows(records, model)

One CSV row per case and method: case indices and settings, status, times, accuracy,
iterations, extra scalar metrics, β (`beta_<name>`), standard errors and initial
conditions (`init_<state>`).
"""
function flat_rows(records, model)
    rows = NamedTuple[]
    for rec in records, method in (:EGM, :TM)
        haskey(rec, method) || continue
        m = rec[method]
        base = merge((; case = rec.case), rec.indices, rec.settings,
                     (; data_source = rec.data_source, method = string(method), status = m.status,
                      time_smoothing = m.time_smoothing, time_inference = m.time_inference, time_wall = m.time_wall,
                      iterations = m.iterations, accuracy = m.accuracy))
        skip = (:status, :message, :beta, :beta_sd, :init, :time_smoothing, :time_inference, :time_wall,
                :iterations, :accuracy, :fitted)
        extras = _nt([k => getfield(m, k) for k in keys(m) if !(k in skip) && getfield(m, k) isa Real])
        betas = _nt([Symbol("beta_", p) => m.beta[j] for (j, p) in enumerate(model.parameters)])
        sds = _nt([Symbol("beta_sd_", p) => m.beta_sd[j] for (j, p) in enumerate(model.parameters)])
        inits = _nt([Symbol("init_", st) => m.init[k] for (k, st) in enumerate(model.states)])
        push!(rows, merge(base, extras, betas, sds, inits, (; message = m.message)))
    end
    return rows
end

"""
    export_results(name, model, records, settings)

Write benchmarks/results/<name>_<data source>_<timestamp>.json (all records,
including fits) and .csv (flat table), and print their paths.
"""
function export_results(name, model, records, settings)
    mkpath(OUT_DIR)
    source = isempty(records) ? "none" : records[1].data_source
    stamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    base = joinpath(OUT_DIR, "$(name)_$(source)_$(stamp)")
    header = (; script = name, julia_version = string(VERSION), egm_version = string(pkgversion(EGM)),
              threads = Threads.nthreads(), threaded = get(ENV, "EGM_THREADED", "0") == "1",
              date = string(now()), model = (; name = model.name, states = model.states,
                                              parameters = model.parameters, k0 = model.β0, x0 = model.x0),
              settings = settings, runs = records)
    write_json(base * ".json", header)
    write_csv(base * ".csv", flat_rows(records, model))
    println("wrote $(base).json and $(base).csv")
    return base
end

# --- Script drivers ---------------------------------------------------------------------------------

"""
    run_accuracy(name, model; dt_values, noise_levels, init_error, seeds, observed,
                 egm_kwargs = (;), tm_kwargs = (;)) -> records

Port of the loops in accuracy_HS.m / accuracy_GLV.m. For every (dt, noise level,
initial-condition error, seed):
- simulate (or load) measurements on `0:dt:10`;
- draw the initial-condition multipliers `exp.(N(0, error²))`;
- estimate with EGM (inference grid 0:0.5:10, or 0:1:10 for dt = 1) and with TM,
  both from β_init = k0/2 and x_init = multipliers .* x0.
"""
function run_accuracy(name, model; dt_values, noise_levels, init_error, seeds, observed,
                      egm_kwargs = (;), tm_kwargs = (;))
    source, data = dataset_source(name)
    k0, K, L = model.β0, model.K, length(observed)
    β_init = 0.5 .* k0
    cases = [(dt_idx, noise_idx, init_idx, seed) for dt_idx in eachindex(dt_values)
             for noise_idx in eachindex(noise_levels) for init_idx in eachindex(init_error) for seed in seeds]
    println("$name: $(length(cases)) cases, data source $source, $(Threads.nthreads()) thread(s)")
    function case_fn(c)
        dt_idx, noise_idx, init_idx, seed = c
        dt, noise, err = dt_values[dt_idx], noise_levels[noise_idx], init_error[init_idx]
        if source === :matlab
            d = matlab_run(data, matlab_row(data, dt_idx, noise_idx, init_idx, seed), K, L)
            ic_mult = d.ic_mult
        else
            rng = MersenneTwister(10000 * dt_idx + 1000 * noise_idx + 100 * init_idx + seed - 1)
            d = julia_run(model, 0:dt:10, observed, noise, rng)
            ic_mult = exp.(err .* randn(rng, K))
        end
        x_init = ic_mult .* model.x0
        t_gm = collect(0:(dt_idx == 3 ? 1.0 : 0.5):10)
        egm_res = run_egm(model, d.t, d.traces, observed, β_init; x_init = x_init, t_gm = t_gm, egm_kwargs...)
        tm_res = run_tm(model, d.t, d.traces, observed, β_init; x_init = x_init, tm_kwargs...)
        return (; case = "dt=$dt noise=$noise init_error=$err seed=$seed",
                indices = (; dt_idx, noise_idx, init_idx, seed), settings = (; dt, noise, init_error = err),
                data_source = string(source), ic_multiplier = ic_mult, t = d.t, traces = d.traces,
                EGM = method_record(egm_res, k0), TM = method_record(tm_res, k0))
    end
    return run_cases(case_fn, cases)
end

"Model-fit measure of the test_*.m scripts: mean of (y − ŷ)² / x_true over the observed data."
fit_measure(res, d, observed) = res.status == "ok" ?
    sum((d.traces .- res.fitted[:, observed]) .^ 2 ./ d.truth[:, observed]) / length(d.traces) : NaN

"""
    run_test(name, model; t, observed, noise, β_init, x_init, nseeds = 100, prepare = identity,
             egm_kwargs = (;), tm_kwargs = (;), metrics = (res, egm_res, d) -> (;)) -> records

Port of the loop in test_*.m: per seed, simulate (or load) measurements, apply
`prepare` (e.g. drop the first time point), and estimate with EGM and TM. Records
β, initial conditions, times, the fit measure and any extra `metrics`.
"""
function run_test(name, model; t, observed, noise, β_init, x_init, nseeds = 100, prepare = identity,
                  egm_kwargs = (;), tm_kwargs = (;), metrics = (res, egm_res, d) -> (;))
    source, data = dataset_source(name)
    k0, K, L = model.β0, model.K, length(observed)
    seeds = seed_selection(1:nseeds)
    println("$name: $(length(seeds)) seeds, data source $source, $(Threads.nthreads()) thread(s)")
    function case_fn(seed)
        d = source === :matlab ? matlab_run(data, matlab_row(data, seed), K, L) :
                                 julia_run(model, t, observed, noise, MersenneTwister(seed))
        d = prepare(d)
        egm_res = run_egm(model, d.t, d.traces, observed, β_init; x_init = x_init, egm_kwargs...)
        tm_res = run_tm(model, d.t, d.traces, observed, β_init; x_init = x_init, tm_kwargs...)
        rec(res) = method_record(res, k0; fit = fit_measure(res, d, observed), metrics(res, egm_res, d)...)
        return (; case = "seed=$seed", indices = (; seed), settings = (; noise), data_source = string(source),
                t = d.t, traces = d.traces, EGM = rec(egm_res), TM = rec(tm_res))
    end
    return run_cases(case_fn, seeds)
end

"Drop the first time point of a dataset (test_STAT5.m)."
drop_first(d) = merge(d, (; t = d.t[2:end], traces = d.traces[2:end, :], truth = d.truth[2:end, :],
                           noisevar = d.noisevar[2:end, :]))

"""
    sum_of_squares(model, β, d, observed)

Port of sum_of_squares.m: weighted squared error of the model integrated with β from
the data-generating initial conditions `d.init`, weighted by the true noise variances.
"""
function sum_of_squares(model, β, d, observed)
    any(isnan, β) && return NaN
    pred = integrate(model, β, d.init, d.t)
    return sum((d.traces .- pred[:, observed]) .^ 2 ./ d.noisevar[:, observed])
end

"""
    gss(model, egm_result, β, observed)

Port of GSS.m: the generalized (gradient-matching) sum of squares of β, using the
state/derivative estimates and the final Kalman covariances of an EGM run.
"""
function gss(model, r, β, observed)
    (r === nothing || any(isnan, β)) && return NaN
    sr = r.smoothing
    gm = GradientMatching(model, sr, observed)
    aug = augment_covariances(r.kalman.covar, sr.variances_sm, sr.bases.t, sr.bases.t_gm, model.K, length(observed))
    V, _ = EGM.residual_covariance(gm, β, r.gm.X, aug)
    design, response = EGM.gm_design(model, r.gm.X, r.gm.dX, sr.bases.t_gm)
    res = vec(response) - design * β
    return dot(res, EGM.tryinv(EGM.nearest_spd(V)) * res)
end
