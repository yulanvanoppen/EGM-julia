# Phase 6: complete estimation (EGM.estimate and EGM.init_ICmean).
#
#   1. B-spline smoothing of the observed states (Phase 3),
#   2. initial-condition search: one Kalman smoothing + gradient-matching step per
#      candidate vector of initial conditions; the candidate with the smallest
#      weighted residual sum of squares is kept,
#   3. E-M iterations: Kalman smoothing at the current β (Phase 4), then a
#      gradient-matching update (Phase 5), until the relative change in β is below
#      `tol_gm`,
#   4. final model fit from the smoothed initial state.
#
# The model-name special cases of EGM.init_ICmean are replaced by the options
# `ic_scan`, `ic_ties` and `ic_constraint`.

"""
    ic_candidates(center, uncertain; multipliers = 4.0 .^ (-1:0.5:1), ties = Pair{Int,Int}[],
                  constraint = identity) -> Matrix

Candidate initial-condition vectors (one per row) as in `EGM.init_ICmean`:
- every combination of `center[u] * multiplier` over the states in `uncertain`,
  with the first uncertain state varying fastest;
- then `ties` (`target => source` sets state `target` equal to state `source`);
- then `constraint`, a function applied to each candidate vector.

Without uncertain states, the only candidate is `center`.

Examples of the MATLAB special cases:
- generalized Lotka–Volterra: `uncertain = [1]`, `ties = [2 => 1]`;
- STAT5: `uncertain = [3, 4]`, `constraint = x -> (x[4] = min(x[3], x[4]); x)`.
"""
function ic_candidates(center, uncertain; multipliers = 4.0 .^ (-1:0.5:1), ties = Pair{Int,Int}[],
                       constraint = identity)
    center = _fvec(center)
    cands = Vector{Vector{Float64}}()
    if isempty(uncertain)
        push!(cands, copy(center))
    else
        for ms in Iterators.product(ntuple(_ -> multipliers, length(uncertain))...)
            c = copy(center)
            for (u, mult) in zip(uncertain, ms)
                c[u] = center[u] * mult
            end
            push!(cands, c)
        end
    end
    for c in cands, (target, source) in ties
        c[target] = c[source]
    end
    cands = [_fvec(constraint(copy(c))) for c in cands]
    return permutedims(reduce(hcat, cands))
end

"""
    EGMResult

Output of [`egm`](@ref):
- `β`: final estimate; `β_cov`: its approximate covariance (delta method, see
  [`parameter_covariance`](@ref)); `β_history`: one row per E-M iterate (row 1 =
  after the initial-condition search); `SS_history`: weighted residual sums of
  squares; `iterations`: number of E-M iterations.
- `x_init`: selected initial-condition prior mean, with the search results in
  `ic_candidates`, `ic_betas`, `ic_errors` and `ic_index`.
- `init`: smoothed initial state used for the final fit; `fitted`,
  `dfitted` (on `t`) and `fitted_fine`, `dfitted_fine` (on `t_fine`): model
  fit and right-hand side.
- `smoothing`, `kalman`, `gm`: the [`SmoothingResult`](@ref), the last
  [`KalmanResult`](@ref) and the last [`GMStepResult`](@ref).
- `t`, `t_gm`, `t_fine`: time grids; `time`: seconds spent in smoothing and
  inference.
"""
struct EGMResult
    β::Vector{Float64}
    β_cov::Matrix{Float64}
    β_history::Matrix{Float64}
    SS_history::Vector{Float64}
    iterations::Int
    x_init::Vector{Float64}
    ic_candidates::Matrix{Float64}
    ic_betas::Matrix{Float64}
    ic_errors::Vector{Float64}
    ic_index::Int
    init::Vector{Float64}
    fitted::Matrix{Float64}
    dfitted::Matrix{Float64}
    fitted_fine::Matrix{Float64}
    dfitted_fine::Matrix{Float64}
    smoothing::SmoothingResult
    kalman::KalmanResult
    gm::GMStepResult
    t::Vector{Float64}
    t_gm::Vector{Float64}
    t_fine::Vector{Float64}
    time::NTuple{2,Float64}
end

function Base.show(io::IO, r::EGMResult)
    print(io, "EGMResult(β = ", round.(r.β; sigdigits = 5), ", ", r.iterations, " iterations)")
end

"""
    egm(model, t, traces, observed, β_init; kwargs...) -> EGMResult

Estimate the free parameters of `model` from measurements `traces`
(`length(t) × L`) of the states `observed`, starting from `β_init`.

Data and grids
- `t_gm = t`: inference time points (a subset of `t`).

Smoothing
- `knots = nothing`: knots in time units, one vector shared by all observed
  states or one vector per state.
- `autoknots = true`: automatic knot placement when `knots` is not given.
- `niter_sm = 20`, `tol_sm = 1e-3`: smoother iterations and tolerance.
- `positive = true`: clip smoothed values and fits at 1e-12.
- `interactive = false`: choose the knots in the smoothing app
  ([`smoothing_app`](@ref), requires `using GLMakie`). The app starts from
  `knots` or the automatic placement. The smoothing time then includes the time
  spent in the app. If the app is finished with "Finish without estimation", or
  closed, `egm` stops and returns the app's result `(; knots, knots_time,
  smoothing, action)` instead of an `EGMResult`.

Kalman smoothing
- `x_init = model.x0`: initial-condition guess.
- `P_CV = 0.25`: relative prior spread of the initial conditions.
- `σ = 0.05`: process-noise scale.
- `mode = :default`: see [`predict_step`](@ref).
- `alg = Tsit5()`: ODE solver.

Initial-condition search
- `ic_scan`: states to scan; defaults to the hidden states.
- `ic_multipliers = 4.0 .^ (-1:0.5:1)`, `ic_ties`, `ic_constraint`: see
  [`ic_candidates`](@ref).

Gradient matching
- `lb = 0.25 β0`, `ub = 4 β0 + 1e-4 mean(β0)`: parameter bounds, where
  `β0 = model.β0`. As in EGM-v2, they must bracket `model.β0`.
- `prior_mean = 0`, `prior_prec = 0`: Gaussian prior. Alternatively,
  `prior_sd` (a vector) sets `prior_prec = inv(Diagonal(prior_sd.^2))`.
- `niter_gm = 20`, `tol_gm = 1e-4`: E-M iterations and tolerance on the relative
  change in β.

Output
- `verbose = false`: print progress.
"""
function egm(m::ODEModel, t, traces, observed, β_init; knots = nothing, autoknots::Bool = true,
             t_gm = t, niter_sm::Integer = 20, tol_sm = 1e-3, positive::Bool = true, interactive::Bool = false,
             kwargs...)
    t = _fvec(t)
    traces = Matrix{Float64}(reshape(traces, length(t), :))
    t_gm = sort(unique(_fvec(t_gm)))
    all(in(t), t_gm) || throw(ArgumentError("t_gm must be a subset of the data time points t"))
    issorted(t) && allunique(t) || throw(ArgumentError("t must be strictly increasing"))
    t0 = time_ns()
    if interactive                                                    # knots chosen in the app (needs GLMakie)
        app = smoothing_app(t, traces; knots = knots, autoknots = autoknots, t_gm = t_gm,
                            state_names = m.states[collect(observed)], niter = max(1, niter_sm),
                            tol = max(1e-12, tol_sm), positive = positive, estimation = true)
        if app.action !== :estimate                                   # smoothing only
            @info "Estimation skipped: the smoothing app was finished without estimation (action = :$(app.action)). " *
                  "Returning the smoothing result."
            return app
        end
        sr = app.smoothing
    else
        kn = resolve_knots(t, traces; knots = knots, autoknots = autoknots)
        sb = SplineBases(t, t_gm, kn)
        sr = smooth(sb, traces; niter = max(1, niter_sm), tol = max(1e-12, tol_sm), positive = positive)
    end
    time_sm = (time_ns() - t0) / 1e9
    return egm_inference(m, sr, traces, observed, β_init; positive = positive, time_sm = time_sm, kwargs...)
end

"""
    egm_inference(model, sr::SmoothingResult, traces, observed, β_init; kwargs...) -> EGMResult

Steps 2–4 of [`egm`](@ref), starting from an existing smoothing result (for
example, one computed with different settings or loaded from MATLAB). Accepts the
same keywords as `egm` except those for smoothing.
"""
function egm_inference(m::ODEModel, sr::SmoothingResult, traces, observed, β_init;
                       x_init = m.x0, P_CV = 0.25, σ = 0.05, mode::Symbol = :default, alg = Tsit5(),
                       ic_scan = nothing, ic_multipliers = 4.0 .^ (-1:0.5:1), ic_ties = Pair{Int,Int}[],
                       ic_constraint = identity,
                       lb = 0.25 .* m.β0, ub = 4 .* m.β0 .+ 1e-4 * sum(m.β0) / m.P,
                       prior_mean = 0.0, prior_prec = 0.0, prior_sd = nothing,
                       niter_gm::Integer = 20, tol_gm = 1e-4, positive::Bool = true,
                       verbose::Bool = false, time_sm = 0.0)
    K, P = m.K, m.P
    sb = sr.bases
    t, t_gm, t_fine = sb.t, sb.t_gm, sb.t_fine
    traces = Matrix{Float64}(reshape(traces, length(t), :))
    observed = collect(Int, observed)
    β_init = _fvec(β_init)
    lb, ub = _fvec(lb), _fvec(ub)
    length(β_init) == P || throw(DimensionMismatch("β_init needs $P entries"))
    all(lb .< m.β0) && all(m.β0 .< ub) || throw(ArgumentError("lb and ub must bracket model.β0, as in EGM-v2"))
    if prior_sd !== nothing
        prior_prec = inv(Diagonal(_fvec(prior_sd) .^ 2))
    end
    niter_gm, tol_gm = max(1, niter_gm), max(1e-12, tol_gm)

    t0 = time_ns()
    begin
        idx = findall(in(t_gm), t)
        ks = KalmanSmoother(m, t_gm, traces[idx, :], sr.variances_gm, observed;
                            σ = σ, P_CV = P_CV, x_init = x_init, mode = mode, alg = alg)
        gm = GradientMatching(m, sr, observed; lb = lb, ub = ub, prior_mean = prior_mean, prior_prec = prior_prec)
        L = length(observed)
        function smooth_step(β, x0)                                   # Kalman smoothing + GM update
            kr = eks(ks, β; x_init = x0)
            aug = augment_covariances(kr.covar, sr.variances_sm, t, t_gm, K, L)
            return kr, gm_step(gm, β, kr.xbar, aug)
        end

        # initial-condition search (EGM.init_ICmean)
        center = _fvec(x_init)
        center[observed] = sr.smoothed[1, :]
        uncertain = ic_scan === nothing ? gm.hidden : collect(Int, ic_scan)
        cands = ic_candidates(center, uncertain; multipliers = ic_multipliers, ties = ic_ties,
                              constraint = ic_constraint)
        ncand = size(cands, 1)
        ic_betas, ic_errors = zeros(ncand, P), zeros(ncand)
        for c in 1:ncand
            _, ic_step = smooth_step(β_init, cands[c, :])
            ic_betas[c, :], ic_errors[c] = ic_step.β, ic_step.SS
        end
        ic_index = argmin(ic_errors)
        x_chosen = cands[ic_index, :]
        verbose && println("initial conditions: candidate $ic_index of $ncand, SS = $(ic_errors[ic_index])")

        # E-M iterations
        βs = [ic_betas[ic_index, :]]
        SSs = Float64[]
        local r, step
        for iter in 1:niter_gm
            r, step = smooth_step(βs[end], x_chosen)
            push!(βs, step.β)
            push!(SSs, step.SS)
            change = relative_change(βs[end-1], βs[end])
            verbose && println("iteration $iter: β = $(round.(step.β; sigdigits = 6)), relative change $change")
            change < tol_gm && break
        end
    end
    time_est = (time_ns() - t0) / 1e9

    # final fit from the smoothed initial state
    init = r.xbar[:, 1]
    fitted = integrate(m, βs[end], init, t; alg = alg)
    fitted_fine = integrate(m, βs[end], init, t_fine; alg = alg)
    dfitted = rhs_traj(m, fitted, t, βs[end])
    dfitted_fine = rhs_traj(m, fitted_fine, t_fine, βs[end])
    if positive
        fitted, fitted_fine = max.(fitted, 1e-12), max.(fitted_fine, 1e-12)
    end

    β_cov = parameter_covariance(gm, step)
    return EGMResult(βs[end], β_cov, permutedims(reduce(hcat, βs)), SSs, length(SSs), x_chosen, cands, ic_betas,
                     ic_errors, ic_index, init, fitted, dfitted, fitted_fine, dfitted_fine, sr, r, step,
                     t, t_gm, t_fine, (Float64(time_sm), time_est))
end

"""
    egm_cells(model, t, traces, observed, β_init; threaded = true, kwargs...) -> Vector{EGMResult}

Run [`egm`](@ref) independently for each cell (trajectory) in `traces`
(`T × L × N`), in parallel threads if `threaded` (start Julia with
`--threads=auto`). `β_init` and `x_init` (keyword) may be a single vector for all
cells or an `N`-vector of vectors. Other keywords are passed to `egm`.

This gives cell-specific estimates only. EGM-v2 itself handles one trajectory,
and population-level (mixed-effects) inference as in GMGTS is not included.
"""
function egm_cells(m::ODEModel, t, traces::AbstractArray{<:Real,3}, observed, β_init;
                   threaded::Bool = true, x_init = m.x0, kwargs...)
    N = size(traces, 3)
    per_cell(v) = v isa AbstractVector && !isempty(v) && first(v) isa AbstractVector ? v : fill(v, N)
    βs, xs = per_cell(β_init), per_cell(x_init)
    length(βs) == N && length(xs) == N || throw(DimensionMismatch("need one β_init and x_init per cell"))
    results = Vector{EGMResult}(undef, N)
    if threaded
        Threads.@threads for i in 1:N
            results[i] = egm(m, t, traces[:, :, i], observed, βs[i]; x_init = xs[i], kwargs...)
        end
    else
        for i in 1:N
            results[i] = egm(m, t, traces[:, :, i], observed, βs[i]; x_init = xs[i], kwargs...)
        end
    end
    return results
end
