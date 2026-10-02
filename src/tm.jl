# Trajectory matching (TM.m and EGM.estimate_TM): a baseline that fits the integrated
# model directly to the measurements by weighted least squares, optionally including
# the initial conditions, alternated with re-estimation of the noise model
# Var(y) = σ² + τ² ŷ² (as in the spline smoother).
#
# MATLAB uses fmincon in log-parameter space. Here a bounded Levenberg–Marquardt
# method with exact forward sensitivities is used instead (∂x/∂β follows from
# dS/dt = J S + g(x, t), since f is affine in β). Results agree when both reach the
# least-squares optimum; the search paths differ.

"""
    lm_bounded(resjac, θ0, lo, hi; maxiter = 200, xtol = 1e-10, ftol = 1e-12) -> (θ, SS)

Levenberg–Marquardt minimization of `‖r(θ)‖²` subject to `lo ≤ θ ≤ hi`, with
projected steps. `resjac(θ)` returns the residual vector and its Jacobian; it
may throw (for example when integration fails), which counts as an infinite
residual.
"""
function lm_bounded(resjac, θ0, lo, hi; maxiter = 200, xtol = 1e-10, ftol = 1e-12)
    safe(θ) = try
        resjac(θ)
    catch
        nothing
    end
    θ = clamp.(_fvec(θ0), lo, hi)
    rj = safe(θ)
    rj === nothing && throw(ArgumentError("lm_bounded: residuals cannot be evaluated at the starting point"))
    r, Jm = rj
    S = sum(abs2, r)
    λ = 1e-3
    for _ in 1:maxiter
        g = Jm' * r
        A = Jm' * Jm
        free = .!(((θ .<= lo) .& (g .> 0)) .| ((θ .>= hi) .& (g .< 0)))
        any(free) || break
        scale = max.(diag(A)[free], 1e-12 * max(1.0, maximum(diag(A))))
        accepted, done = false, false
        while λ ≤ 1e16
            δ = zeros(length(θ))
            δ[free] = -((A[free, free] + λ * Diagonal(scale)) \ g[free])
            θn = clamp.(θ .+ δ, lo, hi)
            rjn = safe(θn)
            Sn = rjn === nothing ? Inf : sum(abs2, rjn[1])
            if Sn < S
                done = norm(θn .- θ) ≤ xtol * (norm(θ) + xtol) || S - Sn ≤ ftol * S
                θ, (r, Jm), S = θn, rjn, Sn
                λ = max(λ / 10, 1e-12)
                accepted = true
                break
            end
            λ *= 10
        end
        (!accepted || done) && break
    end
    return θ, S
end

"""
    TMResult

Output of [`tm`](@ref):
- `β`, `x0`: estimates; `β_history`: one row per iteration (row 1 = start);
  `iterations`.
- `sigma2`, `tau2`, `variances`: noise model per observed state.
- `fitted`, `dfitted`, `fitted_fine`, `dfitted_fine`: fits on `t` and `t_fine`.
- `ic_candidates`, `ic_errors`, `ic_index`: initial-condition search (empty if
  `estimate_x0 = false`).
"""
struct TMResult
    β::Vector{Float64}
    x0::Vector{Float64}
    β_history::Matrix{Float64}
    iterations::Int
    sigma2::Vector{Float64}
    tau2::Vector{Float64}
    variances::Matrix{Float64}
    fitted::Matrix{Float64}
    dfitted::Matrix{Float64}
    fitted_fine::Matrix{Float64}
    dfitted_fine::Matrix{Float64}
    t::Vector{Float64}
    t_fine::Vector{Float64}
    ic_candidates::Matrix{Float64}
    ic_errors::Vector{Float64}
    ic_index::Int
end

Base.show(io::IO, r::TMResult) =
    print(io, "TMResult(β = ", round.(r.β; sigdigits = 5), ", ", r.iterations, " iterations)")

"""
    tm(model, t, traces, observed, β_init; estimate_x0 = true, x_init = model.x0,
       lb = 0.25 β0, ub = 4 β0 + 1e-4 mean(β0), positive = true, niter = 20, tol = 1e-4,
       ic_scan = nothing, ic_multipliers = 4.0 .^ (-1:0.5:1), ic_ties = Pair{Int,Int}[],
       ic_constraint = identity, reltol = 1e-6, alg = Tsit5()) -> TMResult

Trajectory matching (EGM-v2's `estimate_TM`). It alternates two steps:
1. weighted least squares of the integrated model against the measurements, over
   `β` and, if `estimate_x0`, the initial conditions;
2. re-estimation of the noise model per observed state.

The observed initial conditions start at the first measurements. With
`estimate_x0`, an initial-condition search like that of [`egm`](@ref) selects the
starting point. Initial conditions are bounded below by `eps()` if `positive`
(else −100) and above by 10⁴, as in EGM-v2.
"""
function tm(m::ODEModel, t, traces, observed, β_init; estimate_x0::Bool = true, x_init = m.x0,
            lb = 0.25 .* m.β0, ub = 4 .* m.β0 .+ 1e-4 * sum(m.β0) / m.P, positive::Bool = true,
            niter::Integer = 20, tol = 1e-4, ic_scan = nothing, ic_multipliers = 4.0 .^ (-1:0.5:1),
            ic_ties = Pair{Int,Int}[], ic_constraint = identity, reltol = 1e-6, alg = Tsit5())
    K, P = m.K, m.P
    t = _fvec(t)
    T = length(t)
    traces = Matrix{Float64}(reshape(traces, T, :))
    observed = collect(Int, observed)
    L = length(observed)
    lb, ub = _fvec(lb), _fvec(ub)
    variances = initial_variances(traces)
    x0 = _fvec(x_init)
    x0[observed] = traces[1, :]
    β = _fvec(β_init)
    lo_x, hi_x = fill(positive ? eps() : eps() - 100, K), fill(1e4, K)

    function resjac(θ)
        βθ = θ[1:P]
        xθ = estimate_x0 ? θ[P+1:end] : x0
        X, Sx, Sβ = integrate_sensitivities(m, βθ, xθ, t; reltol = reltol, alg = alg)
        w = 1 ./ sqrt.(variances)
        r = vec((X[:, observed] .- traces) .* w)
        Jm = zeros(T * L, estimate_x0 ? P + K : P)
        for (l, k) in enumerate(observed), i in 1:T
            row = i + T * (l - 1)
            Jm[row, 1:P] = Sβ[k, :, i] .* w[i, l]
            estimate_x0 && (Jm[row, P+1:end] = Sx[k, :, i] .* w[i, l])
        end
        return r, Jm
    end
    lo = estimate_x0 ? vcat(lb, lo_x) : lb
    hi = estimate_x0 ? vcat(ub, hi_x) : ub

    # initial-condition search (TM.init_x0): one least-squares fit per candidate
    cands, ic_errors, ic_index = zeros(0, K), Float64[], 0
    if estimate_x0
        uncertain = ic_scan === nothing ? [k for k in 1:K if !(k in observed)] : collect(Int, ic_scan)
        cands = ic_candidates(x0, uncertain; multipliers = ic_multipliers, ties = ic_ties, constraint = ic_constraint)
        ic_errors = fill(Inf, size(cands, 1))
        fits = Vector{Vector{Float64}}(undef, size(cands, 1))
        for c in axes(cands, 1)
            θc = clamp.(vcat(β, cands[c, :]), lo, hi)
            try
                fits[c], ic_errors[c] = lm_bounded(resjac, θc, lo, hi; maxiter = 50, xtol = 1e-6, ftol = 1e-8)
            catch
                fits[c] = θc
            end
        end
        ic_index = argmin(ic_errors)
        β, x0 = fits[ic_index][1:P], fits[ic_index][P+1:end]
    end

    # alternate least squares and noise re-estimation
    βs = [copy(β)]
    sigma2, tau2 = zeros(L), zeros(L)
    fitted = zeros(T, K)
    for _ in 1:max(1, niter)
        θ, _ = lm_bounded(resjac, estimate_x0 ? vcat(β, x0) : β, lo, hi)
        β = θ[1:P]
        estimate_x0 && (x0 = θ[P+1:end])
        fitted = integrate(m, β, x0, t; reltol = reltol, alg = alg)
        positive && (fitted = max.(fitted, 1e-12))
        for (l, k) in enumerate(observed)
            pred, y = fitted[:, k], traces[:, l]
            design = hcat(ones(T), pred .^ 2)
            response = (pred .- y) .^ 2
            start = nnls2(design, response)
            sum(start) == 0 && (start[1] = sum(response) / T)
            c = noise_mle(start, pred, y)
            sum(c) == 0 && (c[1] = sum(response) / T)
            sigma2[l], tau2[l] = c
            variances[:, l] = design * c
        end
        push!(βs, copy(β))
        relative_change(βs[end-1], βs[end]) < max(1e-12, tol) && break
    end

    t_fine = matlab_linspace(t[1], t[end], 81)
    fitted = integrate(m, β, x0, t; reltol = reltol, alg = alg)
    fitted_fine = integrate(m, β, x0, t_fine; reltol = reltol, alg = alg)
    dfitted, dfitted_fine = rhs_traj(m, fitted, t, β), rhs_traj(m, fitted_fine, t_fine, β)
    if positive
        fitted, fitted_fine = max.(fitted, 1e-12), max.(fitted_fine, 1e-12)
    end
    return TMResult(β, x0, permutedims(reduce(hcat, βs)), length(βs) - 1, sigma2, tau2, variances,
                    fitted, dfitted, fitted_fine, dfitted_fine, t, t_fine, cands, ic_errors, ic_index)
end
