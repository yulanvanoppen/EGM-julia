# Phase 3: spline smoother with an additive + multiplicative noise model.
#
# Replaces Smoother.optimize / update_coefficients / update_variances / fit and
# Optimization.noise_parameters (single trajectory, N = 1). Measurement
# variances are modelled as
#
#     Var(y_i) = σ² + τ² ŷ_i²,
#
# and estimated by feasible weighted least squares (FWLS). Each iteration
# computes WLS spline coefficients for the current variances and then updates
# (σ², τ²) by maximum likelihood.
#
# The MATLAB version uses lsqnonneg for the starting values (exact here, see
# `nnls2`) and fmincon for the likelihood. `noise_mle` replaces fmincon with a
# projected Newton method on the bound-constrained problem. It reaches the
# constrained optimum to high accuracy, so it agrees with fmincon to within
# fmincon's own tolerances.

# --- Linear algebra helpers ---------------------------------------------------------

"Solve `A X = B` for symmetric positive (semi)definite `A` (Cholesky, LU fallback)."
function _spd_solve(A, B)
    S = Symmetric((A + A') / 2)
    F = cholesky(S; check = false)
    return issuccess(F) ? F \ B : lu(Matrix(S)) \ B
end

"""
    tryinv(A)

Inverse of `A`, falling back to the pseudo-inverse if `A` is (nearly) singular
(reciprocal condition number below machine precision). This mirrors the intent
of `tryinv.m`.
"""
function tryinv(A::AbstractMatrix)
    F = lu(A; check = false)
    (issuccess(F) && 1 / cond(A, 1) ≥ eps()) && return inv(F)
    return pinv(A)
end

"Relative change ‖old − new‖/‖old‖ as `eucl_rel.m` (0 if equal, Inf if `old` = 0 ≠ new)."
function relative_change(old, new)
    no, nd = norm(old), norm(old .- new)
    iszero(no) && return iszero(nd) ? 0.0 : Inf
    return nd / no
end

# --- Spline coefficients --------------------------------------------------------------

"""
    wls_coefficients(B, y, variances) -> (lincomb, δ)

Weighted least-squares spline coefficients with weights `1 ./ max.(variances, 1e-10)`.
`lincomb = (BᵀWB)⁻¹BᵀW` is the linear map from data to coefficients, and
`δ = lincomb * y`.
"""
function wls_coefficients(B::AbstractMatrix, y::AbstractVector, variances::AbstractVector)
    W = Diagonal(1 ./ max.(variances, 1e-10))
    lincomb = _spd_solve(B' * W * B, B' * W)
    return lincomb, lincomb * y
end

# --- Noise parameters -------------------------------------------------------------------

"""
    nnls2(X, y)

Exact non-negative least squares for a two-column design matrix `X`: the best
feasible solution over the four possible supports.
"""
function nnls2(X::AbstractMatrix, y::AbstractVector)
    size(X, 2) == 2 || throw(ArgumentError("nnls2 needs exactly two columns"))
    best, bestres = zeros(2), sum(abs2, y)
    candidates = Vector{Float64}[]
    c = X \ y
    all(≥(0), c) && push!(candidates, c)
    for j in 1:2
        cj = dot(X[:, j], y) / sum(abs2, X[:, j])
        cj ≥ 0 && push!(candidates, j == 1 ? [cj, 0.0] : [0.0, cj])
    end
    for c in candidates
        res = sum(abs2, y .- X * c)
        if res < bestres
            best, bestres = c, res
        end
    end
    return best
end

"""
    noise_objective(c, pred, y)

Negative log-likelihood (up to constants) of the noise model
`y ~ N(pred, c[1] + c[2] pred²)`: `Σ log vᵢ + (yᵢ − predᵢ)²/vᵢ`. Returns `Inf`
outside the domain (some vᵢ ≤ 0).
"""
function noise_objective(c, pred, y)
    s = 0.0
    for (p, yi) in zip(pred, y)
        v = c[1] + c[2] * p^2
        v > 0 || return Inf
        s += log(v) + (yi - p)^2 / v
    end
    return s
end

"Gradient and Hessian of [`noise_objective`](@ref) with respect to `c`."
function noise_gradient_hessian(c, pred, y)
    g = zeros(2)
    H = zeros(2, 2)
    for (p, yi) in zip(pred, y)
        a = p^2
        v = c[1] + c[2] * a
        r2 = (yi - p)^2
        d = (1.0, a)
        w1 = 1 / v - r2 / v^2
        w2 = -1 / v^2 + 2 * r2 / v^3
        for j in 1:2
            g[j] += w1 * d[j]
            for k in 1:2
                H[j, k] += w2 * d[j] * d[k]
            end
        end
    end
    return g, H
end

"""
    noise_mle(start, pred, y; maxiter = 500)

Maximum-likelihood estimate of `(σ², τ²) ≥ 0` for `Var(y) = σ² + τ² pred²`,
by projected Newton iterations with backtracking, starting from `start`.
"""
function noise_mle(start, pred, y; maxiter = 500)
    c = max.(Float64.(collect(start)), 0.0)
    f = noise_objective(c, pred, y)
    isfinite(f) || throw(ArgumentError("noise_mle: starting point outside the domain"))
    for _ in 1:maxiter
        g, H = noise_gradient_hessian(c, pred, y)
        free = [!(c[j] ≤ 0 && g[j] > 0) for j in 1:2]              # variables not held at the bound
        any(free) || break
        d = zeros(2)
        Hf, gf = H[free, free], g[free]
        newton = isposdef(Symmetric(Hf))
        if newton
            d[free] = -(Hf \ gf)
        else                                                        # diagonally scaled descent
            d[free] = -gf ./ max.(abs.(diag(Hf)), eps())
        end
        # Stop on a small Newton step, not on a small decrease in f: changes in f near the
        # optimum drown in rounding error, which would limit the parameters to ~sqrt(eps).
        dot(g, d) ≥ 0 && break                                      # no descent direction left
        norm(d) ≤ 1e-12 * max(norm(c), eps()) && break
        if newton && norm(d) ≤ 1e-6 * norm(c)                       # quadratic regime: full step
            cn = max.(c .+ d, 0.0)
            fn = noise_objective(cn, pred, y)
        else                                                        # projected backtracking
            α, accepted = 1.0, false
            cn, fn = c, f
            while α > 1e-20
                cn = max.(c .+ α .* d, 0.0)
                fn = noise_objective(cn, pred, y)
                if fn ≤ f + 1e-4 * dot(g, cn .- c)
                    accepted = true
                    break
                end
                α /= 2
            end
            accepted || break
        end
        step = norm(cn .- c)
        c, f = cn, fn
        step ≤ 1e-14 * max(norm(c), eps()) && break
    end
    return c
end

# --- Smoother ---------------------------------------------------------------------------

"""
    SmoothingResult

Output of [`smooth`](@ref). Per-state quantities are vectors indexed by the
observed state; grid quantities are `time × state` matrices.

Fields: `bases` (the [`SplineBases`](@ref)), `delta` (spline coefficients),
`sigma2`, `tau2` (noise parameters), `variances_sm` (`T × L`, data grid),
`variances_gm` (`Tg × L`), `smoothed`, `dsmoothed` (`Tg × L`),
`smoothed_fine`, `dsmoothed_fine` (`Tf × L`), `lincomb` (`card × T` maps from
data to coefficients), `var_delta` (coefficient covariances) and `iterations`.
"""
struct SmoothingResult
    bases::SplineBases
    delta::Vector{Vector{Float64}}
    sigma2::Vector{Float64}
    tau2::Vector{Float64}
    variances_sm::Matrix{Float64}
    variances_gm::Matrix{Float64}
    smoothed::Matrix{Float64}
    dsmoothed::Matrix{Float64}
    smoothed_fine::Matrix{Float64}
    dsmoothed_fine::Matrix{Float64}
    lincomb::Vector{Matrix{Float64}}
    var_delta::Vector{Matrix{Float64}}
    iterations::Int
end

"Initial variances used by `Smoother`: `y² + 0.1 mean(y)²` (mostly multiplicative)."
initial_variances(traces::AbstractMatrix) = traces .^ 2 .+ 0.1 .* (sum(traces; dims = 1) ./ size(traces, 1)) .^ 2

"""
    update_noise(B, B_gm, δ, y) -> (c, variances, variances_gm, start)

One noise-parameter update for a single state, as `Smoother.update_variances`:
`lsqnonneg` start, likelihood maximization, and the implied variances on the
data and inference grids. Returns `c = [σ², τ²]` and the start used.
"""
function update_noise(B, B_gm, δ, y)
    pred = B * δ
    design = hcat(ones(length(pred)), pred .^ 2)
    response = (pred .- y) .^ 2
    start = nnls2(design, response)
    sum(start) == 0 && (start[1] = sum(response) / length(response))
    c = noise_mle(start, pred, y)
    sum(c) == 0 && (c[1] = sum(response) / length(response))
    pred_gm = B_gm * δ
    return c, design * c, hcat(ones(length(pred_gm)), pred_gm .^ 2) * c, start
end

"""
    smooth(bases::SplineBases, traces; niter = 20, tol = 1e-3, positive = true,
           variances_init = initial_variances(traces)) -> SmoothingResult

Smooth the measurements `traces` (`T × L`, observed states on the data grid) with
B-splines, jointly estimating the noise model `Var(y) = σ² + τ² ŷ²` per state by
FWLS. This follows `Smoother.optimize` and `Smoother.fit`:
- iterate WLS coefficients and noise-parameter updates until the relative change
  in the coefficients is below `tol` (at most `niter` iterations);
- evaluate the splines on the inference and fine grids;
- clip the smoothed values at 1e-12 if `positive`.
"""
function smooth(sb::SplineBases, traces::AbstractMatrix; niter::Integer = 20, tol = 1e-3,
                positive::Bool = true, variances_init = nothing)
    T, L = size(traces)
    length(sb.splines) == L || throw(DimensionMismatch("bases for $(length(sb.splines)) states, traces has $L"))
    size(sb.B[1], 1) == T || throw(DimensionMismatch("bases for $(size(sb.B[1], 1)) time points, traces has $T"))
    Tg = length(sb.t_gm)

    variances = variances_init === nothing ? initial_variances(traces) : Matrix{Float64}(variances_init)
    variances_gm = zeros(Tg, L)
    delta = [zeros(b.card) for b in sb.splines]
    lincomb = [zeros(b.card, T) for b in sb.splines]
    sigma2, tau2 = zeros(L), zeros(L)

    iterations = 0
    for iter in 1:max(1, niter)
        iterations = iter
        delta_old = deepcopy(delta)
        for k in 1:L                                                   # WLS coefficients
            lincomb[k], delta[k] = wls_coefficients(sb.B[k], traces[:, k], variances[:, k])
        end
        for k in 1:L                                                   # noise parameters
            c, v, v_gm, _ = update_noise(sb.B[k], sb.B_gm[k], delta[k], traces[:, k])
            sigma2[k], tau2[k] = c
            variances[:, k] = v
            variances_gm[:, k] = v_gm
        end
        relative_change(reduce(vcat, delta_old), reduce(vcat, delta)) < tol && break
    end

    smoothed = reduce(hcat, [sb.B_gm[k] * delta[k] for k in 1:L])
    dsmoothed = reduce(hcat, [sb.dB_gm[k] * delta[k] for k in 1:L])
    smoothed_fine = reduce(hcat, [sb.B_fine[k] * delta[k] for k in 1:L])
    dsmoothed_fine = reduce(hcat, [sb.dB_fine[k] * delta[k] for k in 1:L])
    positive && (smoothed = max.(smoothed, 1e-12))

    var_delta = [tryinv(sb.B[k]' * (Diagonal(max.(variances[:, k], 1e-10)) \ sb.B[k])) for k in 1:L]

    return SmoothingResult(sb, delta, sigma2, tau2, variances, variances_gm, smoothed, dsmoothed,
                           smoothed_fine, dsmoothed_fine, lincomb, var_delta, iterations)
end

Base.show(io::IO, r::SmoothingResult) =
    print(io, "SmoothingResult(", length(r.delta), " states, ", r.iterations, " iterations)")
