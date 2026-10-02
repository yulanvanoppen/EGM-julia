# Phase 5: gradient-matching step.
#
# Replaces GM.m (constructor, estimate, covariances), Optimization.QPGLS and
# nearestSPD.m. Given state estimates on the inference grid (B-spline smooths for
# observed states, Kalman smooths for hidden states) and their joint error
# covariance, one step solves the generalized least-squares problem
#
#     dX(t) − h(X(t)) ≈ g(X(t)) β    at the interior time points,
#
# weighted by the (linearized) covariance V of the residuals, subject to box
# constraints on β and an optional Gaussian prior. Derivatives of hidden states
# come from a Savitzky–Golay differentiation matrix.

# --- Savitzky–Golay differentiation -----------------------------------------------------

"""
    sgolay_derivative_weights(order, framelen)

First-derivative weights (per unit grid spacing) of a Savitzky–Golay filter,
i.e. column 2 of MATLAB's `[~, g] = sgolay(order, framelen)`.
"""
function sgolay_derivative_weights(order::Integer, framelen::Integer)
    isodd(framelen) && framelen > order || throw(ArgumentError("need odd framelen > order"))
    h = (framelen - 1) ÷ 2
    S = [Float64(d)^p for d in -h:h, p in 0:order]
    G = S / (S' * S)                                                  # S (SᵀS)⁻¹
    return G[:, 2]
end

"""
    sgolay_matrix(t)

`T × T` differentiation matrix of `GM.m` on the grid `t`:
- rows 4 … T−3: 7-point cubic Savitzky–Golay derivative;
- rows 3 and T−2: 5-point cubic;
- rows 2 and T−1: 3-point central difference;
- rows 1 and T: one-sided first differences.

Row `i` is divided by the local spacing `dt[i]` (one-sided at the ends, the
half-width of the centered interval inside). The weights assume equal spacing;
on non-uniform grids, as in EGM-v2, the result is an approximation.
"""
function sgolay_matrix(t::AbstractVector)
    T = length(t)
    T ≥ 5 || throw(ArgumentError("sgolay_matrix needs at least 5 time points"))
    g7, g5, g3 = sgolay_derivative_weights(3, 7), sgolay_derivative_weights(3, 5), sgolay_derivative_weights(1, 3)
    D = zeros(T, T)
    for i in 4:T-3, d in -3:3
        D[i, i+d] = g7[d+4]
    end
    D[3, 1:5] = g5
    D[T-2, T-4:T] = g5
    D[2, 1:3] = g3
    D[T-1, T-2:T] = g3
    D[1, 1:2] = [-1, 1]
    D[T, T-1:T] = [-1, 1]
    dt = vcat(t[2] - t[1], (t[3:end] .- t[1:end-2]) ./ 2, t[end] - t[end-1])
    return D ./ dt
end

# --- Linear algebra helpers ---------------------------------------------------------------

"""
    nearest_spd(A)

Nearest symmetric positive definite matrix in the Frobenius norm (Higham 1988),
a port of `nearestSPD.m` including its eigenvalue nudging until a Cholesky
factorization succeeds.
"""
function nearest_spd(A::AbstractMatrix)
    r, c = size(A)
    r == c || throw(ArgumentError("A must be square"))
    r == 1 && A[1] ≤ 0 && return fill(eps(), 1, 1)
    B = (A + A') / 2
    F = svd(B)
    H = F.V * Diagonal(F.S) * F.V'
    Ahat = (B + H) / 2
    Ahat = (Ahat + Ahat') / 2
    k = 0
    while !issuccess(cholesky(Symmetric(Ahat); check = false))
        k += 1
        mineig = minimum(eigvals(Symmetric(Ahat)))
        mult = k ≤ 100 ? k^2 : float(k)^10 - 100.0^10 + 100.0^2
        Ahat += (-mineig * mult + eps(mineig)) * I
        if k == 1000
            Ahat += (1e-10 * k^2 + eps(1e-12)) * I
            break
        end
    end
    return Ahat
end

"""
    box_qp(H, f, lb, ub; maxiter = 1000) -> x

Minimize `½xᵀHx + fᵀx` subject to `lb ≤ x ≤ ub` for symmetric positive definite
`H`, with a primal active-set method (exact up to rounding; replaces
`mpcActiveSetSolver`). Bounds may be infinite.
"""
function box_qp(H::AbstractMatrix, f::AbstractVector, lb::AbstractVector, ub::AbstractVector; maxiter = 1000)
    n = length(f)
    all(lb .< ub) || throw(ArgumentError("lb must be strictly below ub"))
    x = clamp.(-(Symmetric(H) \ f), lb, ub)                           # feasible start
    W = [x[i] == lb[i] || x[i] == ub[i] for i in 1:n]                 # working set: variables fixed at a bound
    tol = 1e-12 * max(1.0, norm(f), opnorm(H, 1) * max(1.0, norm(x)))
    solved = false                                                    # x minimizes over the current free set
    for _ in 1:maxiter
        g = H * x + f
        if !solved
            Fr = findall(.!W)
            p = zeros(n)
            isempty(Fr) || (p[Fr] = -(Symmetric(H[Fr, Fr]) \ g[Fr]))
            if norm(p) ≤ 1e-14 * max(1.0, norm(x))
                solved = true
            else
                α, block = 1.0, 0                                     # longest feasible step
                for i in Fr
                    if p[i] < 0 && isfinite(lb[i])
                        a = (lb[i] - x[i]) / p[i]
                        a < α && ((α, block) = (a, i))
                    elseif p[i] > 0 && isfinite(ub[i])
                        a = (ub[i] - x[i]) / p[i]
                        a < α && ((α, block) = (a, i))
                    end
                end
                x = x + max(α, 0.0) * p
                if block == 0
                    solved = true                                     # reached the free-set minimizer
                else
                    x[block] = p[block] < 0 ? lb[block] : ub[block]
                    W[block] = true
                end
                continue
            end
        end
        # optimality of the working set: at lb need g ≥ 0, at ub need g ≤ 0
        viol, worst = 0.0, 0
        for i in findall(W)
            v = x[i] == lb[i] ? -g[i] : g[i]
            if v > viol + tol
                viol, worst = v, i
            end
        end
        worst == 0 && return x
        W[worst] = false
        solved = false
    end
    @warn "box_qp: maximum number of iterations reached"
    return x
end

"""
    qp_gls(design, response, covariance, lb, ub; prior_mean = 0, prior_prec = 0) -> (β, SS)

Box-constrained generalized least squares as `Optimization.QPGLS`: minimizes
`(Y − Xβ)ᵀ V⁻¹ (Y − Xβ) + (β − μ)ᵀ Λ (β − μ)` with `Y = vec(response)`,
`X = design`, `V = covariance`, `μ = prior_mean` and `Λ = prior_prec`, subject to
`lb ≤ β ≤ ub`. Returns the estimate and the weighted residual sum of squares
`SS = (Y − Xβ)ᵀ V⁻¹ (Y − Xβ)`. A scalar `prior_prec` is added to every element of
the Hessian, as in MATLAB; normally pass a `P × P` matrix (or 0).
"""
function qp_gls(design, response, covariance, lb, ub; prior_mean = 0.0, prior_prec = 0.0)
    precision = tryinv(covariance)
    Y = vec(response)
    X = design
    H = X' * precision * X .+ prior_prec
    H = (H + H') / 2
    f = -(X' * precision * Y) .- prior_prec * prior_mean
    isposdef(Symmetric(H)) || (H = nearest_spd(H))
    β = box_qp(H, vec(f), _fvec(lb), _fvec(ub))
    r = Y - X * β
    return β, dot(r, precision * r)
end

# --- Gradient matching -------------------------------------------------------------------

"""
    GradientMatching(model, t, smoothed, dsmoothed, B, dB, lincomb, T_data, observed;
                     lb = fill(-Inf, P), ub = fill(Inf, P), prior_mean = 0, prior_prec = 0)
    GradientMatching(model, sr::SmoothingResult, observed; kwargs...)

Gradient-matching setup on the inference grid `t` (as the `GM` constructor):
- `smoothed`, `dsmoothed`: `T × L` spline estimates of the observed states and
  their derivatives;
- `B`, `dB`: per-state spline bases on `t`;
- `lincomb`: per-state maps from the data (`T_data` points) to spline
  coefficients;
- `lb`, `ub`, `prior_mean`, `prior_prec`: see [`qp_gls`](@ref).
"""
struct GradientMatching{Mdl<:ODEModel,PP}
    model::Mdl
    t::Vector{Float64}
    observed::Vector{Int}
    hidden::Vector{Int}
    smoothed::Matrix{Float64}
    dsmoothed::Matrix{Float64}
    B::Matrix{Float64}                                                # blockdiag of the bases (L·T × Σcard)
    dB::Matrix{Float64}
    lincomb::Matrix{Float64}                                          # blockdiag (Σcard × L·T_data)
    T_data::Int
    fdiff::Matrix{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    prior_mean::Vector{Float64}
    prior_prec::PP
end

function GradientMatching(m::ODEModel, t, smoothed, dsmoothed, B, dB, lincomb, T_data, observed;
                          lb = fill(-Inf, m.P), ub = fill(Inf, m.P), prior_mean = 0.0, prior_prec = 0.0)
    t = _fvec(t)
    T = length(t)
    observed = collect(Int, observed)
    hidden = [k for k in 1:m.K if !(k in observed)]
    L = length(observed)
    smoothed, dsmoothed = reshape(Matrix{Float64}(smoothed), T, L), reshape(Matrix{Float64}(dsmoothed), T, L)
    length(lb) == m.P && length(ub) == m.P || throw(DimensionMismatch("lb and ub need $(m.P) entries"))
    μ = prior_mean isa Number ? fill(Float64(prior_mean), m.P) : _fvec(prior_mean)
    Λ = prior_prec isa Number ? Float64(prior_prec) : Matrix{Float64}(prior_prec)
    return GradientMatching(m, t, observed, hidden, smoothed, dsmoothed, blockdiag_(B), blockdiag_(dB),
                            blockdiag_(lincomb), Int(T_data), sgolay_matrix(t), _fvec(lb), _fvec(ub), μ, Λ)
end

GradientMatching(m::ODEModel, sr::SmoothingResult, observed; kwargs...) =
    GradientMatching(m, sr.bases.t_gm, sr.smoothed, sr.dsmoothed, sr.bases.B_gm, sr.bases.dB_gm, sr.lincomb,
                     size(sr.variances_sm, 1), observed; kwargs...)

blockdiag_(mats) = cat((Matrix{Float64}(A) for A in mats)...; dims = (1, 2))

"""
    states_and_derivatives(gm, xbar) -> (X, dX)

`T × K` state and derivative estimates used for gradient matching: spline
smooths for observed states, Kalman smooths (`xbar`, `K × T`) for hidden
states with Savitzky–Golay derivatives.
"""
function states_and_derivatives(gm::GradientMatching, xbar::AbstractMatrix)
    T, K = length(gm.t), gm.model.K
    X, dX = zeros(T, K), zeros(T, K)
    X[:, gm.observed] = gm.smoothed
    dX[:, gm.observed] = gm.dsmoothed
    fd = gm.fdiff * permutedims(xbar)
    X[:, gm.hidden] = permutedims(xbar[gm.hidden, :])
    dX[:, gm.hidden] = fd[:, gm.hidden]
    return X, dX
end

"""
    gm_design(model, X, dX, t) -> (design, response)

GLS design (`g` stacked over the interior time points, `(K(T−2)) × P`) and
response (`dX − h(X)` at the interior time points, `(T−2) × K`).
"""
function gm_design(m::ODEModel, X, dX, t)
    inner = 2:size(X, 1)-1
    Xi, ti = X[inner, :], t[inner]
    return g_stacked(m, Xi, ti), dX[inner, :] - h_traj(m, Xi, ti)
end

"""
    residual_covariance(gm, β, X, covar_aug) -> (V, variances_XdX)

Covariance of the gradient-matching residuals `dX − g(X)β − h(X)` at the
interior time points (`GM.covariances`). The joint covariance of smoothed states
and measurement errors (`covar_aug`, from [`augment_covariances`](@ref)) is
mapped to the covariance of all states and derivatives (`variances_XdX`,
`2KT × 2KT`, [X; dX] state-major). The residuals are then linearized in X.
A small diagonal regularizer is added and the result made SPD.
"""
function residual_covariance(gm::GradientMatching, β, X, covar_aug)
    m, T, K = gm.model, length(gm.t), gm.model.K
    KT, L = K * T, length(gm.observed)
    LT, LTd = L * T, L * gm.T_data
    size(covar_aug) == (KT + LTd, KT + LTd) || throw(DimensionMismatch("covar_aug must be $(KT + LTd) square"))

    ind_hid = vec((1:T) .+ T .* (gm.hidden .- 1)')
    ind_obs = vec((1:T) .+ T .* (gm.observed .- 1)')
    indices_tk = vec((2:T-1) .+ T .* (0:K-1)')
    indices_2tk = vcat(indices_tk, indices_tk .+ KT)

    dRHS_mixed = df_stacked(m, X, gm.t, β)
    fdiff_full = kron(Matrix(1.0I, K, K), gm.fdiff)
    Δ_X_dX_y = [Matrix(1.0I, KT, KT) zeros(KT, LTd); fdiff_full zeros(KT, LTd); zeros(LTd, KT) Matrix(1.0I, LTd, LTd)]
    covar = Δ_X_dX_y * covar_aug * Δ_X_dX_y'

    Δ_X_dX = zeros(2KT, 2KT + LTd)
    Δ_X_dX[ind_hid, ind_hid] = Matrix(1.0I, KT - LT, KT - LT)
    Δ_X_dX[ind_obs, end-LTd+1:end] = gm.B * gm.lincomb
    Δ_X_dX[ind_hid .+ KT, ind_hid .+ KT] = Matrix(1.0I, KT - LT, KT - LT)
    Δ_X_dX[ind_obs .+ KT, end-LTd+1:end] = gm.dB * gm.lincomb
    variances_XdX = Δ_X_dX * covar * Δ_X_dX'

    dRHS = zeros(KT, KT)
    for j in 1:T
        idx = j .+ T .* (0:K-1)
        dRHS[idx, idx] = dRHS_mixed[idx, :]
    end
    Δ_res = hcat(-dRHS, Matrix(1.0I, KT, KT))
    D = Δ_res[indices_tk, indices_2tk]
    V = D * variances_XdX[indices_2tk, indices_2tk] * D'

    regulator = max.(1e-12, vec(maximum(abs.(X); dims = 1)) ./ 1e6)
    regulator = repeat(regulator; inner = T - 2)
    return nearest_spd(V + Diagonal(regulator)), variances_XdX
end

"""
    GMStepResult

Output of [`gm_step`](@ref): new estimate `β`, weighted residual sum of squares
`SS`, state/derivative estimates `X`, `dX` (`T × K`), GLS `design` and
`response`, residual covariance `V` and `variances_XdX`.
"""
struct GMStepResult
    β::Vector{Float64}
    SS::Float64
    X::Matrix{Float64}
    dX::Matrix{Float64}
    design::Matrix{Float64}
    response::Matrix{Float64}
    V::Matrix{Float64}
    variances_XdX::Matrix{Float64}
end

"""
    gm_step(gm, β, xbar, covar_aug) -> GMStepResult

One gradient-matching update (`GM.estimate`): combine spline and Kalman
estimates, build the GLS problem, compute the residual covariance at the current
`β`, and solve the box-constrained GLS.
"""
function gm_step(gm::GradientMatching, β, xbar, covar_aug)
    X, dX = states_and_derivatives(gm, xbar)
    design, response = gm_design(gm.model, X, dX, gm.t)
    V, vXdX = residual_covariance(gm, _fvec(β), X, covar_aug)
    βnew, SS = qp_gls(design, response, nearest_spd(V), gm.lb, gm.ub;
                      prior_mean = gm.prior_mean, prior_prec = gm.prior_prec)
    return GMStepResult(βnew, SS, X, dX, design, response, V, vXdX)
end

# --- Parameter uncertainty -----------------------------------------------------------------

"""
    beta_jacobian(gm, step::GMStepResult) -> Matrix

Derivatives of the (unconstrained, prior-free) GLS estimate
`β̂ = (GᵀV⁻¹G)⁻¹GᵀV⁻¹(dX − H)` with respect to the states and derivatives at the
interior time points, with `V` held fixed. Returns a `P × 2K(T−2)` matrix:
columns `[∂β/∂X, ∂β/∂dX]`, each block state-major over the interior time points.
This ports `GM.uncertainties`; see section S6 of
https://doi.org/10.1093/bioinformatics/btaf154.
"""
function beta_jacobian(gm::GradientMatching, step::GMStepResult)
    m, t = gm.model, gm.t
    T, K, P = length(t), m.K, m.P
    X, dX = step.X, step.dX
    inner = 2:T-1
    indices_tk = vec(inner .+ T .* (0:K-1)')
    G = g_stacked(m, X, t)[indices_tk, :]
    dg_all = dg_array(m, X, t)
    H = vec(h_traj(m, X, t)[inner, :])
    dh_all = dh_stacked(m, X, t)
    r = vec(dX[inner, :]) - H

    Vinv = tryinv(step.V)
    Thinv = tryinv(G' * Vinv * G)
    dβ_ddX = Thinv * G' * Vinv
    b = Thinv * (G' * Vinv * r)
    Π, Ψ = zeros(P, T, K), zeros(P, T, K)
    for k in 1:K, j in inner
        dg = dg_all[:, :, j, k]                                       # ∂G/∂X[j, k] at the rows of time j
        Vj = Vinv[j - 1 .+ (T - 2) .* (0:K-1), :]
        Π[:, j, k] = (dg' * Vj * G + G' * Vj' * dg) * b               # ∂(GᵀV⁻¹G)/∂X[j, k] β̂
        Ψ[:, j, k] = dg' * Vj * r - G' * Vj' * dh_all[j .+ T .* (0:K-1), k]   # ∂(GᵀV⁻¹(dX − H))/∂X[j, k]
    end
    dβ_dX = Thinv * reshape(Ψ[:, inner, :] - Π[:, inner, :], P, :)
    return hcat(dβ_dX, dβ_ddX)
end

"""
    parameter_covariance(gm, step::GMStepResult) -> Matrix

Delta-method approximation of the covariance of the gradient-matching estimate:
`J Σ Jᵀ`, with `J` from [`beta_jacobian`](@ref) and `Σ` the covariance of the
states and derivatives at the interior time points (`step.variances_XdX`).
Bounds and priors are not taken into account.
"""
function parameter_covariance(gm::GradientMatching, step::GMStepResult)
    T, K = length(gm.t), gm.model.K
    indices_tk = vec((2:T-1) .+ T .* (0:K-1)')
    indices_2tk = vcat(indices_tk, indices_tk .+ K * T)
    J = beta_jacobian(gm, step)
    C = J * step.variances_XdX[indices_2tk, indices_2tk] * J'
    return (C + C') / 2
end
