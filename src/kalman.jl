# Phase 4: extended Kalman filter and RTS smoother with joint covariances.
#
# Replaces Filter.m and EGM.augment_covariances. The smoother runs on the
# inference grid t_gm. Besides smoothed means and covariances it builds the joint
# covariance matrix of all smoothed state errors and all measurement errors,
# which gradient matching (Phase 5) needs to propagate uncertainty.
#
# Process noise. Between time points the model is linearized along the
# predicted trajectory, with diffusion L(x) = M·diag(σx) (see `diffusion`):
#
#     Q_n = ∫ Φ(s) M D(s) D(s)ᵀ Mᵀ Φ(s)ᵀ ds,   D(s) = diag(σ x(s)),   Φ(t_{n−1}) = I,
#
# computed with the trapezoidal rule on 21 points, as `Filter.cumulative_noise`.
# Modes:
#   :default  the propagator Φ_n = Φ(t_n) is used for prediction and smoothing
#             (M enters only the noise);
#   :compat   Φ_n = Φ(t_n)·M, reproducing EGM-v2, where the variational system
#             starts at M (identical to :default when M = I).
#
# The covariance recursions are a literal port of Filter.update_covariance_EKF/EKS,
# including their index layout, so intermediate matrices can be compared directly.

# --- Prediction step ----------------------------------------------------------------------

"""
    process_noise(tout, X, Φ, σ, M) -> Q

Trapezoidal approximation of `∫ Φ(s) M D(s) (Φ(s) M D(s))ᵀ ds` with
`D(s) = diag(σ x(s))`, from states `X` (`n × K`) and propagators `Φ` (`K × K × n`)
at the times `tout`. With `M = I` and MATLAB's raw auxiliary output this is
exactly `Filter.cumulative_noise`.
"""
function process_noise(tout, X, Φ, σ, M)
    K = size(X, 2)
    integrand = [let A = Φ[:, :, i] * M * Diagonal(σ .* X[i, :]); A * A' end for i in eachindex(tout)]
    Q = zeros(K, K)
    for i in 1:length(tout)-1                                        # trapz as in MATLAB
        Q .+= (tout[i+1] - tout[i]) .* (integrand[i] .+ integrand[i+1])
    end
    return Q ./ 2
end

"""
    predict_step(model, β, x0, t0, t1; σ, mode = :default, nsub = 20, reltol = 1e-4,
                 alg = Tsit5()) -> (x_pre, Φ_n, Q_n)

One EKF prediction from `x0` at `t0` to `t1`: integrates the model and its
variational system on `nsub + 1` equally spaced points (as
`Filter.integrate_augmented`) and returns the predicted state, the propagator
used for the covariance update, and the process-noise covariance.
"""
function predict_step(m::ODEModel, β, x0, t0, t1; σ, mode::Symbol = :default, nsub::Integer = 20,
                      reltol = 1e-4, abstol = reltol / 1000, alg = Tsit5())
    mode in (:default, :compat) || throw(ArgumentError("mode must be :default or :compat"))
    tout = matlab_linspace(t0, t1, nsub + 1)
    X, Φ = integrate_variational(m, β, x0, tout; reltol, abstol, alg)
    Q = process_noise(tout, X, Φ, σ, m.M)
    Φn = mode === :compat ? Φ[:, :, end] * m.M : Φ[:, :, end]
    return X[end, :], Φn, Q
end

# --- Kalman smoother ----------------------------------------------------------------------

"""
    KalmanSmoother(model, t, y, variances, observed; σ = 0.05, P_CV = 0.25,
                   x_init = model.x0, mode = :default, nsub = 20, reltol = 1e-4,
                   alg = Tsit5())

Extended Kalman filter/smoother setup on the time grid `t` (the inference grid).

- `y`, `variances`: `T × L` measurements of the observed states and their
  variances (from the spline smoother) on `t`.
- `observed`: indices of the observed states.
- `σ`: process-noise scale.
- `P_CV`, `x_init`: define the initial-condition prior covariance
  `diag((P_CV x_init)² + (0.01 mean(y))²)`. As in `Filter.m`, this covariance
  is fixed at construction, even when [`eks`](@ref) is later called with a
  different `x_init`.
- `mode`, `nsub`, `reltol`, `alg`: see [`predict_step`](@ref).
"""
struct KalmanSmoother{Mdl<:ODEModel,A}
    model::Mdl
    t::Vector{Float64}
    y::Matrix{Float64}
    variances::Matrix{Float64}
    observed::Vector{Int}
    C::Matrix{Float64}
    σ::Float64
    x_init::Vector{Float64}
    P_init::Matrix{Float64}
    mode::Symbol
    nsub::Int
    reltol::Float64
    alg::A
end

function KalmanSmoother(m::ODEModel, t, y, variances, observed; σ = 0.05, P_CV = 0.25, x_init = m.x0,
                        mode::Symbol = :default, nsub::Integer = 20, reltol = 1e-4, alg = Tsit5())
    y = Matrix{Float64}(reshape(y, length(t), :))
    L = size(y, 2)
    observed = collect(Int, observed)
    length(observed) == L || throw(DimensionMismatch("$(length(observed)) observed states, y has $L columns"))
    size(variances) == size(y) || throw(DimensionMismatch("variances must be $(size(y))"))
    mode in (:default, :compat) || throw(ArgumentError("mode must be :default or :compat"))
    C = zeros(L, m.K)
    for (l, k) in enumerate(observed)
        C[l, k] = 1.0
    end
    x_init = _fvec(x_init)
    regularizer = 0.01 * sum(y) / length(y)
    P_init = Matrix(Diagonal((P_CV .* x_init) .^ 2 .+ regularizer^2))
    return KalmanSmoother(m, _fvec(t), y, Matrix{Float64}(variances), observed, C, Float64(σ), x_init,
                          P_init, mode, Int(nsub), Float64(reltol), alg)
end

Base.show(io::IO, ks::KalmanSmoother) =
    print(io, "KalmanSmoother(", ks.model, ", T = ", length(ks.t), ", observed = ", ks.observed,
          ", mode = ", ks.mode, ")")

"""
    KalmanResult

Output of [`eks`](@ref).
- `xbar` (`K × T`), `P` (`K × K × T`), `sd` (`K × T`): smoothed means,
  covariances and standard deviations.
- `covar`: joint covariance of all smoothed state errors and all measurement
  errors, `(K·T + L·T)²`. Both blocks are ordered state-major (index
  `t + T(k − 1)`), states first. This is `filtered.covar` of `Filter.EKS`.
- `xbar_filt`, `P_filt`: filtered means and covariances.
- `xbar_pre`, `P_pre`: predicted means and covariances.
- `Phi`, `Q`: per-step propagators and process-noise covariances (index `n`
  refers to the step `t[n−1] → t[n]`; entry 1 is unused).
"""
struct KalmanResult
    xbar::Matrix{Float64}
    P::Array{Float64,3}
    sd::Matrix{Float64}
    covar::Matrix{Float64}
    xbar_filt::Matrix{Float64}
    P_filt::Array{Float64,3}
    xbar_pre::Matrix{Float64}
    P_pre::Array{Float64,3}
    Phi::Array{Float64,3}
    Q::Array{Float64,3}
end

"""
    eks(ks::KalmanSmoother, β; x_init = ks.x_init) -> KalmanResult

Run the extended Kalman filter and RTS smoother for parameters `β`, starting from
the prior mean `x_init` (with covariance `ks.P_init`).
"""
function eks(ks::KalmanSmoother, β; x_init = ks.x_init)
    m, t = ks.model, ks.t
    predict = (n, x) -> predict_step(m, β, x, t[n-1], t[n]; σ = ks.σ, mode = ks.mode, nsub = ks.nsub,
                                     reltol = ks.reltol, alg = ks.alg)
    return kalman_smoother_core(predict, ks.y, ks.variances, ks.C, _fvec(x_init), ks.P_init)
end

"""
    kalman_smoother_core(predict, y, variances, C, x_init, P_init) -> KalmanResult

Filter/smoother recursions with a generic prediction function
`predict(n, x_{n−1}) -> (x_pre_n, Φ_n, Q_n)`. `y` and `variances` are `T × L`.
[`eks`](@ref) supplies the ODE-based prediction. A linear `predict` gives the
exact Kalman filter and RTS smoother, which the tests use to check the
covariance bookkeeping.
"""
function kalman_smoother_core(predict, y, variances, C, x_init, P_init)
    T, L = size(y)
    K = size(C, 2)
    KT, LT = K * T, L * T
    Id = Matrix{Float64}(I, K, K)

    xbar, xbar_pre, xbar_sm, sd_sm = zeros(K, T), zeros(K, T), zeros(K, T), zeros(K, T)
    P, P_pre, P_sm, Phi, Q = zeros(K, K, T), zeros(K, K, T), zeros(K, K, T), zeros(K, K, T), zeros(K, K, T)
    xbar_pre[:, 1] = x_init
    P_pre[:, :, 1] = P_init

    # EKF
    covar = zeros(T * (K + L), T * (K + L))
    for n in 1:T
        if n > 1
            x_pre, Φn, Qn = predict(n, xbar[:, n-1])
            Phi[:, :, n], Q[:, :, n] = Φn, Qn
            xbar_pre[:, n] = x_pre
            P_pre[:, :, n] = Φn * P[:, :, n-1] * Φn' + Qn
        end
        Σ = Matrix(Diagonal(variances[n, :]))
        Pp = P_pre[:, :, n]
        KG = (Pp * C') / (C * Pp * C' + Σ)
        xbar[:, n] = xbar_pre[:, n] + KG * (y[n, :] - C * xbar_pre[:, n])
        IKC = Id - KG * C
        P[:, :, n] = IKC * Pp * IKC' + KG * Σ * KG'
        _update_covariance_ekf!(covar, n, KG, P[:, :, n], Σ, n > 1 ? Phi[:, :, n] : Id, C, K, L, T)
    end

    # RTS smoother
    xbar_sm[:, T] = xbar[:, T]
    P_sm[:, :, T] = P[:, :, T]
    sd_sm[:, T] = sqrt.(diag(P[:, :, T]))
    order = vcat(KT+1:KT+LT, 1:KT)
    covar = covar[order, order]
    existing = T * (K + L)
    covar = [covar zeros(existing, (T - 1) * K); zeros((T - 1) * K, existing + (T - 1) * K)]
    for n in 1:T-1
        Tn = T - n
        G = P[:, :, Tn] * Phi[:, :, Tn+1]' * tryinv(P_pre[:, :, Tn+1])
        xbar_sm[:, Tn] = xbar[:, Tn] + G * (xbar_sm[:, Tn+1] - xbar_pre[:, Tn+1])
        P_sm[:, :, Tn] = P[:, :, Tn] + G * (P_sm[:, :, Tn+1] - P_pre[:, :, Tn+1]) * G'
        sd_sm[:, Tn] = sqrt.(max.(diag(P_sm[:, :, Tn]), 0.0))
        _update_covariance_eks!(covar, n, G, P_sm[:, :, Tn], Phi[:, :, Tn+1], K, L, T)
    end

    # regroup: smoothed states (state-major, time order), then measurement errors (state-major)
    reordered_L = vec((1:L)' .+ L .* (0:T-1))
    reordered_K = vec(T * (L + K) .+ (1:K)' .+ K .* (T-2:-1:-1))
    final = vcat(reordered_K, reordered_L)
    return KalmanResult(xbar_sm, P_sm, sd_sm, covar[final, final], xbar, P, xbar_pre, P_pre, Phi, Q)
end

# Filter.update_covariance_EKF: layout [filtered errors, time-major (K·T); measurement errors,
# time-major (L·T)].
function _update_covariance_ekf!(covar, n, KG, P, Σ, Φ, C, K, L, T)
    TK = T * K
    if n == 1
        covar[1:K, 1:K] = P
        covar[1:K, TK .+ (1:L)] = KG * Σ
        covar[TK .+ (1:L), 1:K] = (KG * Σ)'
        covar[TK .+ (1:L), TK .+ (1:L)] = Σ
        return covar
    end
    factor = (I - KG * C) * Φ
    prev, cur = 1:K*(n-1), K * (n - 1) .+ (1:K)
    U = covar[prev, K*(n-2) .+ (1:K)]
    covar[prev, cur] = U * factor'
    covar[cur, prev] = covar[prev, cur]'
    covar[cur, cur] = P
    wcols = TK .+ (1:L*(n-1))
    covar[cur, wcols] = factor * covar[K*(n-2) .+ (1:K), wcols]
    covar[wcols, cur] = covar[cur, wcols]'
    ecur = TK + L * (n - 1) .+ (1:L)
    covar[cur, ecur] = KG * Σ
    covar[ecur, cur] = covar[cur, ecur]'
    covar[ecur, ecur] = Σ
    return covar
end

# Filter.update_covariance_EKS: layout [measurement errors (L·T); filtered errors (K·T);
# smoothed errors appended backwards in time, one block per step].
function _update_covariance_eks!(covar, n, G, P_sm, Φ, K, L, T)
    existing = T * (K + L)
    obs = 1:T*L
    W = covar[obs, existing - K*(n+1) .+ (1:K)]
    S = covar[obs, existing + K*(n-2) .+ (1:K)]
    rows = existing - K .+ (1:K*n)
    Tc = covar[rows, existing + K*(n-2) .+ (1:K)]
    new = existing + K * (n - 1) .+ (1:K)
    covar[obs, new] = W + S * G' - W * Φ' * G'
    covar[new, obs] = covar[obs, new]'
    covar[rows, new] = Tc * G'
    covar[new, rows] = covar[rows, new]'
    covar[new, new] = P_sm
    return covar
end

"""
    augment_covariances(covar, variances_sm, t_data, t_gm, K, L)

Extend the joint covariance from [`eks`](@ref) (measurement errors on `t_gm`) to
measurement errors on the full data grid `t_data`, as
`EGM.augment_covariances`:
- the state block is kept;
- cross-covariances are placed at the data points that belong to `t_gm`;
- the measurement block becomes `diag(vec(variances_sm))` (`variances_sm` is
  `T_data × L`, from the spline smoother).

Result size: `(K·Tg + L·T_data)²`.
"""
function augment_covariances(covar, variances_sm, t_data, t_gm, K, L)
    T, Tg = length(t_data), length(t_gm)
    subsample = findall(in(t_gm), t_data)
    length(subsample) == Tg || throw(ArgumentError("t_gm must be a subset of t_data"))
    subsamples = vec(subsample .+ T .* (0:L-1)')
    KTg = K * Tg
    n = KTg + L * T
    aug = zeros(n, n)
    aug[1:KTg, 1:KTg] = covar[1:KTg, 1:KTg]
    aug[1:KTg, KTg .+ subsamples] = covar[1:KTg, KTg+1:end]
    aug[KTg .+ subsamples, 1:KTg] = covar[KTg+1:end, 1:KTg]
    aug[KTg+1:end, KTg+1:end] = Diagonal(vec(variances_sm))
    return aug
end
