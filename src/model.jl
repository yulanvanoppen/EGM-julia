# Phase 1: model layer.
#
# Replaces System.m (IQMtools parsing, symbolic decomposition, CVODES MEX files)
# and the auxiliary model files. A model is a plain Julia function
#
#     f(x, β, t) -> AbstractVector       (length K)
#
# that is affine in the free parameters β (length P):  f(x, β, t) = g(x, t) β + h(x, t).
# Fixed parameters and input functions are captured by the function itself.
# `f` must be written generically (no Float64 type assertions), so that
# ForwardDiff can differentiate it.

"""
    ODEModel(f; states, parameters, x0, β0, M = I, tstops = Float64[], name = "", check = true)

ODE model `dx/dt = f(x, β, t)` with `f` affine in the free parameters `β`.

# Arguments
- `f(x, β, t)`: right-hand side, returning a vector of length `K = length(states)`.
- `states`, `parameters`: names of the states and of the free parameters, in the
  order used by `x` and `β`.
- `x0`, `β0`: nominal initial conditions and parameter values.
- `M`: `K×K` noise-mixing matrix. The process-noise diffusion used by the Kalman
  smoother is `M * Diagonal(σ .* x)`. Default: the identity.
- `tstops`: time points where `f` is not smooth in `t`, such as the breakpoints of
  a piecewise linear input. The ODE solver steps exactly onto these.
- `name`: optional label.
- `check`: verify at random points that `f` is affine in `β` and returns a
  vector of the right length.
"""
struct ODEModel{F}
    f::F
    K::Int
    P::Int
    states::Vector{String}
    parameters::Vector{String}
    x0::Vector{Float64}
    β0::Vector{Float64}
    M::Matrix{Float64}
    tstops::Vector{Float64}
    name::String
end

function ODEModel(f; states, parameters, x0, β0, M = I, tstops = Float64[], name = "", check = true)
    K, P = length(states), length(parameters)
    length(x0) == K || throw(ArgumentError("length(x0) = $(length(x0)) ≠ number of states $K"))
    length(β0) == P || throw(ArgumentError("length(β0) = $(length(β0)) ≠ number of parameters $P"))
    Mmat = M isa UniformScaling ? Matrix{Float64}(M, K, K) : Matrix{Float64}(M)
    size(Mmat) == (K, K) || throw(ArgumentError("M must be $K×$K, got $(size(Mmat))"))
    model = ODEModel(f, K, P, String.(collect(states)), String.(collect(parameters)),
                     _fvec(x0), _fvec(β0), Mmat,
                     sort(Float64.(collect(tstops))), String(name))
    check && check_affine(model)
    return model
end

Base.show(io::IO, m::ODEModel) =
    print(io, "ODEModel(", isempty(m.name) ? "" : repr(m.name) * ", ", "K = ", m.K, ", P = ", m.P, ")")

"""
    check_affine(model; ntrials = 3, t = 0.0, rtol = 1e-10)

Throw an `ArgumentError` unless `f(x, β, t) ≈ g(x, t) β + h(x, t)` at random
states around `x0` and random parameter vectors.
"""
function check_affine(m::ODEModel; ntrials = 3, t = 0.0, rtol = 1e-10)
    fx = m.f(m.x0, m.β0, t)
    length(fx) == m.K || throw(ArgumentError("f returns a vector of length $(length(fx)), expected $(m.K)"))
    for _ in 1:ntrials
        x = m.x0 .+ (1 .+ abs.(m.x0)) .* rand(m.K)
        β = (1 .+ abs.(m.β0)) .* (2 .* rand(m.P) .- 1)
        direct = m.f(x, β, t)
        decomposed = g(m, x, t) * β + h(m, x, t)
        isapprox(direct, decomposed; rtol = rtol, atol = rtol * (1 + norm(direct))) ||
            throw(ArgumentError("f is not affine in β: f(x, β, t) ≠ g(x, t)β + h(x, t) at x = $x, β = $β"))
    end
    return true
end

# --- Pointwise evaluation --------------------------------------------------------

"Right-hand side `f(x, β, t)`."
rhs(m::ODEModel, x, β, t) = m.f(x, β, t)

"Parameter-independent part `h(x, t) = f(x, 0, t)` (length `K`)."
h(m::ODEModel, x, t) = m.f(x, zeros(m.P), t)

"""
    g(model, x, t)

Parameter factor (`K×P`): column `p` equals `f(x, eₚ, t) − h(x, t)`. This is exact
because `f` is affine in `β`.
"""
function g(m::ODEModel, x, t)
    h0 = h(m, x, t)
    G = similar(h0, m.K, m.P)
    e = zeros(m.P)
    for p in 1:m.P
        e[p] = 1.0
        G[:, p] = m.f(x, e, t) - h0
        e[p] = 0.0
    end
    return G
end

"State Jacobian `∂f/∂x` (`K×K`)."
dfdx(m::ODEModel, x, β, t) = ForwardDiff.jacobian(y -> m.f(y, β, t), x)

"State Jacobian of `h` (`K×K`)."
dhdx(m::ODEModel, x, t) = ForwardDiff.jacobian(y -> h(m, y, t), x)

"State derivatives of `g` (`K×P×K`): entry `(k, p, l)` is `∂g_kp/∂x_l`."
dgdx(m::ODEModel, x, t) = reshape(ForwardDiff.jacobian(y -> vec(g(m, y, t)), x), m.K, m.P, m.K)

"""
    diffusion(model, x, σ)

Process-noise diffusion matrix `M * Diagonal(σ .* x)` (`K×K`).
"""
diffusion(m::ODEModel, x, σ) = m.M * Diagonal(σ .* x)

# --- Trajectory evaluation in the MATLAB layouts ------------------------------------
# X is T×K (row i = state at time t[i]). "Stacked" means K vertical blocks of T rows
# (state-major, row (k-1)T + i), matching System.g/df/dh in EGM-v2.

_rows(X) = (view(X, i, :) for i in axes(X, 1))

"`T×K` matrix of right-hand sides along a trajectory."
rhs_traj(m::ODEModel, X::AbstractMatrix, t, β) =
    permutedims(reduce(hcat, [rhs(m, x, β, ti) for (x, ti) in zip(_rows(X), t)]))

"`T×K` matrix of `h` along a trajectory."
h_traj(m::ODEModel, X::AbstractMatrix, t) =
    permutedims(reduce(hcat, [h(m, x, ti) for (x, ti) in zip(_rows(X), t)]))

"`(K·T)×P`: row `(k-1)T + i` is row `k` of `g(X[i, :], t[i])`."
function g_stacked(m::ODEModel, X::AbstractMatrix, t)
    T = size(X, 1)
    out = zeros(m.K * T, m.P)
    for (i, (x, ti)) in enumerate(zip(_rows(X), t))
        out[i .+ T .* (0:m.K-1), :] = g(m, x, ti)
    end
    return out
end

"`(K·T)×K`: row `(k-1)T + i`, column `j` is `∂f_k/∂x_j` at `(X[i, :], t[i])`."
function df_stacked(m::ODEModel, X::AbstractMatrix, t, β)
    T = size(X, 1)
    out = zeros(m.K * T, m.K)
    for (i, (x, ti)) in enumerate(zip(_rows(X), t))
        out[i .+ T .* (0:m.K-1), :] = dfdx(m, x, β, ti)
    end
    return out
end

"`(K·T)×K`: row `(k-1)T + i`, column `l` is `∂h_k/∂x_l` at `(X[i, :], t[i])`."
function dh_stacked(m::ODEModel, X::AbstractMatrix, t)
    T = size(X, 1)
    out = zeros(m.K * T, m.K)
    for (i, (x, ti)) in enumerate(zip(_rows(X), t))
        out[i .+ T .* (0:m.K-1), :] = dhdx(m, x, ti)
    end
    return out
end

"`K×P×T×K`: entry `(k, p, i, l)` is `∂g_kp/∂x_l` at `(X[i, :], t[i])`."
function dg_array(m::ODEModel, X::AbstractMatrix, t)
    T = size(X, 1)
    out = zeros(m.K, m.P, T, m.K)
    for (i, (x, ti)) in enumerate(zip(_rows(X), t))
        out[:, :, i, :] = dgdx(m, x, ti)
    end
    return out
end

# --- Integration -------------------------------------------------------------------

_fvec(v) = Vector{Float64}(vec(collect(v)))                          # accept vectors, 1×n matrices, ranges

function _solve(prob, t, m; alg, reltol, abstol, kwargs...)
    tst = filter(τ -> first(t) < τ < last(t), m.tstops)
    sol = SciMLBase.solve(prob, alg; saveat = t, reltol = reltol, abstol = abstol, tstops = tst, kwargs...)
    SciMLBase.successful_retcode(sol) ||
        error("ODE integration failed with retcode $(sol.retcode) for $(m)")
    return sol
end

"""
    integrate(model, β, x0, t; alg = Tsit5(), reltol = 1e-6, abstol = reltol / 1000, kwargs...)

Integrate the model from `x0` at `t[1]` and return the `T×K` states at the time
points `t`. Any SciML ODE algorithm can be passed as `alg`, for example
`Rodas5P()` from OrdinaryDiffEqRosenbrock or `CVODE_BDF()` from Sundials.
Other keywords are passed to `solve`.
"""
function integrate(m::ODEModel, β, x0, t; alg = Tsit5(), reltol = 1e-6, abstol = reltol / 1000, kwargs...)
    βv = _fvec(β)
    prob = SciMLBase.ODEProblem((x, _, τ) -> m.f(x, βv, τ), _fvec(x0), (first(t), last(t)))
    sol = _solve(prob, t, m; alg, reltol, abstol, kwargs...)
    return permutedims(reduce(hcat, sol.u))
end

# Callable wrapper x ↦ f(x, β, t) with a mutable time, so that one ForwardDiff
# configuration can be reused for every Jacobian evaluation during a solve.
mutable struct _RHSAt{F,B}
    f::F
    β::B
    t::Float64
end
(r::_RHSAt)(y) = r.f(y, r.β, r.t)

# Augmented system [x; vec(Φ); vec(S)] with dΦ/dt = J Φ and, if P > 0, dS/dt = J S + g(x, t)
# (sensitivities with respect to β; ∂f/∂β = g because f is affine in β).
# The fast in-place path is used for plain Float64 states; other number types (for example
# dual numbers from implicit solvers with automatic differentiation) take a generic path.
function _augmented_rhs(m::ODEModel, βv::Vector{Float64}, with_β::Bool)
    K, P = m.K, with_β ? m.P : 0
    nΦ = K * K
    fat = _RHSAt(m.f, βv, 0.0)
    cfg = ForwardDiff.JacobianConfig(fat, zeros(K))
    J = zeros(K, K)
    function rhs!(du, u, _, τ)
        if eltype(u) === Float64 && eltype(du) === Float64 && τ isa Float64
            x = view(u, 1:K)
            fat.t = τ
            du[1:K] .= m.f(x, βv, τ)
            ForwardDiff.jacobian!(J, fat, x, cfg)
            mul!(reshape(view(du, K+1:K+nΦ), K, K), J, reshape(view(u, K+1:K+nΦ), K, K))
            if P > 0
                dS = reshape(view(du, K+nΦ+1:K+nΦ+K*P), K, P)
                mul!(dS, J, reshape(view(u, K+nΦ+1:K+nΦ+K*P), K, P))
                dS .+= g(m, x, τ)
            end
        else
            x = u[1:K]
            Jg = dfdx(m, x, βv, τ)
            du[1:K] .= m.f(x, βv, τ)
            du[K+1:K+nΦ] .= vec(Jg * reshape(u[K+1:K+nΦ], K, K))
            if P > 0
                du[K+nΦ+1:end] .= vec(Jg * reshape(u[K+nΦ+1:end], K, P) + g(m, x, τ))
            end
        end
        return nothing
    end
    return rhs!
end

"""
    integrate_variational(model, β, x0, t; Φ0 = I, alg = Tsit5(), reltol = 1e-6,
                          abstol = reltol / 1000, kwargs...) -> (X, Φ)

Integrate the model together with its variational equation
`dΦ/dt = ∂f/∂x(x(t), β, t) Φ`, starting at `Φ(t[1]) = Φ0`. Returns `X` (`T×K`)
and `Φ` (`K×K×T`). With the default `Φ0 = I`, `Φ[:, :, i]` is the propagator
`Φ(t[i], t[1])`. Passing `Φ0 = model.M` reproduces EGM-v2's `System.auxiliary`
for models with a non-identity initial Φ (MATLAB-compatible mode).
"""
function integrate_variational(m::ODEModel, β, x0, t; Φ0 = I, alg = Tsit5(), reltol = 1e-6,
                               abstol = reltol / 1000, kwargs...)
    K = m.K
    βv = _fvec(β)
    Φ0m = Φ0 isa UniformScaling ? Matrix{Float64}(Φ0, K, K) : Matrix{Float64}(Φ0)
    u0 = vcat(_fvec(x0), vec(Φ0m))
    prob = SciMLBase.ODEProblem(_augmented_rhs(m, βv, false), u0, (first(t), last(t)))
    sol = _solve(prob, t, m; alg, reltol, abstol, kwargs...)
    U = reduce(hcat, sol.u)                                          # K(K+1)×T
    return permutedims(U[1:K, :]), reshape(U[K+1:end, :], K, K, :)
end

"""
    integrate_sensitivities(model, β, x0, t; alg = Tsit5(), reltol = 1e-6,
                            abstol = reltol / 1000, kwargs...) -> (X, Sx, Sβ)

Integrate the model with its forward sensitivities. Returns:
- `X` (`T×K`): the states;
- `Sx = ∂x(t)/∂x0` (`K×K×T`);
- `Sβ = ∂x(t)/∂β` (`K×P×T`).

Because `f` is affine in `β`, `∂f/∂β = g(x, t)`, so the sensitivities follow
`dSβ/dt = J Sβ + g`. Used by trajectory matching ([`tm`](@ref)).
"""
function integrate_sensitivities(m::ODEModel, β, x0, t; alg = Tsit5(), reltol = 1e-6,
                                 abstol = reltol / 1000, kwargs...)
    K, P = m.K, m.P
    βv = _fvec(β)
    u0 = vcat(_fvec(x0), vec(Matrix{Float64}(I, K, K)), zeros(K * P))
    prob = SciMLBase.ODEProblem(_augmented_rhs(m, βv, true), u0, (first(t), last(t)))
    sol = _solve(prob, t, m; alg, reltol, abstol, kwargs...)
    U = reduce(hcat, sol.u)
    return permutedims(U[1:K, :]), reshape(U[K+1:K+K*K, :], K, K, :), reshape(U[K+K*K+1:end, :], K, P, :)
end

"""
    auxiliary_layout(X, Φ)

Arrange the output of [`integrate_variational`](@ref) as EGM-v2's
`System.auxiliary`: a `T×K(K+1)` matrix with the states first, followed by `Φ`
in column-major order (Ph11, Ph21, …, PhK1, Ph12, …).
"""
auxiliary_layout(X, Φ) = hcat(X, permutedims(reshape(Φ, size(Φ, 1) * size(Φ, 2), :)))

# --- Inputs ------------------------------------------------------------------------

"""
    LinearInterpolant(ts, ys)

Piecewise linear function through the points `(ts[i], ys[i])`, for time-dependent
inputs such as STAT5's `input(time)`. It is constant outside `[ts[1], ts[end]]`
(MATLAB's `interp1` returns NaN there). It works with ForwardDiff dual numbers.
Pass `breakpoints(u)` as the model's `tstops`.
"""
struct LinearInterpolant
    ts::Vector{Float64}
    ys::Vector{Float64}
    function LinearInterpolant(ts, ys)
        length(ts) == length(ys) || throw(ArgumentError("ts and ys must have equal length"))
        length(ts) ≥ 2 || throw(ArgumentError("need at least two points"))
        issorted(ts; lt = ≤) || throw(ArgumentError("ts must be strictly increasing"))
        new(Float64.(collect(ts)), Float64.(collect(ys)))
    end
end

function (u::LinearInterpolant)(τ)
    ts, ys = u.ts, u.ys
    τ ≤ ts[1] && return zero(τ) + ys[1]                              # constant extrapolation
    τ ≥ ts[end] && return zero(τ) + ys[end]
    i = searchsortedlast(ts, τ)
    w = (τ - ts[i]) / (ts[i+1] - ts[i])
    return (1 - w) * ys[i] + w * ys[i+1]
end

"Breakpoints of a [`LinearInterpolant`](@ref), for use as `tstops`."
breakpoints(u::LinearInterpolant) = copy(u.ts)
