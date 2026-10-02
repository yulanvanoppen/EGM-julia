# Synthetic data (Generator.m): cell-specific parameters, integrated trajectories,
# and measurements with additive and multiplicative noise.

"""
    generate_data(model, t; β = model.β0, x0 = model.x0, observed = 1:K, N = 1,
                  D = nothing, D_mult = 0.0, lognormal = false, error_std = 0.05,
                  error_const = 0.0, rng = Random.default_rng(), reltol = 1e-8,
                  alg = Tsit5()) -> NamedTuple

Simulate measurements of `N` cells. Each step is as in EGM-v2's `Generator`:
1. **Cell parameters.** Cell `i` has parameters `βᵢ ~ N(β, D)`. By default
   `D = Diagonal((D_mult β)²)`, so `D_mult = 0` gives identical cells. With
   `lognormal = true`, the log-normal distribution with the same mean and
   covariance is used: `LD = log(1 + D ./ (ββᵀ))`, `Lb = log β − diag(LD)/2`.
2. **Trajectories.** Each cell is integrated from `x0`.
3. **Measurements.** The observed states are perturbed as
   `y = ε_add + x (1 + ε_mul)`, with `ε_add ~ N(0, error_const²)` and
   `ε_mul ~ N(0, error_std²)`.

Returns `(; traces, truth, β, t, observed, noisevar)`:
- `traces`: `T × L × N` measurements (`traces[:, :, 1]` for a single cell);
- `truth`: `T × K × N` noise-free trajectories;
- `β`: `N × P` cell parameters;
- `noisevar`: `T × L × N` measurement variances.
"""
function generate_data(m::ODEModel, t; β = m.β0, x0 = m.x0, observed = 1:m.K, N::Integer = 1,
                       D = nothing, D_mult = 0.0, lognormal::Bool = false, error_std = 0.05,
                       error_const = 0.0, rng = Random.default_rng(), reltol = 1e-8, alg = Tsit5())
    K, P = m.K, m.P
    t = _fvec(t)
    T = length(t)
    b = _fvec(β)
    observed = collect(Int, observed)
    Dm = D === nothing ? Matrix(Diagonal((D_mult .* b) .^ 2)) : Matrix{Float64}(D)
    βs = zeros(N, P)
    if lognormal
        LD = log.(1 .+ Dm ./ (b * b'))
        Lb = log.(b) .- diag(LD) ./ 2
        C = iszero(LD) ? zeros(P, P) : Matrix(cholesky(Symmetric(LD)).L)
        for i in 1:N
            βs[i, :] = exp.(Lb .+ C * randn(rng, P))
        end
    else
        C = iszero(Dm) ? zeros(P, P) : Matrix(cholesky(Symmetric(Dm)).L)
        for i in 1:N
            βs[i, :] = b .+ C * randn(rng, P)
        end
    end
    truth = zeros(T, K, N)
    for i in 1:N
        truth[:, :, i] = integrate(m, βs[i, :], x0, t; reltol = reltol, alg = alg)
    end
    x = truth[:, observed, :]
    add = error_const .* randn(rng, size(x))
    mul = error_std .* randn(rng, size(x))
    traces = add .+ x .* (1 .+ mul)
    noisevar = error_const^2 .+ error_std^2 .* x .^ 2
    return (; traces, truth, β = βs, t, observed, noisevar)
end
