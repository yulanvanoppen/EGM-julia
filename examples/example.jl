# Example: infer the parameters of the Higgins–Selkov oscillator from noisy
# measurements of one of its two states (P), with an uncertain initial condition
# for the hidden state (S). Mirrors example.m of EGM-v2.
#
# Run from the package folder:   julia --project=examples examples/example.jl
# (in a REPL: using Pkg; Pkg.activate("examples"); Pkg.instantiate(); include("examples/example.jl"))

using EGM, LinearAlgebra, Random, Printf

# Model: dS/dt = r − c S P²,  dP/dt = c S P² − k P,  with c = 0.95 fixed and β = (r, k)
c = 0.95
f(x, β, t) = [β[1] - c * x[1] * x[2]^2,
              c * x[1] * x[2]^2 - β[2] * x[2]]
model = ODEModel(f; name = "higgins_selkov", states = ["S", "P"], parameters = ["r", "k"],
                 x0 = [1.0, 1.0], β0 = [1.0, 1.0])
# (equivalently: model = load_iqm("model_HS.txt"; fixed = ["c"]))

# Synthetic data: P measured at t = 0, 0.5, …, 10 with 2% multiplicative noise
t = 0:0.5:10
data = generate_data(model, t; observed = [2], error_std = 0.02, rng = MersenneTwister(1))
y = data.traces[:, :, 1]                                               # 21 × 1

# Extended gradient matching from a poor initial guess, with a 50% error in S(0)
result = egm(model, t, y, [2], [0.5, 0.5]; x_init = [1.5, 1.0], σ = 0.01)

# Trajectory matching for comparison
result_tm = tm(model, t, y, [2], [0.5, 0.5]; x_init = [1.5, 1.0])

se = sqrt.(diag(result.β_cov))
println("\nparameter   true      EGM (± s.e.)            TM")
for (p, name) in enumerate(model.parameters)
    @printf("%-9s  %.4f    %.4f ± %.4f        %.4f\n", name, model.β0[p], result.β[p], se[p], result_tm.β[p])
end
@printf("\nEGM: %d E-M iterations, %.2f s smoothing + %.2f s inference\n", result.iterations, result.time...)
@printf("estimated S(0): EGM %.3f, TM %.3f (true %.3f)\n", result.init[1], result_tm.x0[1], model.x0[1])

# result.fitted (on t) and result.fitted_fine (on result.t_fine) hold the model fits for
# plotting, e.g. with Plots.jl:
#   using Plots
#   plot(result.t_fine, result.fitted_fine; label = ["S" "P"])
#   scatter!(t, y; label = "P data")
