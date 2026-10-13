# Interactive smoothing of simulated data from the bifunctional two-component system (TCS),
# with the settings of test_TCS.m: states 3, 5 and 6 observed at t = 0, 1, 3, 5, 10, …, 50,
# 5% multiplicative noise, and initial knots at t = 3 and 25.
#
# In a REPL in the package folder:
#     using Pkg; Pkg.activate("examples"); Pkg.instantiate()
#     include("examples/smoothing_app_TCS.jl")

using GLMakie, EGM, Random

model = compile_iqm(load_iqm(joinpath(@__DIR__, "..", "benchmarks", "models", "model_TCS.txt");
                             fixed = ["k2", "k3", "k6", "k8"]))
observed = [3, 5, 6]
t = [0, 1, 3, 5:5:50...]

# 1. Many cells: 30 cells with 10% cell-to-cell parameter variability
cells = generate_data(model, t; N = 30, D_mult = 0.1, observed = observed, error_std = 0.05,
                      x0 = model.x0 .+ 1e-8, rng = MersenneTwister(1))
out = smoothing_app(t, cells.traces; state_names = model.states[observed], knots = [3, 25])
println("knots (time units): ", out.knots_time)

# 2. One trajectory, as in test_TCS.m, followed by EGM (or press "Finish without estimation")
#single = generate_data(model, t; observed = observed, error_std = 0.05, x0 = model.x0 .+ 1e-8,
#                        #rng = MersenneTwister(2))
k0 = model.β0
result = egm(model, t, cells.traces[:, :, 1], observed, 0.5 .* k0; interactive = true, knots = [0, 3, 25],
              x_init = [0, 0, 0.2, 2.5, 0, 0] .+ eps(), lb = 0.1 .* k0, ub = 10 .* k0,
              prior_mean = [1, 1, 1, 1e-8], prior_prec = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 25.0], ic_scan = [4])
 if result isa EGMResult
     println("EGM estimate ", model.parameters, ": ", result.β, "   (true: ", k0, ")")
 else
     println("smoothing only; knots (time units): ", result.knots_time)
 end


## Using user-supplied data
#out = smoothing_app(t, traces; state_names = ["RR", "RRp", "RRpDNA"], knots = [3, 25])
## Here t is the vector of time points and traces is a T × L matrix for one trajectory, or a T × L × N array for N cells. 
## knots is optional: without it, the automatic placement provides the starting knots.

## Using Matlab-generated data
# using MAT
# d = matread("benchmarks/datasets/test_TCS.mat")
# y = d["traces"][1]              # seed 1: 11 × 3 matrix
# out = smoothing_app(t, y; state_names = model.states[observed])