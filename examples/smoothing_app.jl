# Interactive smoothing app: choose B-spline knots by hand for one or more trajectories.
#
# In a REPL in the package folder:
#     using Pkg; Pkg.activate("examples"); Pkg.instantiate()      # first time: installs GLMakie (a few minutes)
#     include("examples/smoothing_app.jl")
#
# A window opens; Julia waits until you press Finish (or close the window).
# Controls:
#   State(s)                  edit all states together, or one state
#   Automatic knot placement  knots from the placement heuristic
#   #Knots                    replace the knots by equidistant ones
#   Manual knot selector      left click adds a knot, right click removes the nearest one
#   #Cells                    number of trajectories shown (several cells only)
#   Smooth / Live             smooth now / smooth after every change
#   Finish                    return the knots and the smoothing result
#                             (from egm: "Finish and run estimation" or "Finish without estimation")

using GLMakie, EGM, Random

c = 0.95
f(x, β, t) = [β[1] - c * x[1] * x[2]^2, c * x[1] * x[2]^2 - β[2] * x[2]]
model = ODEModel(f; name = "higgins_selkov", states = ["S", "P"], parameters = ["r", "k"],
                 x0 = [1.0, 1.0], β0 = [1.0, 1.0])
t = 0:0.5:10

# 1. Many cells (GMGTS-style data): 50 cells with 10% parameter variability, both states measured
cells = generate_data(model, t; N = 50, D_mult = 0.1, error_std = 0.03, rng = MersenneTwister(1))
out = smoothing_app(t, cells.traces; state_names = model.states)
println("knots (time units): ", out.knots_time)
# out.smoothing.smoothed (time × state × cell), out.smoothing.sigma2, out.smoothing.tau2, …

# 2. One trajectory, used directly for EGM (only P measured); egm opens the app itself
single = generate_data(model, t; observed = [2], error_std = 0.02, rng = MersenneTwister(2))
result = egm(model, t, single.traces[:, :, 1], [2], [0.5, 0.5]; x_init = [1.5, 1.0], σ = 0.01, interactive = true)
if result isa EGMResult
    println("EGM estimate: ", result.β)
else                                                                  # "Finish without estimation" or closed
    println("smoothing only; knots (time units): ", result.knots_time)
end
