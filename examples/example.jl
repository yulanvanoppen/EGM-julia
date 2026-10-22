# Example: infer the parameters of the Higgins–Selkov oscillator from noisy measurements
# of one of its two states (P), with an uncertain initial condition for the hidden state
# (S), for 10 simulated datasets. Compares EGM with trajectory matching (TM), and plots
# the estimates and, for one dataset, the data and both fits.
#
# Needs GLMakie for the plots (included in the `examples` environment):
#     using Pkg; Pkg.activate("path/to/EGM-julia/examples"); Pkg.instantiate()
#     include("path/to/EGM-julia/examples/example.jl")

using GLMakie, EGM, LinearAlgebra, Random, Printf

# --- Model: dS/dt = r − c S P²,  dP/dt = c S P² − k P,  with c = 0.95 fixed and β = (r, k) ---
c = 0.95
f(x, β, t) = [β[1] - c * x[1] * x[2]^2,
              c * x[1] * x[2]^2 - β[2] * x[2]]
model = ODEModel(f; name = "higgins_selkov", states = ["S", "P"], parameters = ["r", "k"],
                 x0 = [1.0, 1.0], β0 = [1.0, 1.0])
# (equivalently: model = load_iqm("model_HS.txt"; fixed = ["c"]))

# --- Settings ---------------------------------------------------------------------------------
t = 0:0.5:10                     # measurement times
noise = 0.02                     # 2% multiplicative measurement noise
seeds = 1:10                     # one dataset per seed; change these for other datasets
β_init = [0.5, 0.5]              # starting guess for β (truth: model.β0 = [1, 1])
x_init = [1.5, 1.0]              # starting guess for the initial conditions (50% error in S(0))
egm_settings = (; σ = 0.07, niter_gm = 20)   # EGM's process noise and maximum number of E-M iterations

relerr(β) = norm(β .- model.β0) / norm(model.β0)
median_(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

# --- Estimate for every dataset -----------------------------------------------------------------
# The random number generator is seeded explicitly, so each seed always gives the same dataset.
runs = []
for seed in seeds
    data = generate_data(model, t; observed = [2], error_std = noise, rng = MersenneTwister(seed))
    y = data.traces[:, :, 1]                                           # 21 × 1 measurements of P
    t_egm = @elapsed r_egm = egm(model, t, y, [2], β_init; x_init = x_init, egm_settings...)
    t_tm = @elapsed r_tm = tm(model, t, y, [2], β_init; x_init = x_init)
    push!(runs, (; seed, y, truth = data.truth[:, :, 1], egm = r_egm, tm = r_tm, t_egm, t_tm))
end

err_egm = [relerr(r.egm.β) for r in runs]
err_tm = [relerr(r.tm.β) for r in runs]

println("\n seed |  EGM: r       k       error  iter |  TM: r       k       error | S(0): EGM   TM")
for (r, e1, e2) in zip(runs, err_egm, err_tm)
    @printf(" %4d | %8.4f %7.4f %7.3f %5d | %8.4f %7.4f %7.3f | %9.3f %6.3f\n", r.seed, r.egm.β..., e1,
            r.egm.iterations, r.tm.β..., e2, r.egm.init[1], r.tm.x0[1])
end
@printf("\nmedian relative error: EGM %.3f, TM %.3f\n", median_(err_egm), median_(err_tm))
n_cap = count(r -> r.egm.iterations == egm_settings.niter_gm, runs)
n_cap > 0 && println("EGM reached the maximum of $(egm_settings.niter_gm) iterations for $n_cap of $(length(runs)) datasets")
if length(runs) > 1
    @printf("mean time per dataset (without the first, which includes compilation): EGM %.2f s, TM %.2f s\n",
            sum(r.t_egm for r in runs[2:end]) / (length(runs) - 1), sum(r.t_tm for r in runs[2:end]) / (length(runs) - 1))
end

# --- Diagnostics -------------------------------------------------------------------------------
# How EGM's estimate evolved for the first dataset, and whether the result depends on the
# starting guess (it should not, if the data determine β well).
r1 = runs[1]
println("\nEGM iterates for seed $(r1.seed) (row 1 = after the initial-condition search):")
display(round.(r1.egm.β_history; digits = 4))
alt = egm(model, t, r1.y, [2], [2.0, 2.0]; x_init = x_init, egm_settings...)
@printf("same data, start β = [2, 2]: EGM β = [%.4f, %.4f]\n", alt.β...)

# --- Figure 1: estimates over all datasets ---------------------------------------------------------
fig1 = Figure(size = (1100, 450))
ax1 = Axis(fig1[1, 1]; xlabel = "r", ylabel = "k", title = "Estimates for $(length(runs)) datasets")
scatter!(ax1, [r.egm.β[1] for r in runs], [r.egm.β[2] for r in runs]; label = "EGM", markersize = 12)
scatter!(ax1, [r.tm.β[1] for r in runs], [r.tm.β[2] for r in runs]; label = "TM", marker = :diamond, markersize = 12)
scatter!(ax1, [model.β0[1]], [model.β0[2]]; label = "true", marker = :star5, markersize = 22, color = :black)
scatter!(ax1, [β_init[1]], [β_init[2]]; label = "start", marker = :xcross, markersize = 16, color = :gray50)
axislegend(ax1; position = :lt)

ax2 = Axis(fig1[1, 2]; xlabel = "dataset (seed)", ylabel = "relative error ‖β − β₀‖ / ‖β₀‖",
           title = "Relative error per dataset", xticks = collect(seeds))
scatterlines!(ax2, collect(seeds), err_egm; label = "EGM")
scatterlines!(ax2, collect(seeds), err_tm; label = "TM", marker = :diamond)
axislegend(ax2; position = :lt)
display(GLMakie.Screen(), fig1)

# --- Figure 2: data and fits for one dataset ------------------------------------------------------
show_run = argmin(err_egm)        # dataset with the most accurate EGM estimate; e.g. argmax(err_egm) for the worst
r = runs[show_run]
fig2 = Figure(size = (1100, 450))
Label(fig2[0, 1:2], @sprintf("Dataset %d:  EGM β = [%.3f, %.3f],  TM β = [%.3f, %.3f],  true β = [1, 1]",
                             r.seed, r.egm.β..., r.tm.β...); fontsize = 16)
# EGM's fit is the model integrated with its final β from its smoothed initial state r.egm.init.
# For comparison, also shown: the states EGM used for gradient matching (spline smooth of P,
# Kalman smooth of S), and the model with EGM's β started from the true initial state.
egm_true_x0 = integrate(model, r.egm.β, model.x0, r.egm.t_fine)
for (k, name) in enumerate(model.states)
    ax = Axis(fig2[1, k]; xlabel = "time", ylabel = name, title = k == 2 ? "$name (measured)" : "$name (not measured)")
    lines!(ax, t, r.truth[:, k]; label = "true", color = :gray70, linewidth = 4)
    k == 2 && scatter!(ax, t, r.y[:, 1]; label = "data", color = :black)
    lines!(ax, r.egm.t_fine, r.egm.fitted_fine[:, k]; label = "EGM fit", linewidth = 2, color = Cycled(1))
    scatter!(ax, r.egm.t_gm, r.egm.gm.X[:, k]; label = "EGM smoothed states", color = Cycled(1), marker = :utriangle,
             markersize = 8)
    lines!(ax, r.egm.t_fine, egm_true_x0[:, k]; label = "EGM β from true x₀", color = Cycled(1), linestyle = :dot,
           linewidth = 2)
    lines!(ax, r.tm.t_fine, r.tm.fitted_fine[:, k]; label = "TM fit", linewidth = 2, linestyle = :dash, color = Cycled(2))
    k == 2 && axislegend(ax; position = :rt, framevisible = false)
end
@printf("\ndataset %d: initial conditions used for the fits  EGM [%.3f, %.3f]   TM [%.3f, %.3f]   true [%.3f, %.3f]\n",
        r.seed, r.egm.init..., r.tm.x0..., model.x0...)
display(GLMakie.Screen(), fig2)

# save("example_estimates.png", fig1); save("example_fits.png", fig2)    # to keep the figures
