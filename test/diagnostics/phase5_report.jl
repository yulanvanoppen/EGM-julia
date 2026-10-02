# Phase 5 diagnostic report: relative errors of the gradient-matching step against MATLAB.
#
#     julia --project=. test/diagnostics/phase5_report.jl

using EGM, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "reference.jl"))
include(joinpath(@__DIR__, "..", "models.jl"))
include(joinpath(@__DIR__, "..", "helpers.jl"))

rel(a, b) = norm(a .- b) / max(norm(b), eps())
line(label, a, b) = @printf("    %-30s rel %.2e\n", label, rel(a, b))

for case in filter(has_reference, REFERENCE_CASES)
    ref = load_reference(case)
    d, sp, sm, it = ref["data"], ref["spline"], ref["smoother"], ref["iterations"]
    inf = ref["settings"]["inference"]
    m = CASE_MODELS[case]()
    K, P, L = m.K, m.P, asint(d["L"])
    t_gm = asvec(d["t_gm"])
    pr = inf["prior"]
    μ = length(pr["mean"]) == 1 ? asscalar(pr["mean"]) : asvec(pr["mean"])
    Λ = length(pr["prec"]) == 1 ? asscalar(pr["prec"]) : reshape(pr["prec"], P, P)
    gm = GradientMatching(m, t_gm, reshape(sm["smoothed"], :, L), reshape(sm["dsmoothed"], :, L),
                          ascells(sp["basis_gm"]), ascells(sp["dbasis_gm"]), ascells(sm["lincomb_data"]),
                          asint(d["T"]), Int.(asvec(d["observed"]));
                          lb = asvec(inf["lb"]), ub = asvec(inf["ub"]), prior_mean = μ, prior_prec = Λ)
    println("\n=== $case ===")
    line("fdiff", gm.fdiff, ref["gm"]["fdiff"])
    n_it = asint(it["n"])
    β_hist = reshape(it["beta"], n_it + 1, P)
    SS_hist = asvec(it["SS"])
    for i in unique([1, n_it])
        xbar, covar_aug = slice_iter(it["xbar"], 2, i), slice_iter(it["covar_aug"], 2, i)
        println("  iteration $i")
        X, dX = EGM.states_and_derivatives(gm, xbar)
        line("X", X, slice_iter(it["smoothed_filtered"], 2, i))
        line("dX", dX, slice_iter(it["dsmoothed_dfiltered"], 2, i))
        design, response = EGM.gm_design(m, X, dX, t_gm)
        line("design", design, slice_iter(it["design"], 2, i))
        line("response", response, slice_iter(it["response"], 2, i))
        V, vXdX = EGM.residual_covariance(gm, β_hist[i, :], X, covar_aug)
        line("variances_XdX", vXdX, slice_iter(it["variances_XdX"], 2, i))
        line("V", V, slice_iter(it["V"], 2, i))
        β, SS = qp_gls(slice_iter(it["design"], 2, i), slice_iter(it["response"], 2, i),
                       EGM.nearest_spd(slice_iter(it["V"], 2, i)), asvec(inf["lb"]), asvec(inf["ub"]);
                       prior_mean = μ, prior_prec = Λ)
        line("β (GLS from MATLAB inputs)", β, β_hist[i+1, :])
        line("SS (GLS from MATLAB inputs)", SS, SS_hist[i])
        r = gm_step(gm, β_hist[i, :], xbar, covar_aug)
        line("β (complete step)", r.β, β_hist[i+1, :])
        @printf("    %-30s julia %s\n    %-30s matlab %s\n", "", round.(r.β; sigdigits = 8), "", round.(β_hist[i+1, :]; sigdigits = 8))
    end
end
