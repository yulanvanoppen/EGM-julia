# Phase 3 diagnostic report: relative errors of the smoother against MATLAB, per
# stage, plus the noise-likelihood values reached by Julia and fmincon.
#
#     julia --project=. test/diagnostics/phase3_report.jl

using EGM, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "reference.jl"))

rel(a, b) = norm(a .- b) / max(norm(b), eps())
line(label, a, b) = @printf("    %-22s rel %.2e\n", label, rel(a, b))

for case in filter(has_reference, REFERENCE_CASES)
    ref = load_reference(case)
    sp, sm, d = ref["spline"], ref["smoother"], ref["data"]
    set = ref["settings"]["smoothing"]
    L = asint(d["L"])
    traces = reshape(d["traces"], :, L)
    B = [Matrix{Float64}(b) for b in ascells(sp["B"])]
    v_init = reshape(sm["variances_init"], :, L)
    start_ref = reshape(sm["noise_start_step1"], L, 2)
    c_step1 = hcat(asvec(sm["sigma2_step1"]), asvec(sm["tau2_step1"]))
    println("\n=== $case ===")

    println("  single FWLS step (MATLAB inputs):")
    for k in 1:L
        y = traces[:, k]
        _, δ1 = EGM.wls_coefficients(B[k], y, v_init[:, k])
        δ1_ref = asvec(ascells(sm["delta_step1"])[k])
        pred = B[k] * δ1_ref
        design = hcat(ones(length(pred)), pred .^ 2)
        c = EGM.noise_mle(start_ref[k, :], pred, y)
        fj, fm = EGM.noise_objective(c, pred, y), EGM.noise_objective(c_step1[k, :], pred, y)
        println("   state $k")
        line("delta", δ1, δ1_ref)
        line("lsqnonneg start", EGM.nnls2(design, (pred .- y) .^ 2), start_ref[k, :])
        @printf("    %-22s julia [%.6e, %.6e]  matlab [%.6e, %.6e]  rel %.2e\n", "noise (σ², τ²)",
                c..., c_step1[k, :]..., rel(c, c_step1[k, :]))
        @printf("    %-22s julia %.12g  matlab %.12g  (julia − matlab = %.2e)\n", "neg. log-likelihood",
                fj, fm, fj - fm)
    end

    sb = SplineBases(asvec(d["t"]), asvec(d["t_gm"]), [asvec(k) for k in ascells(sp["knots"])];
                     t_fine = asvec(d["t_fine"]))
    r = smooth(sb, traces; niter = asint(set["niter"]), tol = asscalar(set["tol"]),
               positive = Bool(asscalar(set["positive"])))
    println("  full smoother (Julia: $(r.iterations) iterations):")
    δ_ref = [asvec(x) for x in ascells(sm["delta"])]
    for k in 1:L
        println("   state $k")
        line("delta", r.delta[k], δ_ref[k])
        line("(σ², τ²)", [r.sigma2[k], r.tau2[k]], [asvec(sm["sigma2"])[k], asvec(sm["tau2"])[k]])
        @printf("    %-22s julia [%.6e, %.6e]  matlab [%.6e, %.6e]\n", "", r.sigma2[k], r.tau2[k],
                asvec(sm["sigma2"])[k], asvec(sm["tau2"])[k])
        line("lincomb", r.lincomb[k], ascells(sm["lincomb_data"])[k])
        line("var_delta", r.var_delta[k], ascells(sm["var_delta"])[k])
    end
    line("smoothed", r.smoothed, reshape(sm["smoothed"], :, L))
    line("dsmoothed", r.dsmoothed, reshape(sm["dsmoothed"], :, L))
    line("smoothed_fine", r.smoothed_fine, reshape(sm["smoothed_fine"], :, L))
    line("dsmoothed_fine", r.dsmoothed_fine, reshape(sm["dsmoothed_fine"], :, L))
    line("variances_sm", r.variances_sm, reshape(sm["variances_sm"], :, L))
    line("variances_gm", r.variances_gm, reshape(sm["variances_gm"], :, L))
end
