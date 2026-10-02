# Phase 6 diagnostic report: IC search, E-M iterations and final estimates against MATLAB.
#
#     julia --project=. test/diagnostics/phase6_report.jl

using EGM, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "reference.jl"))
include(joinpath(@__DIR__, "..", "models.jl"))
include(joinpath(@__DIR__, "..", "helpers.jl"))

rel(a, b) = norm(a .- b) / max(norm(b), eps())
fmt(v) = join([@sprintf("%.6g", x) for x in v], ", ")

for case in filter(has_reference, REFERENCE_CASES)
    ref = load_reference(case)
    d, ic, it, fin = ref["data"], ref["ic"], ref["iterations"], ref["final"]
    m = CASE_MODELS[case]()
    L = asint(d["L"])
    t, traces, observed = asvec(d["t"]), reshape(d["traces"], :, L), Int.(asvec(d["observed"]))
    settings = ref_inference_settings(ref, case, m.P)
    β0 = asvec(ref["meta"]["beta0"])
    n_ref = asint(it["n"])
    β_hist_ref = reshape(it["beta"], n_ref + 1, m.P)
    println("\n=== $case ===")

    r = egm_inference(m, ref_smoothing(ref), traces, observed, β0; settings...)
    @printf("  from MATLAB smoothing (%.1f s):\n", r.time[2])
    @printf("    IC choice: julia %d, matlab %d of %d; IC errors rel %.2e; IC betas rel %.2e\n",
            r.ic_index, asint(ic["chosen_index"]), size(r.ic_candidates, 1),
            rel(r.ic_errors, asvec(ic["errors"])), rel(r.ic_betas, ic["betas"]))
    @printf("    iterations: julia %d, matlab %d\n", r.iterations, n_ref)
    for i in 1:max(size(r.β_history, 1), n_ref + 1)
        jl = i ≤ size(r.β_history, 1) ? fmt(r.β_history[i, :]) : "-"
        ml = i ≤ n_ref + 1 ? fmt(β_hist_ref[i, :]) : "-"
        @printf("      β[%2d] julia [%s]   matlab [%s]\n", i, jl, ml)
    end
    @printf("    final β rel %.2e; fitted rel %.2e\n", rel(r.β, asvec(fin["beta"])), rel(r.fitted, fin["fitted"]))

    sset = ref["settings"]["smoothing"]
    knots = USER_KNOTS[case]
    rf = egm(m, t, traces, observed, β0; knots = knots, autoknots = knots === nothing, t_gm = asvec(d["t_gm"]),
             niter_sm = asint(sset["niter"]), tol_sm = asscalar(sset["tol"]), settings...)
    @printf("  complete run from raw data (%.1f s + %.1f s):\n", rf.time...)
    @printf("    IC choice: julia %d, matlab %d; iterations: julia %d, matlab %d\n",
            rf.ic_index, asint(ic["chosen_index"]), rf.iterations, n_ref)
    @printf("    final β julia [%s]\n            matlab [%s]   rel %.2e\n", fmt(rf.β), fmt(asvec(fin["beta"])),
            rel(rf.β, asvec(fin["beta"])))
end
