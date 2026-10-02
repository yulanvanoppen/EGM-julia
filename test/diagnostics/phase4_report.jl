# Phase 4 diagnostic report: relative errors of the Kalman filter/smoother against MATLAB.
#
#     julia --project=. test/diagnostics/phase4_report.jl

using EGM, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "reference.jl"))
include(joinpath(@__DIR__, "..", "models.jl"))
include(joinpath(@__DIR__, "..", "helpers.jl"))

rel(a, b) = norm(a .- b) / max(norm(b), eps())
line(label, a, b) = @printf("    %-34s rel %.2e\n", label, rel(a, b))

for case in filter(has_reference, REFERENCE_CASES)
    ref = load_reference(case)
    d, sm, f, ic = ref["data"], ref["smoother"], ref["filter"], ref["ic"]
    it, st = ref["iterations"], ref["steps"]
    m = CASE_MODELS[case]()
    K, L = m.K, asint(d["L"])
    mode = Bool(asscalar(ref["meta"]["patched"])) ? :default : :compat
    t, t_gm = asvec(d["t"]), asvec(d["t_gm"])
    Tg = length(t_gm)
    y_gm = reshape(d["traces"], :, L)[findall(in(t_gm), t), :]
    σ = asscalar(f["sigma"])
    ks = KalmanSmoother(m, t_gm, y_gm, reshape(sm["variances_gm"], :, L), Int.(asvec(d["observed"]));
                        σ = σ, P_CV = asscalar(ref["settings"]["filter"]["P_CV"]),
                        x_init = asvec(ic["x_init_start"]), mode = mode)
    println("\n=== $case (mode $mode) ===")
    line("P_init", ks.P_init, f["P_init"])

    out = reshape(st["out"], 21, K * (K + 1), Tg - 1)
    Q_ref, Φ_ref = reshape(st["Q"], K, K, Tg - 1), reshape(st["Phi"], K, K, Tg - 1)
    x0s, tspan, tout = reshape(st["x0"], K, Tg - 1), reshape(st["tspan"], 2, Tg - 1), reshape(st["tout"], 21, Tg - 1)
    Qff, Φff, Qj, Φj, xj = similar(Q_ref), similar(Φ_ref), similar(Q_ref), similar(Φ_ref), zeros(K, Tg - 1)
    for n in 1:Tg-1
        Φraw = cat([reshape(out[i, K+1:end, n], K, K) for i in 1:21]...; dims = 3)
        Qff[:, :, n] = EGM.process_noise(tout[:, n], out[:, 1:K, n], Φraw, σ, I)
        Φff[:, :, n] = mode === :default ? Φraw[:, :, end] / m.M : Φraw[:, :, end]
        xj[:, n], Φj[:, :, n], Qj[:, :, n] = predict_step(m, asvec(st["beta"]), x0s[:, n], tspan[1, n], tspan[2, n];
                                                          σ = σ, mode = mode)
    end
    println("  4a prediction steps")
    line("Q from MATLAB integration", Qff, Q_ref)
    line("Φ from MATLAB integration", Φff, Φ_ref)
    line("x_pre, Julia integration", xj, out[end, 1:K, :])
    line("Φ, Julia integration", Φj, Φ_ref)
    line("Q, Julia integration", Qj, Q_ref)

    n_it = asint(it["n"])
    β_hist = reshape(it["beta"], n_it + 1, :)
    v_sm = reshape(sm["variances_sm"], :, L)
    for i in unique([1, n_it])
        r = eks(ks, β_hist[i, :]; x_init = asvec(ic["chosen"]))
        covar_raw, covar_aug = slice_iter(it["covar_raw"], 2, i), slice_iter(it["covar_aug"], 2, i)
        println("  4b/4c smoother, iteration $i")
        line("xbar", r.xbar, slice_iter(it["xbar"], 2, i))
        line("P", r.P, slice_iter(it["P"], 3, i))
        line("sd", r.sd, slice_iter(it["sd"], 2, i))
        line("covar (state block)", r.covar[1:K*Tg, 1:K*Tg], covar_raw[1:K*Tg, 1:K*Tg])
        line("covar (state × measurement)", r.covar[1:K*Tg, K*Tg+1:end], covar_raw[1:K*Tg, K*Tg+1:end])
        line("augmented, MATLAB input", augment_covariances(covar_raw, v_sm, t, t_gm, K, L), covar_aug)
        line("augmented, Julia run", augment_covariances(r.covar, v_sm, t, t_gm, K, L), covar_aug)
    end
end
