# Phase 6: complete estimation.
#   A. Self-contained checks (always run): IC candidate construction, and parameter
#      recovery on synthetic Higgins–Selkov data (fully and partially observed).
#   B. Comparison with the MATLAB reference files (when present):
#      - IC candidates from MATLAB's inputs (exact);
#      - inference from MATLAB's smoothing result: IC search and E-M loop
#        (differences only from ODE solver tolerances);
#      - complete run from the raw data (adds the Phase 3 noise-optimizer differences).

using EGM: egm, egm_inference, ic_candidates, EGMResult


@testset "Phase 6: complete estimation" begin

    @testset "ic_candidates" begin
        mults = 4.0 .^ (-1:0.5:1)
        c = ic_candidates([1.0, 2.0, 3.0], [2])
        @test size(c) == (5, 3)
        @test c[:, 2] ≈ 2 .* mults && all(c[:, 1] .== 1) && all(c[:, 3] .== 3)
        c = ic_candidates([1.0, 2.0, 3.0], [1, 3])                       # first uncertain state fastest
        @test size(c) == (25, 3)
        @test c[1:5, 1] ≈ mults && all(c[1:5, 3] .≈ 3 * mults[1]) && all(c[6:10, 3] .≈ 3 * mults[2])
        c = ic_candidates([1.0, 2.0, 3.0], [1]; ties = [2 => 1])
        @test c[:, 2] == c[:, 1]
        c = ic_candidates([1.0, 2.0, 0.5, 4.0], [3, 4]; constraint = stat5_ic_constraint)
        @test all(c[:, 4] .≤ c[:, 3])
        @test ic_candidates([1.0, 2.0], Int[]) == [1.0 2.0]
    end

    @testset "synthetic Higgins–Selkov" begin
        rng = MersenneTwister(8)
        m = higgins_selkov()
        t = collect(0:0.5:10)
        truth = integrate(m, m.β0, m.x0, t; reltol = 1e-10, abstol = 1e-12)
        data = truth .* (1 .+ 0.01 .* randn(rng, size(truth)))

        r = egm(m, t, data, [1, 2], 0.5 .* m.β0)                         # fully observed
        @test r isa EGMResult
        @test 1 ≤ r.iterations ≤ 20
        @test relerr(r.β, m.β0) < 0.05
        @test size(r.fitted) == size(truth) && relerr(r.fitted, truth) < 0.05

        r = egm(m, t, data[:, 2:2], [2], 0.5 .* m.β0; x_init = [1.5, 0.8])   # only P observed
        @test relerr(r.β, m.β0) < 0.2
        @test relerr(r.fitted[:, 2], truth[:, 2]) < 0.05
        @test r.ic_index == argmin(r.ic_errors)
        @test size(r.ic_candidates) == (5, 2)
    end

    for case in filter(has_reference, REFERENCE_CASES)
        @testset "$case vs MATLAB" begin
            ref = load_reference(case)
            d, sm, ic, it, fin = ref["data"], ref["smoother"], ref["ic"], ref["iterations"], ref["final"]
            m = CASE_MODELS[case]()
            K, P = m.K, m.P
            L = asint(d["L"])
            t = asvec(d["t"])
            traces = reshape(d["traces"], :, L)
            observed = Int.(asvec(d["observed"]))
            settings = ref_inference_settings(ref, case, P)
            β0 = asvec(ref["meta"]["beta0"])
            n_ref = asint(it["n"])
            β_ref = asvec(fin["beta"])

            # IC candidates from MATLAB's inputs
            center = copy(settings.x_init)
            center[observed] = reshape(sm["smoothed"], :, L)[1, :]
            cands = ic_candidates(center, get(settings, :ic_scan, [k for k in 1:K if !(k in observed)]);
                                  ties = get(settings, :ic_ties, Pair{Int,Int}[]),
                                  constraint = get(settings, :ic_constraint, identity))
            @test isapprox(cands, ic["conditions"]; rtol = 1e-14)

            # inference from MATLAB's smoothing result
            r = egm_inference(m, ref_smoothing(ref), traces, observed, β0; settings...)
            @test r.ic_index == asint(ic["chosen_index"])
            @test isapprox(r.ic_errors, asvec(ic["errors"]); rtol = 1e-2)
            @test isapprox(r.ic_betas, ic["betas"]; rtol = 1e-2)
            @test abs(r.iterations - n_ref) ≤ 2
            @test isapprox(r.β, β_ref; rtol = 5e-3)
            @test isapprox(r.fitted, fin["fitted"]; rtol = 1e-2)

            # complete run from the raw data
            sset = ref["settings"]["smoothing"]
            knots = USER_KNOTS[case]
            rf = egm(m, t, traces, observed, β0; knots = knots, autoknots = knots === nothing,
                     t_gm = asvec(d["t_gm"]), niter_sm = asint(sset["niter"]), tol_sm = asscalar(sset["tol"]),
                     settings...)
            @test rf.ic_index == asint(ic["chosen_index"])
            @test isapprox(rf.β, β_ref; rtol = 2e-2)
        end
    end
end
