# Phase 0: the reference files exist, are complete and internally consistent.

const REQUIRED_KEYS = ("meta", "data", "settings", "system", "spline", "smoother", "filter",
                       "ic", "steps", "iterations", "gm", "final", "selfcheck")

@testset "Phase 0: reference files" begin
    available = filter(has_reference, REFERENCE_CASES)
    if isempty(available)
        @warn "No reference files in $REFDIR; run matlab/export_reference.m first."
        @test_skip "reference files present"
    end

    for case in available
        @testset "$case" begin
            ref = load_reference(case)
            @test all(k -> haskey(ref, k), REQUIRED_KEYS)

            d, s, it = ref["data"], ref["system"], ref["iterations"]
            K, P = asint(s["K"]), asint(s["P"])
            T, L = asint(d["T"]), asint(d["L"])
            Tg = length(asvec(d["t_gm"]))
            n = asint(it["n"])
            M = ref["meta"]["M"]

            # data
            @test dims(d["traces"], 2) == (T, L)
            @test length(asvec(d["t"])) == T
            @test all(in(1:K), asvec(d["observed"]))
            @test all(in(asvec(d["t"])), asvec(d["t_gm"]))

            # system
            @test length(ascells(s["states"])) == K
            @test length(asvec(s["k0"])) == P
            @test dims(s["g"], 2) == (K * Tg, P)
            @test dims(s["h"], 2) == (Tg, K)
            @test dims(s["df"], 2) == (K * Tg, K)
            @test dims(s["dg"], 4) == (K, P, Tg, K)
            @test dims(s["dh"], 2) == (K * Tg, K)
            @test dims(s["aux_tight"], 2) == (T, K * (K + 1))
            @test size(M) == (K, K)
            @test s["aux_initial_Phi"] ≈ M

            # splines / smoother
            @test length(ascells(ref["spline"]["B"])) == L
            @test dims(ref["smoother"]["smoothed"], 2) == (Tg, L)
            @test dims(ref["smoother"]["variances_sm"], 2) == (T, L)
            @test dims(ref["smoother"]["variances_gm"], 2) == (Tg, L)

            # prediction steps
            st = ref["steps"]
            @test dims(st["Q"], 3) == (K, K, Tg - 1)
            @test dims(st["Phi"], 3) == (K, K, Tg - 1)
            @test dims(st["out"], 3) == (21, K * (K + 1), Tg - 1)

            # iterations
            @test n ≥ 1
            @test dims(it["beta"], 2) == (n + 1, P)
            @test dims(it["xbar"], 3) == (K, Tg, n)
            @test dims(it["P"], 4) == (K, K, Tg, n)
            @test dims(it["covar_aug"], 3) == (K * Tg + L * T, K * Tg + L * T, n)
            @test dims(it["V"], 3) == (K * (Tg - 2), K * (Tg - 2), n)
            @test dims(it["design"], 3) == (K * (Tg - 2), P, n)
            @test all(isfinite, it["beta"])

            # IC selection
            ic = ref["ic"]
            @test size(ic["conditions"], 2) == K
            @test asvec(ic["chosen"]) ≈ asvec(ref["filter"]["x_init_chosen"])

            # MATLAB-side consistency checks
            sc = ref["selfcheck"]
            @test asscalar(sc["qp_max_relerr"]) < 1e-8
            if Bool(asscalar(sc["performed"]))
                @test asscalar(sc["beta_relerr"]) < 1e-10
                @test asscalar(sc["icmean_relerr"]) < 1e-10
            end
        end
    end
end
