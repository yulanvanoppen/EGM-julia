# Phase 3: spline smoother with additive + multiplicative noise.
#   A. Self-contained checks (always run): NNLS, noise MLE (optimality, boundary
#      case, recovery), WLS, and a full smoothing run on synthetic data.
#   B. Comparison with the MATLAB reference files (when present), module by module:
#      single FWLS step from MATLAB's inputs (tight), fit() from MATLAB's final
#      coefficients/variances (tight), and the full converged smoother (looser,
#      since fmincon and the Julia optimizer stop at slightly different points).

using EGM: SplineBases, smooth, resolve_knots
using Random

"f(c) is not improved by small feasible perturbations of c (bound-constrained optimality)."
function locally_optimal(f, c; rel = 1e-4)
    fc = f(c)
    s = rel * max(norm(c), eps())
    for d in ([1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1])
        cp = max.(c .+ s .* d, 0.0)
        cp == c && continue
        f(cp) ≥ fc - 1e-12 * abs(fc) || return false
    end
    return true
end

@testset "Phase 3: spline smoother" begin

    @testset "helpers" begin
        @test EGM.relative_change([1.0, 0.0], [1.0, 0.0]) == 0
        @test EGM.relative_change([0.0, 0.0], [1.0, 0.0]) == Inf
        @test EGM.relative_change([0.0, 0.0], [0.0, 0.0]) == 0
        @test EGM.relative_change([3.0, 4.0], [3.0, 5.0]) ≈ 0.2
        A = [2.0 1.0; 1.0 3.0]
        @test EGM.tryinv(A) ≈ inv(A)
        @test EGM.tryinv([1.0 1.0; 1.0 1.0]) ≈ pinv([1.0 1.0; 1.0 1.0])
    end

    @testset "nnls2" begin
        rng = MersenneTwister(1)
        for trial in 1:20
            X = hcat(ones(15), rand(rng, 15) .* 3)
            y = X * [rand(rng) - 0.3, rand(rng) - 0.3] .+ 0.05 .* randn(rng, 15)
            c = EGM.nnls2(X, y)
            g = X' * (X * c .- y)                                       # KKT conditions
            @test all(≥(0), c)
            @test all(j -> c[j] > 0 ? abs(g[j]) < 1e-9 * (1 + norm(X' * y)) : g[j] ≥ -1e-9 * (1 + norm(X' * y)), 1:2)
        end
    end

    @testset "noise_mle" begin
        rng = MersenneTwister(2)
        pred = collect(range(0.5, 3; length = 2000))
        σ2, τ2 = 0.01, 0.04
        y = pred .+ sqrt.(σ2 .+ τ2 .* pred .^ 2) .* randn(rng, length(pred))
        f(c) = EGM.noise_objective(c, pred, y)
        c = EGM.noise_mle([0.05, 0.001], pred, y)
        @test isapprox(c, [σ2, τ2]; rtol = 0.25)                         # recovers the truth
        @test locally_optimal(f, c)

        # boundary optimum: residual variance decreasing in pred ⇒ τ² = 0, σ² = mean(r²)
        pred = collect(range(1, 3; length = 50))
        r2 = 0.01 .* (1.5 .- pred .^ 2 ./ 9)
        y = pred .+ sqrt.(r2)
        c = EGM.noise_mle([0.005, 0.005], pred, y)
        @test c[2] == 0
        @test isapprox(c[1], sum(r2) / length(r2); rtol = 1e-10)
        @test locally_optimal(c -> EGM.noise_objective(c, pred, y), c)
    end

    @testset "WLS coefficients" begin
        B = basis(BSpline(4, [0, 0.4, 1]), collect(range(0, 1; length = 30)))
        y = sin.(range(0, 3; length = 30))
        lincomb, δ = EGM.wls_coefficients(B, y, ones(30))
        @test δ ≈ B \ y                                                  # equal weights = ordinary LS
        @test lincomb * B ≈ I                                            # left inverse of B
    end

    @testset "synthetic smoothing run" begin
        rng = MersenneTwister(3)
        t = collect(0:0.25:10)
        truth = hcat(2 .+ sin.(t), 1 .+ 0.5 .* cos.(0.5 .* t))
        traces = truth .* (1 .+ 0.03 .* randn(rng, size(truth)))
        sb = SplineBases(t, t[1:2:end], resolve_knots(t, traces))
        r = smooth(sb, traces)
        @test 1 ≤ r.iterations ≤ 20
        @test size(r.smoothed) == (21, 2) && size(r.smoothed_fine) == (81, 2)
        @test maximum(abs, r.smoothed .- truth[1:2:end, :]) < 0.15
        @test all(r.sigma2 .≥ 0) && all(r.tau2 .≥ 0)
        @test all(v -> v ≈ v' && isposdef(Symmetric(v)), r.var_delta)
        @test r.variances_sm ≈ r.sigma2' .+ r.tau2' .* reduce(hcat, [sb.B[k] * r.delta[k] for k in 1:2]) .^ 2
    end

    for case in filter(has_reference, REFERENCE_CASES)
        @testset "$case vs MATLAB" begin
            ref = load_reference(case)
            sp, sm, d = ref["spline"], ref["smoother"], ref["data"]
            set = ref["settings"]["smoothing"]
            L = asint(d["L"])
            traces = reshape(d["traces"], :, L)
            B = [Matrix{Float64}(b) for b in ascells(sp["B"])]
            B_gm = [Matrix{Float64}(b) for b in ascells(sp["basis_gm"])]
            v_init = reshape(sm["variances_init"], :, L)
            start_ref = reshape(sm["noise_start_step1"], L, 2)
            c_step1 = hcat(asvec(sm["sigma2_step1"]), asvec(sm["tau2_step1"]))

            @test EGM.initial_variances(traces) ≈ v_init

            for k in 1:L
                y = traces[:, k]
                # single FWLS step from MATLAB's inputs
                _, δ1 = EGM.wls_coefficients(B[k], y, v_init[:, k])
                δ1_ref = asvec(ascells(sm["delta_step1"])[k])
                @test isapprox(δ1, δ1_ref; rtol = 1e-10)

                pred = B[k] * δ1_ref
                design = hcat(ones(length(pred)), pred .^ 2)
                @test isapprox(EGM.nnls2(design, (pred .- y) .^ 2), start_ref[k, :]; rtol = 1e-8)

                # fmincon (interior point) stops slightly inside the feasible region when the
                # optimum lies on a bound (σ² = 0 or τ² = 0); noise_mle reaches the bound. The
                # decisive checks are therefore optimality and a likelihood at least as good as
                # fmincon's; the parameters themselves agree only to a few percent.
                c = EGM.noise_mle(start_ref[k, :], pred, y)
                f(c) = EGM.noise_objective(c, pred, y)
                @test f(c) ≤ f(c_step1[k, :]) + 1e-9 * abs(f(c_step1[k, :]))  # at least as good as fmincon
                @test locally_optimal(f, c)
                @test isapprox(c, c_step1[k, :]; rtol = 5e-2)
                @test isapprox(design * c_step1[k, :], reshape(sm["variances_sm_step1"], :, L)[:, k]; rtol = 1e-12)
            end

            # fit() from MATLAB's converged coefficients and variances
            v_ref = reshape(sm["variances_sm"], :, L)
            δ_ref = [asvec(x) for x in ascells(sm["delta"])]
            for k in 1:L
                sm_k = max.(B_gm[k] * δ_ref[k], Bool(asscalar(set["positive"])) ? 1e-12 : -Inf)
                @test isapprox(sm_k, reshape(sm["smoothed"], :, L)[:, k]; rtol = 1e-12)
                V = EGM.tryinv(B[k]' * (Diagonal(max.(v_ref[:, k], 1e-10)) \ B[k]))
                @test isapprox(V, ascells(sm["var_delta"])[k]; rtol = 1e-10)
            end

            # full smoother with MATLAB's knots and settings. Differences come from the noise
            # parameters above (fmincon vs. exact bound-constrained optimum) and propagate mildly
            # into the weights; tolerances are set accordingly.
            sb = SplineBases(asvec(d["t"]), asvec(d["t_gm"]), [asvec(k) for k in ascells(sp["knots"])];
                             t_fine = asvec(d["t_fine"]))
            r = smooth(sb, traces; niter = asint(set["niter"]), tol = asscalar(set["tol"]),
                       positive = Bool(asscalar(set["positive"])))
            for k in 1:L
                @test isapprox(r.delta[k], δ_ref[k]; rtol = 1e-4)
                @test isapprox([r.sigma2[k], r.tau2[k]], [asvec(sm["sigma2"])[k], asvec(sm["tau2"])[k]]; rtol = 5e-2)
                @test isapprox(r.lincomb[k], ascells(sm["lincomb_data"])[k]; rtol = 5e-3)
                @test isapprox(r.var_delta[k], ascells(sm["var_delta"])[k]; rtol = 5e-2)
            end
            @test isapprox(r.smoothed, reshape(sm["smoothed"], :, L); rtol = 1e-4)
            @test isapprox(r.dsmoothed, reshape(sm["dsmoothed"], :, L); rtol = 1e-3)
            @test isapprox(r.smoothed_fine, reshape(sm["smoothed_fine"], :, L); rtol = 1e-4)
            @test isapprox(r.dsmoothed_fine, reshape(sm["dsmoothed_fine"], :, L); rtol = 1e-3)
            @test isapprox(r.variances_sm, v_ref; rtol = 5e-2)
            @test isapprox(r.variances_gm, reshape(sm["variances_gm"], :, L); rtol = 5e-2)
        end
    end
end
