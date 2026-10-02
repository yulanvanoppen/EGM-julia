# Phase 4: extended Kalman filter/smoother with joint covariances.
#   A. Self-contained checks (always run):
#      - process-noise quadrature and prediction step (incl. :default vs :compat);
#      - an exact check of the filter/smoother recursions and of the joint covariance
#        bookkeeping on a linear-Gaussian model against brute-force Gaussian
#        conditioning;
#      - augment_covariances.
#   B. Comparison with the MATLAB reference files (when present):
#      4a prediction steps (Q, Φ from MATLAB's integration: exact; full Julia step: solver
#         tolerance), 4b smoothed means/covariances, 4c joint covariance (augmentation from
#         MATLAB's input: exact; full Julia run: solver tolerance).

using EGM: KalmanSmoother, eks, predict_step, augment_covariances

blockdiag(mats...) = cat(mats...; dims = (1, 2))


"""
Brute-force smoothed means and joint covariance of (smoothed errors, measurement errors)
for the linear-Gaussian model x₁ ~ N(x_init, P0), xₙ = Φₙxₙ₋₁ + wₙ, yₙ = Cxₙ + eₙ.
Returned in the state-major layout of `KalmanResult.covar`.
"""
function linear_gaussian_reference(Φs, Qs, P0, C, Σs, y, x_init)
    K, L, T = size(P0, 1), size(C, 1), length(Σs)
    F = zeros(K * T, K * T)                                          # x (time-major) = F [δ₀; w₂; …; w_T]
    for n in 1:T
        rows = K * (n - 1) .+ (1:K)
        if n > 1
            F[rows, :] = Φs[n] * F[K*(n-2) .+ (1:K), :]
        end
        F[rows, rows] += I
    end
    mean_x = zeros(K, T)
    mean_x[:, 1] = x_init
    for n in 2:T
        mean_x[:, n] = Φs[n] * mean_x[:, n-1]
    end
    Σw = blockdiag(P0, Qs[2:T]...)
    Σe = blockdiag(Σs...)
    Cb = kron(Matrix(1.0I, T, T), C)
    Σxx = F * Σw * F'
    H = (Σxx * Cb') / (Cb * Σxx * Cb' + Σe)
    xhat = vec(mean_x) + H * (vec(permutedims(y)) - Cb * vec(mean_x))
    A = hcat(H * Cb * F - F, H)                                      # smoothed error = A [w; e]
    Σz = blockdiag(Σw, Σe)
    cov_ss = A * Σz * A'
    cov_se = A * Σz * vcat(zeros(K * T, L * T), Matrix(1.0I, L * T, L * T))
    full = [cov_ss cov_se; cov_se' Σe]
    px = [k + K * (n - 1) for n in 1:T, k in 1:K] |> vec             # state-major ← time-major
    pe = [l + L * (n - 1) for n in 1:T, l in 1:L] |> vec
    p = vcat(px, K * T .+ pe)
    return reshape(xhat, K, T), full[p, p]
end

@testset "Phase 4: Kalman filter/smoother" begin

    @testset "process noise and prediction" begin
        tout = collect(range(0, 1; length = 5))
        X = repeat([1.0 2.0], 5)
        Φ = cat([Matrix(1.0I, 2, 2) for _ in 1:5]...; dims = 3)
        @test EGM.process_noise(tout, X, Φ, 0.1, I) ≈ Diagonal(0.01 .* [1, 4])
        M = [1.0 1.0; 0.0 1.0]
        @test EGM.process_noise(tout, X, Φ, 0.1, M) ≈ M * Diagonal(0.01 .* [1, 4]) * M'

        m = higgins_selkov()
        x_pre, Φn, Qn = predict_step(m, m.β0, [1.2, 0.8], 1.0, 1.5; σ = 0.05, reltol = 1e-10)
        Xr, Φr = integrate_variational(m, m.β0, [1.2, 0.8], [1.0, 1.5]; reltol = 1e-11, abstol = 1e-13)
        @test x_pre ≈ Xr[end, :]
        @test Φn ≈ Φr[:, :, end]
        @test Qn ≈ Qn' && isposdef(Symmetric(Qn))
        _, _, Qfine = predict_step(m, m.β0, [1.2, 0.8], 1.0, 1.5; σ = 0.05, reltol = 1e-10, nsub = 400)
        @test isapprox(Qn, Qfine; rtol = 1e-3)                           # trapezoid rule converges

        s = stat5()
        x0 = [0.9, 0.3, 0.05, 0.02, 0.01, 0.005, 0.002]
        _, Φd, Qd = predict_step(s, s.β0, x0, 10.0, 12.0; σ = 0.01, mode = :default, reltol = 1e-10)
        _, Φc, Qc = predict_step(s, s.β0, x0, 10.0, 12.0; σ = 0.01, mode = :compat, reltol = 1e-10)
        @test Qd ≈ Qc
        @test Φc ≈ Φd * s.M
        @test_throws ArgumentError predict_step(s, s.β0, x0, 10.0, 12.0; σ = 0.01, mode = :other)
    end

    @testset "linear-Gaussian model: exact recursions and covariance bookkeeping" begin
        rng = MersenneTwister(4)
        K, L, T = 3, 2, 6
        C = [1.0 0 0; 0 0 1]
        Φs = vcat([Matrix(1.0I, K, K)], [I + 0.2 .* randn(rng, K, K) for _ in 2:T])      # entry 1 unused
        Qs = vcat([zeros(K, K)], [let A = 0.1 .* randn(rng, K, K); A * A' + 0.01I end for _ in 2:T])
        P0 = let A = randn(rng, K, K); A * A' + 0.5I end
        Σs = [Matrix(Diagonal(0.05 .+ 0.1 .* rand(rng, L))) for _ in 1:T]
        y = randn(rng, T, L)
        variances = permutedims(reduce(hcat, [diag(S) for S in Σs]))
        x_init = [1.0, -0.5, 2.0]

        predict = (n, x) -> (Φs[n] * x, Φs[n], Qs[n])
        r = EGM.kalman_smoother_core(predict, y, variances, C, x_init, P0)
        xhat, covar_ref = linear_gaussian_reference(Φs, Qs, P0, C, Σs, y, x_init)

        @test r.xbar ≈ xhat                                              # RTS = Gaussian conditioning
        @test isapprox(r.covar, covar_ref; rtol = 1e-10)                  # joint covariance bookkeeping
        @test r.covar ≈ r.covar'
        for n in 1:T                                                     # P blocks inside covar
            idx = n .+ T .* (0:K-1)
            @test r.P[:, :, n] ≈ r.covar[idx, idx]
        end
        @test r.sd ≈ reduce(hcat, [sqrt.(diag(r.P[:, :, n])) for n in 1:T])
    end

    @testset "augment_covariances" begin
        K, L = 2, 1
        t_data, t_gm = collect(0.0:0.5:2.0), [0.0, 1.0, 2.0]
        Tg, T = 3, 5
        A = randn(MersenneTwister(5), K * Tg + L * Tg, K * Tg + L * Tg)
        covar = A * A'
        v = collect(1.0:5.0)
        aug = augment_covariances(covar, v, t_data, t_gm, K, L)
        @test size(aug) == (K * Tg + L * T, K * Tg + L * T)
        @test aug[1:K*Tg, 1:K*Tg] == covar[1:K*Tg, 1:K*Tg]
        @test aug[1:K*Tg, K*Tg .+ [1, 3, 5]] == covar[1:K*Tg, K*Tg+1:end]
        @test all(iszero, aug[1:K*Tg, K*Tg .+ [2, 4]])
        @test aug[K*Tg+1:end, K*Tg+1:end] == Diagonal(v)
        @test_throws ArgumentError augment_covariances(covar, v, t_data, [0.0, 0.7, 2.0], K, L)
    end

    for case in filter(has_reference, REFERENCE_CASES)
        @testset "$case vs MATLAB" begin
            ref = load_reference(case)
            d, sm, f, ic = ref["data"], ref["smoother"], ref["filter"], ref["ic"]
            it, st = ref["iterations"], ref["steps"]
            m = CASE_MODELS[case]()
            K = m.K
            L = asint(d["L"])
            mode = Bool(asscalar(ref["meta"]["patched"])) ? :default : :compat
            t, t_gm = asvec(d["t"]), asvec(d["t_gm"])
            Tg = length(t_gm)
            idx = findall(in(t_gm), t)
            y_gm = reshape(d["traces"], :, L)[idx, :]
            σ = asscalar(f["sigma"])
            ks = KalmanSmoother(m, t_gm, y_gm, reshape(sm["variances_gm"], :, L), Int.(asvec(d["observed"]));
                                σ = σ, P_CV = asscalar(ref["settings"]["filter"]["P_CV"]),
                                x_init = asvec(ic["x_init_start"]), mode = mode)
            @test isapprox(ks.P_init, f["P_init"]; rtol = 1e-12)
            @test ks.C == reshape(f["C"], L, K)

            # 4a: prediction steps
            β_st = asvec(st["beta"])
            out = reshape(st["out"], 21, K * (K + 1), Tg - 1)
            Q_ref = reshape(st["Q"], K, K, Tg - 1)
            Φ_ref = reshape(st["Phi"], K, K, Tg - 1)
            x0s = reshape(st["x0"], K, Tg - 1)
            tspan = reshape(st["tspan"], 2, Tg - 1)
            tout = reshape(st["tout"], 21, Tg - 1)
            Qj, Φj, xj = similar(Q_ref), similar(Φ_ref), zeros(K, Tg - 1)
            for n in 1:Tg-1
                X = out[:, 1:K, n]
                Φraw = cat([reshape(out[i, K+1:end, n], K, K) for i in 1:21]...; dims = 3)
                @test isapprox(EGM.process_noise(tout[:, n], X, Φraw, σ, I), Q_ref[:, :, n]; rtol = 1e-12)
                Φn = mode === :default ? Φraw[:, :, end] / m.M : Φraw[:, :, end]
                @test isapprox(Φn, Φ_ref[:, :, n]; rtol = 1e-12)
                xj[:, n], Φj[:, :, n], Qj[:, :, n] = predict_step(m, β_st, x0s[:, n], tspan[1, n], tspan[2, n];
                                                                  σ = σ, mode = mode)
            end
            @test isapprox(xj, out[end, 1:K, :]; rtol = 1e-3)
            @test isapprox(Φj, Φ_ref; rtol = 1e-3)
            @test isapprox(Qj, Q_ref; rtol = 1e-3)

            # 4b/4c: smoother at the first and last E-M iteration
            n_it = asint(it["n"])
            β_hist = reshape(it["beta"], n_it + 1, :)
            v_sm = reshape(sm["variances_sm"], :, L)
            for i in unique([1, n_it])
                r = eks(ks, β_hist[i, :]; x_init = asvec(ic["chosen"]))
                covar_raw = slice_iter(it["covar_raw"], 2, i)
                covar_aug = slice_iter(it["covar_aug"], 2, i)
                @test isapprox(r.xbar, slice_iter(it["xbar"], 2, i); rtol = 1e-3)
                @test isapprox(r.P, slice_iter(it["P"], 3, i); rtol = 1e-3)
                @test isapprox(r.sd, slice_iter(it["sd"], 2, i); rtol = 1e-3)
                @test isapprox(r.covar, covar_raw; rtol = 1e-3)
                @test isapprox(augment_covariances(covar_raw, v_sm, t, t_gm, K, L), covar_aug; rtol = 1e-14)
                @test isapprox(augment_covariances(r.covar, v_sm, t, t_gm, K, L), covar_aug; rtol = 1e-3)
            end
        end
    end
end
