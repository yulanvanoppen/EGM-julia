# Phase 5: gradient-matching step.
#   A. Self-contained checks (always run): Savitzky–Golay weights and exactness,
#      nearest_spd, box-constrained QP (KKT conditions), GLS with prior, and exact
#      parameter recovery from noise-free states and derivatives.
#   B. Comparison with the MATLAB reference files (when present), from MATLAB's inputs
#      at the first and last E-M iteration: differentiation matrix, state/derivative
#      estimates, GLS design, residual covariance, and the updated parameters.

using EGM: GradientMatching, gm_step, qp_gls

"KKT conditions of min ½xᵀHx + fᵀx s.t. lb ≤ x ≤ ub."
function box_kkt(H, f, lb, ub, x; tol = 1e-9)
    g = H * x + f
    s = tol * (1 + norm(f) + opnorm(H) * norm(x))
    all(lb .- s .≤ x .≤ ub .+ s) || return false
    for i in eachindex(x)
        atl, atu = abs(x[i] - lb[i]) ≤ s, abs(x[i] - ub[i]) ≤ s
        if atl
            g[i] ≥ -s || return false
        elseif atu
            g[i] ≤ s || return false
        else
            abs(g[i]) ≤ s || return false
        end
    end
    return true
end


@testset "Phase 5: gradient matching" begin

    @testset "Savitzky–Golay differentiation" begin
        @test EGM.sgolay_derivative_weights(3, 7) ≈ [22, -67, -58, 0, 58, 67, -22] ./ 252
        @test EGM.sgolay_derivative_weights(3, 5) ≈ [1, -8, 0, 8, -1] ./ 12
        @test EGM.sgolay_derivative_weights(1, 3) ≈ [-0.5, 0, 0.5]

        t = collect(0:0.5:10)
        T = length(t)
        D = EGM.sgolay_matrix(t)
        cubic(t) = 1 - t + 0.3t^2 - 0.05t^3
        dcubic(t) = -1 + 0.6t - 0.15t^2
        @test (D * cubic.(t))[3:T-2] ≈ dcubic.(t[3:T-2])                 # 7- and 5-point cubic rows
        quad(t) = 2 + t - 0.4t^2
        @test (D * quad.(t))[[2, T - 1]] ≈ (1 .- 0.8 .* t)[[2, T - 1]]   # central differences
        @test D * (3 .* t .+ 1) ≈ fill(3.0, T)                           # all rows exact for lines
        tn = [0.0, 1, 3, 4, 6, 7, 8]                                     # non-uniform: one-sided ends
        Dn = EGM.sgolay_matrix(tn)
        @test Dn[1, 1:2] ≈ [-1, 1] ./ 1 && Dn[end, end-1:end] ≈ [-1, 1] ./ 1
        @test Dn[2, 1:3] ≈ [-0.5, 0, 0.5] ./ 1.5
    end

    @testset "nearest_spd" begin
        A = [4.0 1.0; 1.0 3.0]
        @test EGM.nearest_spd(A) ≈ A
        B = [1.0 2.0; 2.0 1.0]                                           # indefinite
        S = EGM.nearest_spd(B)
        @test S ≈ S' && isposdef(S)
        @test EGM.nearest_spd(fill(-1.0, 1, 1)) == fill(eps(), 1, 1)
    end

    @testset "box_qp" begin
        rng = MersenneTwister(6)
        for trial in 1:200
            n = rand(rng, 1:6)
            A = randn(rng, n, n)
            H = A * A' + 1e-3I
            f = 3 .* randn(rng, n)
            lb = -2 .* rand(rng, n)
            ub = lb .+ 0.1 .+ 3 .* rand(rng, n)
            trial % 5 == 0 && (ub[1] = Inf)
            trial % 7 == 0 && (lb[end] = -Inf)
            x = EGM.box_qp(H, f, lb, ub)
            @test box_kkt(H, f, lb, ub, x)
        end
        H, f = [2.0 0.5; 0.5 1.0], [-1.0, 0.3]
        @test EGM.box_qp(H, f, [-Inf, -Inf], [Inf, Inf]) ≈ -(H \ f)       # inactive bounds
    end

    @testset "qp_gls" begin
        rng = MersenneTwister(7)
        X = randn(rng, 20, 3)
        Y = randn(rng, 10, 2)
        A = randn(rng, 20, 20)
        V = A * A' + I
        P = inv(V)
        β, SS = qp_gls(X, Y, V, fill(-Inf, 3), fill(Inf, 3))
        @test β ≈ (X' * P * X) \ (X' * P * vec(Y))
        @test SS ≈ dot(vec(Y) - X * β, P * (vec(Y) - X * β))
        μ, Λ = [1.0, 2.0, 3.0], Diagonal([1.0, 0.0, 4.0])
        β, _ = qp_gls(X, Y, V, fill(-Inf, 3), fill(Inf, 3); prior_mean = μ, prior_prec = Matrix(Λ))
        @test β ≈ (X' * P * X + Λ) \ (X' * P * vec(Y) + Λ * μ)
        lb, ub = [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]
        β, _ = qp_gls(X, Y, V, lb, ub)
        H = X' * P * X
        @test box_kkt((H + H') / 2, -(X' * P * vec(Y)), lb, ub, β)
    end

    @testset "exact recovery from noise-free states" begin
        for build in (higgins_selkov, generalized_lv, stat5)
            m = build()
            t = build === stat5 ? collect(2.0:2.0:40.0) : collect(0:0.5:10)
            X = integrate(m, m.β0, m.x0, t; reltol = 1e-10, abstol = 1e-12)
            dX = EGM.rhs_traj(m, X, t, m.β0)
            design, response = EGM.gm_design(m, X, dX, t)
            V = Matrix(1.0I, size(design, 1), size(design, 1))
            β, SS = qp_gls(design, response, V, 0.1 .* m.β0, 10 .* m.β0)
            @test β ≈ m.β0
            @test SS < 1e-16
        end
    end

    for case in filter(has_reference, REFERENCE_CASES)
        @testset "$case vs MATLAB" begin
            ref = load_reference(case)
            d, sp, sm, it = ref["data"], ref["spline"], ref["smoother"], ref["iterations"]
            inf = ref["settings"]["inference"]
            m = CASE_MODELS[case]()
            K, P = m.K, m.P
            L = asint(d["L"])
            t_gm = asvec(d["t_gm"])
            Tg = length(t_gm)
            μ, Λ = ref_prior(inf, P)
            gm = GradientMatching(m, t_gm, reshape(sm["smoothed"], :, L), reshape(sm["dsmoothed"], :, L),
                                  ascells(sp["basis_gm"]), ascells(sp["dbasis_gm"]), ascells(sm["lincomb_data"]),
                                  asint(d["T"]), Int.(asvec(d["observed"]));
                                  lb = asvec(inf["lb"]), ub = asvec(inf["ub"]), prior_mean = μ, prior_prec = Λ)
            @test isapprox(gm.fdiff, ref["gm"]["fdiff"]; rtol = 1e-12)

            n_it = asint(it["n"])
            β_hist = reshape(it["beta"], n_it + 1, P)
            SS_hist = asvec(it["SS"])
            for i in unique([1, n_it])
                xbar = slice_iter(it["xbar"], 2, i)
                covar_aug = slice_iter(it["covar_aug"], 2, i)
                β_in, β_out = β_hist[i, :], β_hist[i+1, :]

                # inputs to the GLS problem
                X, dX = EGM.states_and_derivatives(gm, xbar)
                @test isapprox(X, slice_iter(it["smoothed_filtered"], 2, i); rtol = 1e-12)
                @test isapprox(dX, slice_iter(it["dsmoothed_dfiltered"], 2, i); rtol = 1e-12)
                design, response = EGM.gm_design(m, X, dX, t_gm)
                design_ref = slice_iter(it["design"], 2, i)
                response_ref = slice_iter(it["response"], 2, i)
                @test isapprox(design, design_ref; rtol = 1e-12, atol = 1e-14)
                @test isapprox(response, response_ref; rtol = 1e-12, atol = 1e-14)

                # residual covariance
                V, vXdX = EGM.residual_covariance(gm, β_in, X, covar_aug)
                V_ref = slice_iter(it["V"], 2, i)
                @test isapprox(vXdX, slice_iter(it["variances_XdX"], 2, i); rtol = 1e-9)
                @test isapprox(V, V_ref; rtol = 1e-9)

                # GLS from MATLAB's design, response and V
                β, SS = qp_gls(design_ref, response_ref, EGM.nearest_spd(V_ref), asvec(inf["lb"]), asvec(inf["ub"]);
                               prior_mean = μ, prior_prec = Λ)
                @test isapprox(β, β_out; rtol = 1e-8)
                @test isapprox(SS, SS_hist[i]; rtol = 1e-8)

                # complete step
                r = gm_step(gm, β_in, xbar, covar_aug)
                @test isapprox(r.β, β_out; rtol = 1e-7)
                @test isapprox(r.SS, SS_hist[i]; rtol = 1e-7)
            end
        end
    end
end
