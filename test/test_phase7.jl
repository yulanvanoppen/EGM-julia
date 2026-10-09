# Phase 7: extras.
#   - forward sensitivities (used by trajectory matching) vs finite differences;
#   - trajectory matching on synthetic data;
#   - delta-method uncertainties: analytic ∂β/∂(X, dX) vs finite differences of the GLS
#     estimator, and a covariance check;
#   - data generator and multiple cells;
#   - IQM model-file parser against the hand-written test models.

using EGM: tm, generate_data, egm_cells, load_iqm, parse_iqm, compile_iqm, integrate_sensitivities,
           parameter_covariance

@testset "Phase 7: extras" begin

    @testset "forward sensitivities" begin
        for build in (higgins_selkov, stat5)
            m = build()
            t = build === stat5 ? collect(2.0:6.0:38.0) : collect(0:2.0:10)
            tight = (reltol = 1e-11, abstol = 1e-13)
            X, Sx, Sβ = integrate_sensitivities(m, m.β0, m.x0, t; tight...)
            Xv, Φ = integrate_variational(m, m.β0, m.x0, t; tight...)
            @test X ≈ Xv
            @test Sx ≈ Φ
            Sfd = fd_jacobian(β -> integrate(m, β, m.x0, t; tight...)[end, :], m.β0; ε = 1e-5)
            @test isapprox(Sβ[:, :, end], Sfd; rtol = 1e-5)
        end
    end

    @testset "trajectory matching" begin
        m = higgins_selkov()
        t = collect(0:0.5:10)
        truth = integrate(m, m.β0, m.x0, t; reltol = 1e-10, abstol = 1e-12)
        r = tm(m, t, truth, [1, 2], 0.6 .* m.β0; estimate_x0 = false, niter = 3)   # noise-free, x0 known
        @test isapprox(r.β, m.β0; rtol = 1e-4)
        r = tm(m, t, truth[:, 2:2], [2], 0.6 .* m.β0; x_init = [1.5, 0.8], niter = 3)   # P observed, x0 estimated
        @test isapprox(r.β, m.β0; rtol = 1e-3)
        @test isapprox(r.x0, m.x0; rtol = 1e-3)
        @test r.ic_index == argmin(r.ic_errors)
        data = generate_data(m, t; observed = [2], error_std = 0.01, rng = MersenneTwister(9))
        r = tm(m, t, data.traces[:, :, 1], [2], 0.6 .* m.β0; x_init = [1.5, 0.8])
        @test relerr(r.β, m.β0) < 0.2
        @test r.sigma2[1] ≥ 0 && r.tau2[1] ≥ 0
    end

    @testset "parameter uncertainties" begin
        m = higgins_selkov()
        t = collect(0:0.5:10)
        data = generate_data(m, t; observed = [2], error_std = 0.01, rng = MersenneTwister(10))
        r = egm(m, t, data.traces[:, :, 1], [2], 0.5 .* m.β0; x_init = [1.5, 0.8])
        @test size(r.β_cov) == (2, 2)
        @test r.β_cov ≈ r.β_cov' && all(eigvals(Symmetric(r.β_cov)) .≥ -1e-12)

        # analytic Jacobian of the GLS estimate vs finite differences (V fixed)
        sr = r.smoothing
        gm = GradientMatching(m, sr, [2]; lb = fill(-Inf, 2), ub = fill(Inf, 2))
        step = r.gm
        T, K = length(sr.bases.t_gm), m.K
        Vinv = inv(step.V)
        function gls(X, dX)
            design, response = EGM.gm_design(m, X, dX, sr.bases.t_gm)
            return (design' * Vinv * design) \ (design' * Vinv * vec(response))
        end
        J = EGM.beta_jacobian(gm, step)
        cols = Vector{Float64}[]
        for which in (:X, :dX), k in 1:K, j in 2:T-1
            Xp, Xm, dXp, dXm = copy(step.X), copy(step.X), copy(step.dX), copy(step.dX)
            A, B = which === :X ? (Xp, Xm) : (dXp, dXm)
            ε = 1e-6 * max(1.0, abs(A[j, k]))
            A[j, k] += ε
            B[j, k] -= ε
            push!(cols, (gls(Xp, dXp) - gls(Xm, dXm)) ./ (2ε))
        end
        @test isapprox(J, reduce(hcat, cols); rtol = 1e-5)
    end

    @testset "data generator" begin
        m = higgins_selkov()
        t = collect(0:1.0:10)
        d = generate_data(m, t; N = 4, D_mult = 0.1, observed = [2], rng = MersenneTwister(11))
        @test size(d.traces) == (11, 1, 4) && size(d.truth) == (11, 2, 4) && size(d.β) == (4, 2)
        @test d.truth[:, :, 3] ≈ integrate(m, d.β[3, :], m.x0, t; reltol = 1e-8)
        d0 = generate_data(m, t; error_std = 0.0)
        @test d0.traces[:, :, 1] == d0.truth[:, :, 1] && d0.β == permutedims(m.β0)
        dl = generate_data(m, t; N = 50, D_mult = 0.5, lognormal = true, rng = MersenneTwister(12))
        @test all(dl.β .> 0)
        @test generate_data(m, t; rng = MersenneTwister(13)).traces == generate_data(m, t; rng = MersenneTwister(13)).traces
    end

    @testset "multiple cells" begin
        m = higgins_selkov()
        t = collect(0:0.5:10)
        d = generate_data(m, t; N = 3, D_mult = 0.1, error_std = 0.01, rng = MersenneTwister(14))
        rs = egm_cells(m, t, d.traces, [1, 2], 0.5 .* m.β0)
        @test length(rs) == 3
        for i in 1:3
            @test relerr(rs[i].β, d.β[i, :]) < 0.1
        end
        rs1 = egm_cells(m, t, d.traces[:, :, 1:1], [1, 2], 0.5 .* m.β0; threaded = false)
        @test rs1[1].β ≈ rs[1].β
    end

    @testset "IQM model files" begin
        dir = joinpath(@__DIR__, "models")
        rng = MersenneTwister(15)
        for (file, fixed, build) in (("model_HS.txt", ["c"], higgins_selkov),
                                     ("model_GLV.txt", ["r"], generalized_lv),
                                     ("model_STAT5.txt", ["tauinv", "k2"], stat5))
            ref = build()
            m = load_iqm(joinpath(dir, file); fixed = fixed, M = ref.M)
            @test m.name == ref.name
            @test m.states == ref.states
            @test m.parameters == ref.parameters
            @test m.x0 ≈ ref.x0 && m.β0 ≈ ref.β0
            @test m.tstops == ref.tstops
            for _ in 1:5
                x, β, τ = ref.x0 .+ rand(rng, ref.K), ref.β0 .* (0.5 .+ rand(rng, ref.P)), 50 * rand(rng)
                @test EGM.rhs(m, x, β, τ) ≈ EGM.rhs(ref, x, β, τ)
            end
            @test EGM.dfdx(m, ref.x0, ref.β0, 3.0) ≈ EGM.dfdx(ref, ref.x0, ref.β0, 3.0)
        end
        mc = compile_iqm(load_iqm(joinpath(dir, "model_HS.txt"); fixed = ["c"]))  # direct call, no invokelatest
        @test !(mc.f isa EGM.IQMRHS)
        @test Base.invokelatest(EGM.rhs, mc, [1.0, 2.0], [1.0, 1.0], 0.0) ≈ EGM.rhs(higgins_selkov(), [1.0, 2.0], [1.0, 1.0], 0.0)
        m = load_iqm(joinpath(dir, "model_HS.txt"); fixed = ["c"], fixed_values = [0.5])
        @test EGM.rhs(m, [1.0, 2.0], [1.0, 1.0], 0.0) ≈ [1 - 0.5 * 4, 0.5 * 4 - 2]

        text = """
        ********** MODEL NAME
        toy
        ********** MODEL STATES
        d/dt(A) = k1 - k2*A   % production and decay
        d/dt(B) = flux - 0.5*B
        A(0) = 1
        B(0) = 0
        ********** MODEL PARAMETERS
        k1 = 0.1
        k2 = 0.5 {annotation}
        ********** MODEL VARIABLES
        flux = k2*A
        """
        toy = parse_iqm(text)
        @test toy.states == ["A", "B"] && toy.parameters == ["k1", "k2"] && toy.x0 == [1.0, 0.0]
        @test EGM.rhs(toy, [2.0, 1.0], [0.1, 0.5], 0.0) ≈ [0.1 - 1.0, 1.0 - 0.5]
        @test_throws ArgumentError parse_iqm(replace(text, "k1 - k2*A" => "k1 - k2^2*A"))   # not affine
        @test_throws ArgumentError parse_iqm(text; fixed = ["k3"])
    end
end
