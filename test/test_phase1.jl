# Phase 1: model layer.
#   A. Self-contained checks (always run): inputs, affine check, Jacobians vs.
#      finite differences, stacked layouts, Φ vs. finite-difference sensitivities.
#   B. Comparison with the MATLAB reference files (when present).

using EGM: ODEModel, integrate, integrate_variational, LinearInterpolant


@testset "Phase 1: model layer" begin

    @testset "LinearInterpolant" begin
        u = LinearInterpolant([0, 1, 3], [0.0, 2.0, 0.0])
        @test u(0.5) ≈ 1.0
        @test u(1.0) ≈ 2.0
        @test u(2.0) ≈ 1.0
        @test u(-1.0) == 0.0                                             # constant extrapolation
        @test u(5.0) == 0.0
        @test EGM.ForwardDiff.derivative(u, 0.5) ≈ 2.0
        @test EGM.ForwardDiff.derivative(u, 2.0) ≈ -1.0
        @test EGM.breakpoints(u) == [0.0, 1.0, 3.0]
        @test_throws ArgumentError LinearInterpolant([0, 0, 1], [1, 2, 3])
    end

    @testset "model construction checks" begin
        @test_throws ArgumentError ODEModel((x, β, t) -> [β[1]^2 * x[1]];                # not affine in β
                                            states = ["x"], parameters = ["a"], x0 = [1.0], β0 = [1.0])
        @test_throws ArgumentError ODEModel((x, β, t) -> [β[1] * x[1]];                  # wrong output length
                                            states = ["x", "y"], parameters = ["a"], x0 = [1.0, 1.0], β0 = [1.0])
        @test_throws ArgumentError ODEModel((x, β, t) -> [β[1] * x[1]];                  # wrong M size
                                            states = ["x"], parameters = ["a"], x0 = [1.0], β0 = [1.0],
                                            M = ones(2, 2))
        m = higgins_selkov()
        @test m.M == Matrix(I, 2, 2)                                     # default M = I
    end

    for (label, build, T_end) in [("HS", higgins_selkov, 10.0), ("GLV", generalized_lv, 10.0),
                                  ("STAT5", stat5, 60.0)]
        @testset "$label" begin
            m = build()
            K, P = m.K, m.P
            x = m.x0 .+ 0.3 .* (1:K) ./ K
            β = 1.3 .* m.β0
            t = 7.0

            # decomposition and Jacobians
            @test EGM.rhs(m, x, β, t) ≈ EGM.g(m, x, t) * β + EGM.h(m, x, t)
            @test isapprox(EGM.dfdx(m, x, β, t), fd_jacobian(y -> EGM.rhs(m, y, β, t), x); rtol = 1e-6)
            @test isapprox(EGM.dhdx(m, x, t), fd_jacobian(y -> EGM.h(m, y, t), x); rtol = 1e-6, atol = 1e-8)
            @test isapprox(reshape(EGM.dgdx(m, x, t), K * P, K),
                           fd_jacobian(y -> vec(EGM.g(m, y, t)), x); rtol = 1e-6, atol = 1e-8)

            # stacked trajectory layouts agree with pointwise evaluation
            X = permutedims(hcat(x, 1.1 .* x, 0.9 .* x))                 # 3×K
            ts = [1.0, 2.0, 3.0]
            rows = 2 .+ 3 .* (0:K-1)
            @test EGM.g_stacked(m, X, ts)[rows, :] ≈ EGM.g(m, X[2, :], ts[2])
            @test EGM.df_stacked(m, X, ts, β)[rows, :] ≈ EGM.dfdx(m, X[2, :], β, ts[2])
            @test EGM.dh_stacked(m, X, ts)[rows, :] ≈ EGM.dhdx(m, X[2, :], ts[2])
            @test EGM.dg_array(m, X, ts)[:, :, 3, :] ≈ EGM.dgdx(m, X[3, :], ts[3])
            @test EGM.h_traj(m, X, ts)[2, :] ≈ EGM.h(m, X[2, :], ts[2])
            @test EGM.rhs_traj(m, X, ts, β)[3, :] ≈ EGM.rhs(m, X[3, :], β, ts[3])

            # integration and variational system
            tgrid = collect(range(0.0, T_end; length = 6))
            tight = (reltol = 1e-11, abstol = 1e-13)
            Xv, Φ = integrate_variational(m, m.β0, m.x0, tgrid; tight...)
            @test isapprox(Xv, integrate(m, m.β0, m.x0, tgrid; tight...); rtol = 1e-8)
            @test Φ[:, :, 1] ≈ I
            Φfd = fd_jacobian(x0 -> integrate(m, m.β0, x0, tgrid; tight...)[end, :], m.x0; ε = 1e-4)
            @test isapprox(Φ[:, :, end], Φfd; rtol = 1e-5)

            _, ΦM = integrate_variational(m, m.β0, m.x0, tgrid; Φ0 = m.M, tight...)
            @test isapprox(ΦM[:, :, end], Φ[:, :, end] * m.M; rtol = 1e-7)

            A = EGM.auxiliary_layout(Xv, Φ)
            @test size(A) == (length(tgrid), K * (K + 1))
            @test A[end, K+1:end] == vec(Φ[:, :, end])

            # process-noise diffusion
            @test EGM.diffusion(m, x, 0.1) ≈ m.M * Diagonal(0.1 .* x)
        end
    end

    for case in filter(has_reference, REFERENCE_CASES)
        @testset "$case vs MATLAB" begin
            ref = load_reference(case)
            s, d = ref["system"], ref["data"]
            m = CASE_MODELS[case]()
            K, P = m.K, m.P

            # model definition matches the MATLAB model file
            @test m.states == String.(ascells(s["states"]))
            @test m.parameters == String.(ascells(s["parameters_variable"]))
            @test m.x0 ≈ asvec(s["x0"])
            @test m.β0 ≈ asvec(s["k0"])
            @test m.M ≈ ref["meta"]["M"]

            # decomposition and Jacobians at the exported test points
            X = s["eval_states"]
            t = asvec(s["eval_times"])
            β = asvec(s["eval_beta"])
            Tg = length(t)
            tol = (rtol = 1e-10, atol = 1e-12)
            @test isapprox(EGM.g_stacked(m, X, t), s["g"]; tol...)
            @test isapprox(EGM.h_traj(m, X, t), s["h"]; tol...)
            @test isapprox(EGM.rhs_traj(m, X, t, β), s["rhs"]; tol...)
            @test isapprox(EGM.df_stacked(m, X, t, β), s["df"]; tol...)
            @test isapprox(EGM.dh_stacked(m, X, t), s["dh"]; tol...)
            @test isapprox(EGM.dg_array(m, X, t), reshape(s["dg"], K, P, Tg, K); tol...)

            # integration (MATLAB: CVODES via IQMtools)
            # IQMtools only receives reltol, so CVODES runs at IQM's default absolute
            # tolerance. This limits the reference to ~1e-5 relative accuracy even at
            # reltol = 1e-10 (the Julia solution is converged to ~1e-11; see
            # test/diagnostics/phase1_report.jl). The tolerances below reflect the
            # accuracy of the reference, not of the Julia code.
            tt = asvec(d["t"])
            tight = (reltol = 1e-10, abstol = 1e-12)
            Xj = integrate(m, m.β0, m.x0, tt; tight...)
            @test isapprox(Xj, s["traj_tight"]; rtol = 5e-5)
            @test isapprox(Xj, s["traj_default"]; rtol = 2e-4)

            # auxiliary system in MATLAB-compatible mode (Φ starts at M)
            Xa, Φa = integrate_variational(m, m.β0, m.x0, tt; Φ0 = m.M, tight...)
            @test isapprox(EGM.auxiliary_layout(Xa, Φa), s["aux_tight"]; rtol = 5e-5)
        end
    end
end
