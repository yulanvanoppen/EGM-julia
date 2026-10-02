# Phase 2: B-splines and knot placement.
#   A. Self-contained checks (always run): basis properties, derivatives, knot helpers.
#   B. Comparison with the MATLAB reference files (when present): knots, scaled grids
#      and all basis matrices to machine precision.

using EGM: BSpline, basis, basis_diff, place_knots, resolve_knots, SplineBases


@testset "Phase 2: B-splines and knots" begin

    @testset "BSpline construction" begin
        b = BSpline(4, [0, 0.3, 0.5, 1])
        @test b.knots == [0.3, 0.5]
        @test b.card == 6
        b = BSpline(4, 4)                                                # 4 equidistant segments
        @test b.knots ≈ [0.25, 0.5, 0.75]
        @test b.card == 7
        b = BSpline(4, [0.0, 1.0])                                       # no interior knots: Bernstein basis
        @test b.card == 4
        x = [0.0, 0.2, 0.7, 1.0]
        @test basis(b, x) ≈ [(1 .- x) .^ 3  3 .* x .* (1 .- x) .^ 2  3 .* x .^ 2 .* (1 .- x)  x .^ 3]
    end

    @testset "basis properties" begin
        for knots in ([0, 0.3, 0.5, 1], [0, 0.1, 0.2, 0.6, 0.65, 1], [0, 1])
            b = BSpline(4, knots)
            x = collect(range(0, 1; length = 201))
            B, dB = basis(b, x), basis_diff(b, x)
            @test size(B) == (201, b.card)
            @test all(≥(-1e-14), B)
            @test all(sum(B; dims = 2) .≈ 1)                              # partition of unity, incl. x = 1
            @test maximum(abs, sum(dB; dims = 2)) < 1e-10                 # derivatives sum to zero
            xs = [0.05, 0.37, 0.61, 0.93]                                # derivative vs finite differences
            ε = 1e-6
            @test isapprox(basis_diff(b, xs), (basis(b, xs .+ ε) .- basis(b, xs .- ε)) ./ (2ε); rtol = 1e-6)
            p(x) = 1 - 2x + 3x^2 - 4x^3                                  # cubics are reproduced exactly
            coef = B \ p.(x)
            @test maximum(abs, B * coef .- p.(x)) < 1e-10
        end
    end

    @testset "knot helpers" begin
        @test EGM.matlab_linspace(0.0, 1.0, 5) == [0, 0.25, 0.5, 0.75, 1]
        @test EGM.normalize_knots([-1, 3, 3, 12], 0:10) ≈ [0, 0.3, 1]      # clamp, dedupe, add ends
        @test EGM.default_knots(0:0.5:10) ≈ 0:0.1:1                      # round((21 − 1)/2) + 1 = 11 knots
        @test resolve_knots(0:10, ones(11, 2); knots = [2, 5]) == [[0, 0.2, 0.5, 1], [0, 0.2, 0.5, 1]]
        @test resolve_knots(0:10, ones(11, 2); knots = [[2], [4, 6]]) == [[0, 0.2, 1], [0, 0.4, 0.6, 1]]
        @test_throws ArgumentError resolve_knots(0:10, ones(11, 2); knots = [[2]])

        t = collect(0:0.5:10)
        y = hcat(exp.(-(t .- 3) .^ 2), 1 .- exp.(-t))
        knots = place_knots(t, y)
        @test length(knots) == 2
        for k in knots
            @test first(k) == 0 && last(k) == 1
            @test issorted(k) && allunique(k)
            @test all(in((t .- t[1]) ./ (t[end] - t[1])), k)               # knots lie on data points
        end
    end

    @testset "SplineBases" begin
        t = collect(0:0.5:10)
        sb = SplineBases(t, t[1:2:end], [[0, 0.3, 1], [0, 0.5, 0.7, 1]])
        @test length(sb.t_fine) == 81
        @test size(sb.B[1]) == (21, 5) && size(sb.B_gm[2]) == (11, 6) && size(sb.dB_fine[1]) == (81, 5)
        @test sb.B_gm[1] ≈ sb.B[1][1:2:end, :]
        @test sb.dB[2] ≈ basis_diff(sb.splines[2], sb.scaled) ./ 10
    end

    for case in filter(has_reference, REFERENCE_CASES)
        @testset "$case vs MATLAB" begin
            ref = load_reference(case)
            sp, d = ref["spline"], ref["data"]
            L = asint(d["L"])
            t, t_gm, t_fine = asvec(d["t"]), asvec(d["t_gm"]), asvec(d["t_fine"])
            traces = reshape(d["traces"], :, L)
            knots_ref = [asvec(k) for k in ascells(sp["knots"])]
            tol = (rtol = 1e-12, atol = 1e-12)

            # knots: automatic placement or normalized user input
            if Bool(asscalar(sp["autoknots"]))
                @test USER_KNOTS[case] === nothing
                knots = place_knots(t, traces)
            else
                knots = resolve_knots(t, traces; knots = USER_KNOTS[case])
                @test all(knots .≈ [asvec(k) for k in ascells(sp["knots_input"])])
            end
            @test length.(knots) == length.(knots_ref)
            @test all(isapprox.(knots, knots_ref; atol = 1e-14))

            # grids
            @test isapprox(EGM.matlab_linspace(t[1], t[end], 81), t_fine; atol = 1e-12)
            sb = SplineBases(t, t_gm, knots_ref; order = asint(sp["order"]), t_fine = t_fine)
            @test isapprox(sb.scaled, asvec(sp["scaled"]); atol = 1e-15)
            @test isapprox(sb.scaled_gm, asvec(sp["scaled_gm"]); atol = 1e-15)
            @test isapprox(sb.scaled_fine, asvec(sp["scaled_fine"]); atol = 1e-15)
            @test [b.card for b in sb.splines] == asint.(asvec(sp["card"]))

            # basis matrices on all grids
            for k in 1:L
                @test isapprox(sb.B[k], ascells(sp["B"])[k]; tol...)
                @test isapprox(sb.dB[k], ascells(sp["dB"])[k]; tol...)
                @test isapprox(sb.B_gm[k], ascells(sp["basis_gm"])[k]; tol...)
                @test isapprox(sb.dB_gm[k], ascells(sp["dbasis_gm"])[k]; tol...)
                @test isapprox(sb.B_fine[k], ascells(sp["B_fine"])[k]; tol...)
                @test isapprox(sb.dB_fine[k], ascells(sp["dB_fine"])[k]; tol...)
                grid = asvec(sp["grid"])
                @test isapprox(basis(sb.splines[k], grid), ascells(sp["grid_basis"])[k]; tol...)
                @test isapprox(basis_diff(sb.splines[k], grid), ascells(sp["grid_dbasis"])[k]; tol...)
            end
        end
    end
end
