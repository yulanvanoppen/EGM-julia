# Interactive smoothing, back end (no GUI):
#   - the mutable multi-trajectory Smoother reproduces `smooth` for one trajectory,
#     and gives each of N identical cells the single-trajectory result;
#   - multi-trajectory knot placement;
#   - every app action of SmoothingSession (state selection, manual knots, equidistant
#     and automatic knots, live/manual smoothing, finishing, cell ordering).

using EGM: Smoother, smoothing_result, SmoothingSession, select_states!, click_knot!, equidistant_knots!,
           automatic_knots!, smooth!, finish!, CellSmoothingResult

@testset "Interactive smoothing (back end)" begin
    m = higgins_selkov()
    t = collect(0:0.5:10)

    @testset "Smoother vs smooth (one trajectory)" begin
        d = generate_data(m, t; observed = [1, 2], error_std = 0.03, rng = MersenneTwister(16))
        y = d.traces[:, :, 1]
        t_gm = t[1:2:end]
        r1 = smooth(SplineBases(t, t_gm, resolve_knots(t, y)), y)
        s = Smoother(t, y; t_gm = t_gm)
        EGM.optimize!(s)
        r2 = smoothing_result(s)
        @test r2 isa SmoothingResult
        @test r2.iterations == r1.iterations
        @test all(isapprox.(r2.delta, r1.delta; rtol = 1e-10))
        @test isapprox([r2.sigma2; r2.tau2], [r1.sigma2; r1.tau2]; rtol = 1e-8)
        for f in (:variances_sm, :variances_gm, :smoothed, :dsmoothed, :smoothed_fine, :dsmoothed_fine)
            @test isapprox(getfield(r2, f), getfield(r1, f); rtol = 1e-8)
        end
        @test all(isapprox.(r2.lincomb, r1.lincomb; rtol = 1e-8))
        @test all(isapprox.(r2.var_delta, r1.var_delta; rtol = 1e-8))
    end

    @testset "identical cells give the single-trajectory result" begin
        d = generate_data(m, t; observed = [1, 2], error_std = 0.03, rng = MersenneTwister(17))
        y = d.traces[:, :, 1]
        knots = [[2.0, 5.0, 7.5], [3.0, 6.0]]
        s1 = Smoother(t, y; knots = knots)
        EGM.optimize!(s1)
        s3 = Smoother(t, cat(y, y, y; dims = 3); knots = knots)
        EGM.optimize!(s3)
        r = smoothing_result(s3)
        @test r isa CellSmoothingResult
        @test size(r.smoothed) == (21, 2, 3) && size(r.variances_sm) == (21, 2, 3)
        for k in 1:2, i in 1:3
            @test isapprox(s3.delta[k][:, i], s1.delta[k][:, 1]; rtol = 1e-7)
        end
        @test isapprox(s3.sigma2 .+ s3.tau2, s1.sigma2 .+ s1.tau2; rtol = 1e-6)
    end

    @testset "multi-trajectory knot placement" begin
        d = generate_data(m, t; N = 6, D_mult = 0.1, observed = [1, 2], error_std = 0.02, rng = MersenneTwister(18))
        kn = place_knots(t, d.traces)
        @test length(kn) == 2
        for k in kn
            @test first(k) == 0 && last(k) == 1 && issorted(k) && allunique(k)
            @test all(in((t .- t[1]) ./ (t[end] - t[1])), k)
        end
        @test place_knots(t, d.traces[:, :, 1:1]) == place_knots(t, d.traces[:, :, 1])
    end

    @testset "cell order" begin
        base = 1 .+ sin.(t)
        traces = cat((reshape(c .* base, :, 1) for c in (0.9, 1.0, 1.1, 1.5))...; dims = 3)
        @test EGM.cell_order(traces) == [3, 4, 1, 2]                      # most typical, then most atypical first
        @test EGM.cell_order(traces[:, :, 1:1]) == [1]
    end

    @testset "session actions" begin
        d = generate_data(m, t; N = 4, D_mult = 0.1, observed = [1, 2], error_std = 0.02, rng = MersenneTwister(19))
        sess = SmoothingSession(t, d.traces; state_names = ["S", "P"], knots = [2.0, 5.0, 7.5])
        @test sess.active == [1, 2] && sess.ncells == 4
        @test sort(sess.cells_sorted) == 1:4 && EGM.shown_cells(sess) == sess.cells_sorted
        @test all(!iszero, sess.smoother.delta[1])                        # smoothed at start

        select_states!(sess, "P")
        @test sess.active == [2]
        select_states!(sess, :all)
        @test sess.active == [1, 2]
        @test_throws ArgumentError select_states!(sess, "X")

        equidistant_knots!(sess, 3)                                       # all states, live
        @test EGM.interior_knots(sess, 1) ≈ [0.25, 0.5, 0.75] && EGM.interior_knots(sess, 2) ≈ [0.25, 0.5, 0.75]
        @test isempty(sess.stale) && all(!iszero, sess.smoother.delta[2])

        click_knot!(sess, 2.0)                                            # add at t = 2 (normalized 0.2), all states
        @test EGM.interior_knots(sess, 2) ≈ [0.2, 0.25, 0.5, 0.75]
        click_knot!(sess, 2.4; remove = true)                             # nearest is t = 2.5
        @test EGM.interior_knots(sess, 1) ≈ [0.2, 0.5, 0.75]
        @test EGM.interior_knots_time(sess, 1) ≈ [2.0, 5.0, 7.5]

        select_states!(sess, 1)
        equidistant_knots!(sess, 1)                                       # only state 1
        @test EGM.interior_knots(sess, 1) ≈ [0.5] && EGM.interior_knots(sess, 2) ≈ [0.2, 0.5, 0.75]

        sess.live = false                                                 # manual smoothing from here on
        equidistant_knots!(sess, EGM.max_knots(sess))
        n = length(EGM.interior_knots(sess, 1))
        click_knot!(sess, 3.3)                                            # at the maximum: no knot added
        @test length(EGM.interior_knots(sess, 1)) == n == EGM.max_knots(sess)

        automatic_knots!(sess)
        @test sess.smoother.knots[1] == place_knots(t, d.traces)[1]
        smooth!(sess)

        equidistant_knots!(sess, 2)
        @test 1 in sess.stale && all(iszero, sess.smoother.delta[1])
        smooth!(sess)
        @test isempty(sess.stale) && all(!iszero, sess.smoother.delta[1])

        equidistant_knots!(sess, 4)                                       # left unsmoothed …
        out = finish!(sess)                                               # … finish smooths it
        @test isempty(sess.stale) && all(!iszero, sess.smoother.delta[1])
        @test out.smoothing isa CellSmoothingResult
        @test out.knots[1] ≈ [0, 0.2, 0.4, 0.6, 0.8, 1]
        @test out.knots_time[1] ≈ [0, 2, 4, 6, 8, 10]
        @test size(out.smoothing.smoothed_fine) == (81, 2, 4)
    end

    @testset "single-trajectory session and egm(interactive) without GLMakie" begin
        d = generate_data(m, t; observed = [2], error_std = 0.02, rng = MersenneTwister(20))
        sess = SmoothingSession(t, d.traces[:, :, 1]; state_names = ["P"])
        @test sess.cells_sorted == [1] && sess.ncells == 1
        out = finish!(sess)
        @test out.smoothing isa SmoothingResult
        if !isdefined(Main, :GLMakie)                                     # the app needs `using GLMakie`
            y = d.traces[:, :, 1]
            @test_throws MethodError egm(m, t, y, [2], 0.5 .* m.β0; interactive = true)

            # stand-in for the window, finishing at once with a chosen action, to test how egm
            # continues or stops (called through invokelatest because the method is new)
            stub_action = Ref(:finish)
            @eval EGM.smoothing_app(t, traces::Matrix{Float64}; estimation = false, state_names = nothing, kwargs...) =
                merge(EGM.finish!(EGM.SmoothingSession(t, traces; state_names = state_names, kwargs...)),
                      (; action = $(stub_action)[]))
            out = Base.invokelatest(egm, m, t, y, [2], 0.5 .* m.β0; interactive = true)
            @test out isa NamedTuple && out.action === :finish && out.smoothing isa SmoothingResult
            stub_action[] = :closed
            @test Base.invokelatest(egm, m, t, y, [2], 0.5 .* m.β0; interactive = true).action === :closed
            stub_action[] = :estimate
            r = Base.invokelatest(egm, m, t, y, [2], 0.5 .* m.β0; interactive = true, x_init = [1.5, 0.8])
            @test r isa EGMResult && all(isfinite, r.β)
        end
    end
end
