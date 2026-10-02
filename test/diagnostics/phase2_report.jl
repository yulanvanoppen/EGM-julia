# Phase 2 diagnostic report: knots side by side and basis-matrix deviations.
#
#     julia --project=. test/diagnostics/phase2_report.jl

using EGM, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "reference.jl"))
include(joinpath(@__DIR__, "..", "models.jl"))
include(joinpath(@__DIR__, "..", "helpers.jl"))

for case in filter(has_reference, REFERENCE_CASES)
    ref = load_reference(case)
    sp, d = ref["spline"], ref["data"]
    L = asint(d["L"])
    t, t_gm, t_fine = asvec(d["t"]), asvec(d["t_gm"]), asvec(d["t_fine"])
    traces = reshape(d["traces"], :, L)
    knots_ref = [asvec(k) for k in ascells(sp["knots"])]
    auto = Bool(asscalar(sp["autoknots"]))
    knots = auto ? place_knots(t, traces) : resolve_knots(t, traces; knots = USER_KNOTS[case])
    println("\n=== $case (", auto ? "automatic" : "user", " knots) ===")
    for k in 1:L
        println("  state $k julia : ", round.(knots[k]; digits = 6))
        println("  state $k matlab: ", round.(knots_ref[k]; digits = 6))
    end
    sb = SplineBases(t, t_gm, knots_ref; order = asint(sp["order"]), t_fine = t_fine)
    for (name, jl, key) in (("B", sb.B, "B"), ("dB", sb.dB, "dB"), ("B_gm", sb.B_gm, "basis_gm"),
                            ("dB_gm", sb.dB_gm, "dbasis_gm"), ("B_fine", sb.B_fine, "B_fine"),
                            ("dB_fine", sb.dB_fine, "dB_fine"))
        for k in 1:L
            R = ascells(sp[key])[k]
            if size(jl[k]) != size(R)
                @printf("  %-8s state %d SIZE MISMATCH julia %s matlab %s\n", name, k, size(jl[k]), size(R))
            else
                v, i = findmax(abs.(jl[k] .- R))
                @printf("  %-8s state %d max|Δ| %.2e at %s\n", name, k, v, Tuple(i))
            end
        end
    end
end
