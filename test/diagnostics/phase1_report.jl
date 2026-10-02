# Phase 1 diagnostic report: prints the error of every MATLAB comparison made in
# test_phase1.jl, including where the largest deviation occurs.
#
# Usage, from the package folder (MAT must be loadable; see the instructions
# in the chat, or run `] add MAT` once in your default environment):
#
#     julia --project=. test/diagnostics/phase1_report.jl

using EGM, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "reference.jl"))
include(joinpath(@__DIR__, "..", "models.jl"))

"Print relative error, largest absolute deviation and its location, and PASS/FAIL."
function compare(label, A, B; rtol, atol = 0.0, colnames = nothing)
    A = Array{Float64}(A)
    B = Array{Float64}(B)
    if size(A) != size(B)
        @printf("  %-26s SIZE MISMATCH  julia %s  matlab %s\n", label, size(A), size(B))
        return
    end
    D = abs.(A .- B)
    rel = norm(A .- B) / max(norm(B), eps())
    maxabs, idx = findmax(D)
    pass = norm(A .- B) ≤ max(atol, rtol * max(norm(A), norm(B)))
    loc = Tuple(idx)
    if colnames !== nothing && length(loc) == 2
        loc = "t-index $(loc[1]), $(colnames[loc[2]])"
    end
    @printf("  %-26s %s  rel %.2e (tol %.0e)  max|Δ| %.2e at %s  [julia %.8g, matlab %.8g]\n",
            label, pass ? "PASS" : "FAIL", rel, rtol, maxabs, loc, A[idx], B[idx])
end

"Column labels for the auxiliary layout: states, then Ph_ij."
aux_names(m) = vcat(m.states, ["Ph$(i)$(j)" for j in 1:m.K for i in 1:m.K])

for case in filter(has_reference, REFERENCE_CASES)
    ref = load_reference(case)
    s, d = ref["system"], ref["data"]
    m = CASE_MODELS[case]()
    K, P = m.K, m.P
    println("\n=== $case ===")

    println("  names/values: states ", m.states == String.(ascells(s["states"])),
            ", parameters ", m.parameters == String.(ascells(s["parameters_variable"])),
            ", x0 ", m.x0 ≈ asvec(s["x0"]), ", β0 ", m.β0 ≈ asvec(s["k0"]),
            ", M ", m.M ≈ ref["meta"]["M"])

    X = s["eval_states"]
    t = asvec(s["eval_times"])
    β = asvec(s["eval_beta"])
    Tg = length(t)
    compare("g", EGM.g_stacked(m, X, t), s["g"]; rtol = 1e-10, atol = 1e-12)
    compare("h", EGM.h_traj(m, X, t), s["h"]; rtol = 1e-10, atol = 1e-12)
    compare("rhs", EGM.rhs_traj(m, X, t, β), s["rhs"]; rtol = 1e-10, atol = 1e-12)
    compare("df", EGM.df_stacked(m, X, t, β), s["df"]; rtol = 1e-10, atol = 1e-12)
    compare("dh", EGM.dh_stacked(m, X, t), s["dh"]; rtol = 1e-10, atol = 1e-12)
    compare("dg", EGM.dg_array(m, X, t), reshape(s["dg"], K, P, Tg, K); rtol = 1e-10, atol = 1e-12)

    tt = asvec(d["t"])
    Xj = integrate(m, m.β0, m.x0, tt; reltol = 1e-10, abstol = 1e-12)
    Xj_ref = integrate(m, m.β0, m.x0, tt; reltol = 1e-13, abstol = 1e-15)
    compare("traj_tight", Xj, s["traj_tight"]; rtol = 5e-5, colnames = m.states)
    compare("traj_default", Xj, s["traj_default"]; rtol = 2e-4, colnames = m.states)
    compare("julia traj 1e-10 vs 1e-13", Xj, Xj_ref; rtol = 1e-8, colnames = m.states)

    Xa, Φa = integrate_variational(m, m.β0, m.x0, tt; Φ0 = m.M, reltol = 1e-10, abstol = 1e-12)
    Xa_ref, Φa_ref = integrate_variational(m, m.β0, m.x0, tt; Φ0 = m.M, reltol = 1e-13, abstol = 1e-15)
    A = EGM.auxiliary_layout(Xa, Φa)
    compare("aux_tight", A, s["aux_tight"]; rtol = 5e-5, colnames = aux_names(m))
    compare("julia aux 1e-10 vs 1e-13", A, EGM.auxiliary_layout(Xa_ref, Φa_ref); rtol = 1e-8,
            colnames = aux_names(m))

    # per-column relative errors for the tight comparisons
    for (label, J, R, names) in (("traj_tight", Xj, s["traj_tight"], m.states),
                                 ("aux_tight", A, s["aux_tight"], aux_names(m)))
        colrel = [norm(J[:, j] .- R[:, j]) / max(norm(R[:, j]), eps()) for j in axes(J, 2)]
        worst = sortperm(colrel; rev = true)[1:min(4, length(colrel))]
        println("  $label worst columns: ",
                join(["$(names[j]) $(@sprintf("%.1e", colrel[j]))" for j in worst], ", "))
    end
end
