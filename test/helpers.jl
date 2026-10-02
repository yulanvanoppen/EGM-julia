# Helpers shared by the test files and the diagnostic reports: reference-case settings
# and conversions of MATLAB reference data.

using EGM: SplineBases, SmoothingResult

"User-supplied knots (time units) of the reference cases; `nothing` = automatic placement."
const USER_KNOTS = Dict("HS_P" => nothing, "HS_P_sub" => nothing,
                        "GLV_X3" => 2.5:1.25:7.5,
                        "STAT5" => [[12, 18, 30, 40], [6, 10, 14, 18, 25, 40]],
                        "STAT5_patched" => [[12, 18, 30, 40], [6, 10, 14, 18, 25, 40]])

"Select iteration `i` from an array stacked along dimension `nd + 1` (MATLAB may drop it)."
slice_iter(A, nd, i) = reshape(A, dims(A, nd + 1))[ntuple(_ -> Colon(), nd)..., i]

"Prior mean and precision from the MATLAB settings (scalars kept as scalars, as in QPGLS)."
function ref_prior(inf, P)
    pr = inf["prior"]
    μ = pr["mean"] isa Number || length(pr["mean"]) == 1 ? asscalar(pr["mean"]) : asvec(pr["mean"])
    Λ = pr["prec"] isa Number || length(pr["prec"]) == 1 ? asscalar(pr["prec"]) : reshape(pr["prec"], P, P)
    return μ, Λ
end

stat5_ic_constraint(x) = (x[4] = min(x[3], x[4]); x)

"IC search options that reproduce the model-name special cases of EGM.init_ICmean."
const CASE_IC = Dict("HS_P" => NamedTuple(), "HS_P_sub" => NamedTuple(),
                     "GLV_X3" => (ic_scan = [1], ic_ties = [2 => 1]),
                     "STAT5" => (ic_scan = [3, 4], ic_constraint = stat5_ic_constraint),
                     "STAT5_patched" => (ic_scan = [3, 4], ic_constraint = stat5_ic_constraint))

"Inference settings of a reference case (keywords for egm / egm_inference)."
function ref_inference_settings(ref, case, P)
    inf, fl = ref["settings"]["inference"], ref["settings"]["filter"]
    μ, Λ = ref_prior(inf, P)
    return (; x_init = asvec(fl["x_init"]), P_CV = asscalar(fl["P_CV"]), σ = asscalar(fl["sigma"]),
            mode = Bool(asscalar(ref["meta"]["patched"])) ? :default : :compat,
            lb = asvec(inf["lb"]), ub = asvec(inf["ub"]), prior_mean = μ, prior_prec = Λ,
            niter_gm = asint(inf["niter"]), tol_gm = asscalar(inf["tol"]),
            positive = Bool(asscalar(inf["positive"])), CASE_IC[case]...)
end

"MATLAB's smoothing result as a SmoothingResult."
function ref_smoothing(ref)
    d, sp, sm = ref["data"], ref["spline"], ref["smoother"]
    L = asint(d["L"])
    sb = SplineBases(asvec(d["t"]), asvec(d["t_gm"]), [asvec(k) for k in ascells(sp["knots"])];
                     t_fine = asvec(d["t_fine"]))
    mat(x) = Matrix{Float64}(x)
    col(x) = reshape(Matrix{Float64}(x isa Number ? fill(x, 1, 1) : x), :, L)
    return SmoothingResult(sb, [asvec(x) for x in ascells(sm["delta"])], asvec(sm["sigma2"]), asvec(sm["tau2"]),
                           col(sm["variances_sm"]), col(sm["variances_gm"]), col(sm["smoothed"]),
                           col(sm["dsmoothed"]), col(sm["smoothed_fine"]), col(sm["dsmoothed_fine"]),
                           [mat(x) for x in ascells(sm["lincomb_data"])], [mat(x) for x in ascells(sm["var_delta"])], 0)
end
