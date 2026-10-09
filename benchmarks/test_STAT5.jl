# Port of test_STAT5.m: 100 simulated STAT5 datasets at the experimental time points
# (first point omitted), estimated with EGM and trajectory matching. Records parameter
# estimates, initial conditions, computing times, model fits, the sum of squares
# (sum_of_squares.m) and the generalized sum of squares (GSS.m).
#
#     julia --project=benchmarks benchmarks/test_STAT5.jl
#
# MODE = :compat reproduces the current EGM-v2 code (noise mixing matrix M also in the
# covariance propagation); set MODE = :default for the Julia default.

include(joinpath(@__DIR__, "common.jl"))

const MODE = :compat
const M_STAT5 = let M = Matrix{Float64}(I, 7, 7)                     # x123 = x1 + x2 + 2x3, x23 = x2 + 2x3
    M[1, 2:3] = [1, 2]
    M[2, 3] = 2
    M
end
const MODEL = load_model("model_STAT5.txt"; fixed = ["tauinv", "k2"], M = M_STAT5)

const T_DATA = let lines = filter(!isempty, strip.(readlines(joinpath(MODEL_DIR, "STAT5-data.csv"))))
    [parse(Float64, first(split(l, ','))) for l in lines[2:end]]   # first column, header skipped
end

stat5_constraint(x) = (x[4] = min(x[3], x[4]); x)
const IC = (; ic_scan = [3, 4], ic_constraint = stat5_constraint)
const K0 = MODEL.β0
const BOUNDS = (; lb = 0.1 .* K0, ub = 10 .* K0)
const OBSERVED = [1, 2]

stat5_metrics(res, egm_res, d) = (; SS = sum_of_squares(MODEL, res.beta, d, OBSERVED),
                                    GSS = gss(MODEL, egm_res.result, res.beta, OBSERVED))

records = run_test("test_STAT5", MODEL; t = T_DATA, observed = OBSERVED, noise = 0.02, β_init = 2 .* K0,
                   x_init = [1, 0.1, 0.01, 0.001, 0, 0, 0] .+ 1e-8, prepare = drop_first,
                   egm_kwargs = (; knots = [[12, 18, 30, 40], [6, 10, 14, 18, 25, 40]], σ = 0.001, BOUNDS...,
                                 prior_mean = [1.0, 1.0, 1.0], prior_prec = zeros(3, 3), mode = MODE, IC...),
                   tm_kwargs = (; BOUNDS..., IC...), metrics = stat5_metrics)
export_results("test_STAT5", MODEL, records,
               (; noise = 0.02, β_init = "2 k0", knots = [[12, 18, 30, 40], [6, 10, 14, 18, 25, 40]], σ = 0.001,
                lb = "0.1 k0", ub = "10 k0", prior_mean = [1, 1, 1], prior_prec = "0", mode = string(MODE),
                ic_scan = [3, 4], ic_constraint = "x4 = min(x3, x4)", first_point_omitted = true))
