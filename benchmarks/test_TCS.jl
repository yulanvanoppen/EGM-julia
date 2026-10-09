# Port of test_TCS.m: 100 simulated datasets of the bifunctional two-component system
# (states 3, 5, 6 observed), estimated with EGM and trajectory matching.
#
#     julia --project=benchmarks benchmarks/test_TCS.jl
#
# The initial-condition search scans state 4 (RF), as EGM-v2 does for "bifunctional_TCS".

include(joinpath(@__DIR__, "common.jl"))

MODEL = load_model("model_TCS.txt"; fixed = ["k2", "k3", "k6", "k8"])
K0 = MODEL.β0
BOUNDS = (; lb = 0.1 .* K0, ub = 10 .* K0)
IC = (; ic_scan = [4])

records = run_test("test_TCS", MODEL; t = [0, 1, 3, 5:5:50...], observed = [3, 5, 6], noise = 0.05,
                   β_init = 0.5 .* K0, x_init = [0, 0, 0.2, 2.5, 0, 0] .+ eps(),
                   egm_kwargs = (; knots = [0, 3, 25], BOUNDS..., prior_mean = [1, 1, 1, 1e-8],
                                 prior_prec = Matrix(Diagonal([0.0, 0, 0, 25])), IC...),
                   tm_kwargs = (; BOUNDS..., IC...))
export_results("test_TCS", MODEL, records,
               (; noise = 0.05, β_init = "0.5 k0", knots = [0, 3, 25], lb = "0.1 k0", ub = "10 k0",
                prior_mean = [1, 1, 1, 1e-8], prior_prec_diag = [0, 0, 0, 25], ic_scan = [4]))
