# Port of test_repressilator.m: 100 simulated repressilator datasets (proteins P, Q, R
# observed), estimated with EGM and trajectory matching.
#
#     julia --project=benchmarks benchmarks/test_repressilator.jl
#
# The initial-condition search scans PnDNA and mP, with QnDNA = RnDNA = PnDNA and
# mQ = mR = mP, as EGM-v2 does for "repressilator".

include(joinpath(@__DIR__, "common.jl"))

MODEL = load_model("model_repressilator.txt"; fixed = ["DNAT", "kf", "Kd", "m1", "p1"])
K0 = MODEL.β0
IC = (; ic_scan = [4, 7], ic_ties = [5 => 4, 6 => 4, 8 => 7, 9 => 7])

records = run_test("test_repressilator", MODEL; t = 0:5:100, observed = [1, 2, 3], noise = 0.05,
                   β_init = 0.5 .* K0, x_init = [10, 20, 30, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2],
                   egm_kwargs = (; knots = 10:10:90, IC...), tm_kwargs = IC)
export_results("test_repressilator", MODEL, records,
               (; noise = 0.05, β_init = "0.5 k0", knots = collect(10:10:90), ic_scan = [4, 7],
                ic_ties = "5,6 => 4; 8,9 => 7"))
