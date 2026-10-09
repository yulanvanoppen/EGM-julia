# Port of accuracy_GLV.m: accuracy and computing time of EGM and trajectory matching on the
# generalized Lotka–Volterra model with only X3 observed, over sampling intervals, noise
# levels, initial-condition errors and 50 seeds (3000 cases).
#
#     julia --project=benchmarks benchmarks/accuracy_GLV.jl
#
# The initial-condition search scans X1(0) with X2(0) = X1(0), as EGM-v2 does for
# "generalizedLV". Note: accuracy_GLV.m uses auxiliary_GLV_legacyConvention.txt, which is
# not in the EGM-v2 repository; this port uses the standard variational equation.

include(joinpath(@__DIR__, "common.jl"))

const MODEL = load_model("model_GLV.txt"; fixed = ["r"])
const IC = (; ic_scan = [1], ic_ties = [2 => 1])

const SETTINGS = (; dt_values = [0.25, 0.5, 1.0], noise_levels = [0.005, 0.02, 0.05, 0.1],
                  init_error = 4 .* [0, 0.125, 0.25, 0.375, 0.5], seeds = seed_selection(1:50),
                  observed = [MODEL.K], β_init = "0.5 k0", ic_scan = IC.ic_scan, ic_ties = "2 => 1")

records = run_accuracy("accuracy_GLV", MODEL; dt_values = SETTINGS.dt_values, noise_levels = SETTINGS.noise_levels,
                       init_error = SETTINGS.init_error, seeds = SETTINGS.seeds, observed = SETTINGS.observed,
                       egm_kwargs = IC, tm_kwargs = IC)
export_results("accuracy_GLV", MODEL, records, SETTINGS)
