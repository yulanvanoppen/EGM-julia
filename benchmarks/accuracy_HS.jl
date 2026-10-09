# Port of accuracy_HS.m: accuracy and computing time of EGM and trajectory matching on the
# Higgins–Selkov model with only P observed, over sampling intervals, noise levels,
# initial-condition errors and 50 seeds (3000 cases).
#
#     julia --project=benchmarks benchmarks/accuracy_HS.jl
#
# See benchmarks/README.md for the options (data source, seed subsets, threads).

include(joinpath(@__DIR__, "common.jl"))

MODEL = load_model("model_HS.txt"; fixed = ["c"])

SETTINGS = (; dt_values = [0.25, 0.5, 1.0], noise_levels = [0.005, 0.01, 0.02, 0.05],
                  init_error = 4 .* [0, 0.125, 0.25, 0.375, 0.5], seeds = seed_selection(1:50),
                  observed = [MODEL.K], β_init = "0.5 k0", σ = 0.01)

records = run_accuracy("accuracy_HS", MODEL; dt_values = SETTINGS.dt_values, noise_levels = SETTINGS.noise_levels,
                       init_error = SETTINGS.init_error, seeds = SETTINGS.seeds, observed = SETTINGS.observed,
                       egm_kwargs = (; σ = SETTINGS.σ))
export_results("accuracy_HS", MODEL, records, SETTINGS)
