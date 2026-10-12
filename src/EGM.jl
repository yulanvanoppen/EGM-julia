"""
    EGM

Extended gradient matching (EGM) for parameter inference in ODE models that are
linear in their parameters, with support for partially observed systems via
extended Kalman smoothing inside an approximate E-M loop.

Julia translation of https://github.com/yulanvanoppen/EGM-v2. The main entry point
is [`egm`](@ref); see the README for an overview and DEVELOPMENT.md for the
correspondence with the MATLAB code.
"""
module EGM

using LinearAlgebra
using Random
using ForwardDiff
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase

# Phase 1: model specification, g/h decomposition, Jacobians, integration,
#          variational (Φ) system, process-noise matrix M
include("model.jl")
export ODEModel, integrate, integrate_variational, LinearInterpolant, breakpoints

# Phase 2: B-spline bases and derivatives, knot handling and placement
include("bspline.jl")
export BSpline, basis, basis_diff, place_knots, resolve_knots, SplineBases

# Phase 3: FWLS spline smoothing with additive + multiplicative noise
include("smoother.jl")
export smooth, SmoothingResult

# Phase 4: extended Kalman filter/smoother with joint state/measurement covariances
include("kalman.jl")
export KalmanSmoother, KalmanResult, eks, predict_step, augment_covariances

# Phase 5: gradient matching: Savitzky-Golay differentiation, residual covariance,
#          box-constrained GLS
include("gm.jl")
export GradientMatching, GMStepResult, gm_step, qp_gls

# Phase 6: initial-condition search, E-M loop, final fit
include("estimate.jl")
export egm, egm_inference, egm_cells, EGMResult, ic_candidates

# Interactive smoothing: multi-trajectory smoother and app logic (window in ext/EGMMakieExt.jl)
include("interactive.jl")
export Smoother, smoothing_result, CellSmoothingResult, SmoothingSession, smoothing_app,
       select_states!, click_knot!, equidistant_knots!, automatic_knots!, smooth!, finish!

# Phase 7: trajectory matching, uncertainties, data generation, IQM model files
include("tm.jl")
include("generate.jl")
include("iqm.jl")
export tm, TMResult, generate_data, load_iqm, parse_iqm, compile_iqm, integrate_sensitivities,
       parameter_covariance

end # module
