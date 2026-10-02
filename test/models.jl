# Julia versions of the EGM-v2 test models (model_*.txt), with fixed parameters
# substituted as in the reference cases. Free parameters follow the order of
# System.parameters_variable.

using EGM: ODEModel, LinearInterpolant, breakpoints

"Higgins-Selkov (model_HS.txt), fixed c = 0.95, free β = (r, k)."
function higgins_selkov()
    c = 0.95
    f(x, β, t) = begin
        S, P = x
        r, k = β
        [r - c * S * P^2, c * S * P^2 - k * P]
    end
    return ODEModel(f; name = "higgins_selkov", states = ["S", "P"], parameters = ["r", "k"],
                    x0 = [1.0, 1.0], β0 = [1.0, 1.0])
end

"Generalized Lotka-Volterra (model_GLV.txt), fixed r = 1, free β = (k1, k2)."
function generalized_lv()
    r = 1.0
    f(x, β, t) = begin
        X1, X2, X3 = x
        k1, k2 = β
        [X1 * (r - k1 * X1 - k2 * X3),
         X2 * (r - k1 * X2 - k2 * X1),
         X3 * (r - 1 * X3 - 2 * X2)]
    end
    return ODEModel(f; name = "generalizedLV", states = ["X1", "X2", "X3"], parameters = ["k1", "k2"],
                    x0 = [0.6, 0.6, 0.1], β0 = [1.0, 2.0])
end

const STAT5_INPUT = LinearInterpolant(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60],
    [1.0, 8.5, 14.3, 44.6, 58.3, 50.2, 45.7, 33.3, 36.3, 19.3, 19.7, 18.2, 3.0, 1.5, 0.8, 1.0])

"Noise-mixing matrix for the lumped STAT5 states: x123 = x1 + x2 + 2x3, x23 = x2 + 2x3."
function stat5_M()
    M = Matrix{Float64}(I, 7, 7)
    M[1, 2:3] = [1, 2]
    M[2, 3] = 2
    return M
end

"STAT5 (model_STAT5.txt), fixed k2 = 2.46 and tauinv = 0.15625, free β = (k1, k3, k4)."
function stat5()
    k2, tauinv = 2.46, 0.15625
    u = STAT5_INPUT
    f(x, β, t) = begin
        x123, x23, x3, x4, x3i, x3ii, x3tau = x
        k1, k3, k4 = β
        [2 * k4 * x3tau - 2 * k3 * x3,
         k1 * (x123 - x23) * u(t) - 2 * k3 * x3,
         -k3 * x3 + 0.5 * k2 * (x23 - 2 * x3)^2,
         -k4 * x3tau + k3 * x3,
         3 * tauinv * (x3 - x3i),
         3 * tauinv * (x3i - x3ii),
         3 * tauinv * (x3ii - x3tau)]
    end
    return ODEModel(f; name = "STAT5",
                    states = ["x123", "x23", "x3", "x4", "x3i", "x3ii", "x3tau"],
                    parameters = ["k1", "k3", "k4"],
                    x0 = [1.0, 0, 0, 0, 0, 0, 0], β0 = [0.021, 0.1066, 0.10658],
                    M = stat5_M(), tstops = breakpoints(u))
end

"Model belonging to each reference case."
const CASE_MODELS = Dict("HS_P" => higgins_selkov, "HS_P_sub" => higgins_selkov,
                         "GLV_X3" => generalized_lv,
                         "STAT5" => stat5, "STAT5_patched" => stat5)
