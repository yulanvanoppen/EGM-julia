# Helpers for reading the MATLAB golden files (see test/reference/README.md).
#
# MAT.jl conventions to keep in mind:
#   * MATLAB scalars          -> Float64 / Bool
#   * row/column vectors      -> 1×n / n×1 Matrix (use `asvec`)
#   * trailing singleton dims are dropped by MATLAB (use `dims(x, n)`)
#   * cell arrays             -> Array{Any} (use `ascells`)
#   * 1×1 structs             -> Dict{String,Any}

using MAT, LinearAlgebra

const REFDIR = get(ENV, "EGM_REFDIR", joinpath(@__DIR__, "reference"))
const REFERENCE_CASES = ["HS_P", "HS_P_sub", "GLV_X3", "STAT5", "STAT5_patched"]

reference_path(case) = joinpath(REFDIR, case * ".mat")
has_reference(case) = isfile(reference_path(case))
load_reference(case) = matread(reference_path(case))

asvec(x::AbstractArray) = vec(x)
asvec(x) = [x]
asscalar(x::AbstractArray) = only(x)
asscalar(x) = x
asint(x) = Int(asscalar(x))
ascells(x::AbstractArray) = vec(x)
ascells(x) = Any[x]

"Sizes of `x` along the first `n` dimensions (1 beyond `ndims(x)`)."
dims(x, n) = ntuple(i -> size(x, i), n)

"Relative error ‖a − b‖ / ‖b‖ (absolute error if b = 0)."
function relerr(a, b)
    nb = norm(b)
    na = norm(a .- b)
    return iszero(nb) ? na : na / nb
end
