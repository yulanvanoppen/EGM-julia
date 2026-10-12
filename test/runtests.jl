# Run all tests:            ] test            (or Pkg.test())
# Run selected test files:  Pkg.test(test_args = ["8"])  or  Pkg.test(test_args = ["1", "3"])
# (numbers refer to test_phase<N>.jl; the test files need the setup below, so they are not
#  meant to be include()d on their own)

using Test
using EGM
using LinearAlgebra, Random

include("reference.jl")
include("models.jl")
include("helpers.jl")

const PHASES = isempty(ARGS) ? (0:8) : parse.(Int, ARGS)

@testset "EGM.jl" begin
    for p in PHASES
        include("test_phase$(p).jl")
    end
end
