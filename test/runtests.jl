using Test
using EGM

include("reference.jl")
include("models.jl")
include("helpers.jl")

@testset "EGM.jl" begin
    include("test_phase0.jl")
    include("test_phase1.jl")
    include("test_phase2.jl")
    include("test_phase3.jl")
    include("test_phase4.jl")
    include("test_phase5.jl")
    include("test_phase6.jl")
    include("test_phase7.jl")
    include("test_phase8.jl")
end
