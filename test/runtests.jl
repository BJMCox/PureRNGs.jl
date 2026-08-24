using Test
using PureRNGs

@testset "package shape" begin
    @test PureRNGs isa Module
    @test nameof(PureRNGs) === :PureRNGs
    @test !isdefined(PureRNGs, :greet)
end

include("philox.jl")
include("threefry.jl")
include("families.jl")
include("oracle_conformance.jl")
include("derive.jl")
include("bits.jl")
include("uniform.jl")
include("uniform_allocating.jl")
include("integers.jl")
include("range_allocating.jl")
include("device_base_seams.jl")
include("cuda_extension_metadata.jl")
include("normal.jl")
