using Test
using PureRNGs

@testset "package shape" begin
    @test PureRNGs isa Module
    @test nameof(PureRNGs) === :PureRNGs
    @test !isdefined(PureRNGs, :greet)
end

include("philox.jl")
include("threefry.jl")
