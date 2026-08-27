using Distributions
using PureRNGs
using Test

@testset "R34 Distributions StatefulRNG smoke" begin
    normal_root = Philox4x32(0x812)
    normal = Normal()
    scalar_next, scalar_expected = randn_next(normal_root, Float64)
    mutable_rng = StatefulRNG(normal_root)

    @test rand(mutable_rng, normal) === scalar_expected
    batch_next, batch_expected = randn_next(scalar_next, Float64, 11)
    @test rand(mutable_rng, normal, 11) == batch_expected
    @test mutable_rng.rng === batch_next

    exponential_root = Philox4x32(0x813)
    exponential = Exponential()
    scalar_next, scalar_expected = randexp_next(exponential_root, Float64)
    mutable_rng = StatefulRNG(exponential_root)

    @test rand(mutable_rng, exponential) === scalar_expected
    batch_next, batch_expected = randexp_next(scalar_next, Float64, 11)
    @test rand(mutable_rng, exponential, 11) == batch_expected
    @test mutable_rng.rng === batch_next
end
