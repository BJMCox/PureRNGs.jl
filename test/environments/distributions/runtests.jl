using Distributions
using PureRNGs
using Test

@testset "R34 Distributions StatefulRNG smoke" begin
    root = Philox4x32(0x812)
    distribution = Normal()
    scalar_next, scalar_expected = randn_next(root, Float64)
    mutable_rng = StatefulRNG(root)

    @test rand(mutable_rng, distribution) === scalar_expected
    batch_next, batch_expected = randn_next(scalar_next, Float64, 11)
    @test rand(mutable_rng, distribution, 11) == batch_expected
    @test mutable_rng.rng === batch_next
end
