using Enzyme
using PureRNGs
using Random
using Test

function stateful_normal_objective!(rng, destination, scale)
    Random.randn!(rng, destination)
    return scale * sum(destination)
end

@testset "R65 StatefulRNG reverse normal overwrite" begin
    rng = StatefulRNG(Philox4x32(0x6501))
    expected_rng, expected = randn_next(parent(rng), Float64, 8)
    destination = zeros(8)
    shadow = fill(9.0, 8)
    scale = 1.25

    derivative = only(
        autodiff(
            Reverse,
            stateful_normal_objective!,
            Active,
            Const(rng),
            Duplicated(destination, shadow),
            Active(scale),
        ),
    )

    @test destination == expected
    @test parent(rng) === expected_rng
    @test iszero(shadow)
    @test derivative[3] == sum(expected)
end
