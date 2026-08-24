using PureRNGs
using Random
using Reactant
using Test

const FAMILIES = (
    Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
)

_bits(value::Float32) = reinterpret(UInt32, value)
_bits(value::Float64) = reinterpret(UInt64, value)

function _snapshot(rng)
    range = UInt16(2):UInt16(3):UInt16(74)
    pure = (
        rand(rng, Bool),
        rand(rng, UInt32),
        rand(rng, UInt64),
        _bits(rand(rng, Float32)),
        _bits(rand(rng, Float64)),
        _bits(randn(rng, Float32)),
        _bits(randn(rng, Float64)),
        rand(rng, range),
        randat(rng, UInt64, 3),
        _bits(randnat(rng, Float32, 3)),
    )

    next_rng, bool_value = rand_next(rng, Bool)
    next_rng, uint32_value = rand_next(next_rng, UInt32)
    next_rng, uint64_value = rand_next(next_rng, UInt64)
    next_rng, float32_value = rand_next(next_rng, Float32)
    next_rng, float64_value = rand_next(next_rng, Float64)
    next_rng, normal32_value = randn_next(next_rng, Float32)
    next_rng, normal64_value = randn_next(next_rng, Float64)
    next_rng, range_value = rand_next(next_rng, range)
    continuation = (
        next_rng,
        bool_value,
        uint32_value,
        uint64_value,
        _bits(float32_value),
        _bits(float64_value),
        _bits(normal32_value),
        _bits(normal64_value),
        range_value,
    )

    derivation = (splitrng(rng, Val(3)), subrng(rng, 0x0123456789abcdef))
    return pure, continuation, derivation
end

Reactant.set_default_backend("cpu")

@testset "R42 Reactant compiled values equal eager values" begin
    @test Base.pkgversion(Reactant) == v"0.2.280"
    for F in FAMILIES
        @testset "$F" begin
            rng, _ = rand_next(F(0x123456), Bool)
            eager = _snapshot(rng)
            compiled = Reactant.@compile sync = true _snapshot(rng)
            @test isequal(compiled(rng), eager)
        end
    end
end
