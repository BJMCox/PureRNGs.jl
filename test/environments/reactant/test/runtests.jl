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
_bits(value::Int32) = reinterpret(UInt32, value)
_bits(value::Int64) = reinterpret(UInt64, value)

function _snapshot(rng)
    range = UInt16(2):UInt16(3):UInt16(74)
    pure = (
        rand(rng, Bool),
        rand(rng, UInt32),
        _bits(rand(rng, Int32)),
        rand(rng, UInt64),
        _bits(rand(rng, Int64)),
        _bits(rand(rng, Float32)),
        _bits(rand(rng, Float64)),
        _bits(randn(rng, Float32)),
        _bits(randn(rng, Float64)),
        rand(rng, range),
        randat(rng, UInt64, 3),
        _bits(randat(rng, Int32, 3)),
        _bits(randat(rng, Int64, 3)),
        _bits(randnat(rng, Float32, 3)),
    )

    next_rng, bool_value = rand_next(rng, Bool)
    next_rng, uint32_value = rand_next(next_rng, UInt32)
    next_rng, int32_value = rand_next(next_rng, Int32)
    next_rng, uint64_value = rand_next(next_rng, UInt64)
    next_rng, int64_value = rand_next(next_rng, Int64)
    next_rng, float32_value = rand_next(next_rng, Float32)
    next_rng, float64_value = rand_next(next_rng, Float64)
    next_rng, normal32_value = randn_next(next_rng, Float32)
    next_rng, normal64_value = randn_next(next_rng, Float64)
    next_rng, range_value = rand_next(next_rng, range)
    continuation = (
        next_rng,
        bool_value,
        uint32_value,
        _bits(int32_value),
        uint64_value,
        _bits(int64_value),
        _bits(float32_value),
        _bits(float64_value),
        _bits(normal32_value),
        _bits(normal64_value),
        range_value,
    )

    derivation = (splitrng(rng, Val(3)), subrng(rng, 0x0123456789abcdef))
    return pure, continuation, derivation
end

function _primitive_step(rng)
    next_rng, value = rand_next(rng, UInt32)
    return next_rng, value, rand(rng, UInt64), randat(rng, UInt32, 3)
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

@testset "R42 dynamic carrier reuse" begin
    first = Philox4x32(0x0123456789abcdef)
    second = Philox4x32(0xfedcba9876543210)
    advanced, _ = rand_next(first, UInt64)
    first_carrier = Reactant.to_rarray(first)
    second_carrier = Reactant.to_rarray(second)
    advanced_carrier = Reactant.to_rarray(advanced)
    @test !(first_carrier isa AbstractPureRNG)

    compiled = Reactant.@compile sync = true _primitive_step(first_carrier)
    for (carrier, eager) in (
        (first_carrier, first),
        (second_carrier, second),
        (advanced_carrier, advanced),
    )
        next_carrier, value, pure, addressed = compiled(carrier)
        eager_next, eager_value, eager_pure, eager_addressed = _primitive_step(eager)
        @test value == eager_value
        @test pure == eager_pure
        @test addressed == eager_addressed

        reused_next, reused_value, _, _ = compiled(next_carrier)
        eager_reused_next, eager_reused_value, _, _ = _primitive_step(eager_next)
        @test reused_value == eager_reused_value
        @test typeof(reused_next) === typeof(next_carrier)
        @test typeof(eager_reused_next) === typeof(eager_next)
    end
end
