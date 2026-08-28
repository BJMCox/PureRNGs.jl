using PureRNGs
using Distributions
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
const REACTANT_EXT = Base.get_extension(PureRNGs, :PureRNGsReactantExt)
const REACTANT_DISTRIBUTIONS_EXT =
    Base.get_extension(PureRNGs, :PureRNGsReactantDistributionsExt)

struct NegativeIntegerRange <: AbstractRange{Int64}
    first::Int64
    step::Int64
    length::Int64
end

Base.first(range::NegativeIntegerRange) = range.first
Base.last(range::NegativeIntegerRange) = range.first + (range.length - 1) * range.step
Base.length(range::NegativeIntegerRange) = range.length
Base.step(range::NegativeIntegerRange) = range.step
Base.getindex(range::NegativeIntegerRange, index::Integer) =
    range.first + (Int64(index) - 1) * range.step

_same_value(got, expected::Bool) = Bool(got) === expected
_same_value(got, expected::T) where {T<:Unsigned} = T(got) === expected
_same_value(got, expected::T) where {T<:Signed} =
    reinterpret(unsigned(T), T(got)) === reinterpret(unsigned(T), expected)
_same_value(got, expected::Float32) =
    reinterpret(UInt32, Float32(got)) === reinterpret(UInt32, expected)
_same_value(got, expected::Float64) =
    reinterpret(UInt64, Float64(got)) === reinterpret(UInt64, expected)
_same_value(got, expected::AbstractPureRNG) =
    Array(got.state) == Array(Reactant.to_rarray(expected).state)
_same_value(got::Tuple, expected::Tuple) =
    length(got) == length(expected) && all(_same_value.(got, expected))

function _normal_components(rng::AbstractPureRNG, ::Type{T}) where {T}
    position = rng.position
    raw = PureRNGs._extract_bits_unchecked(
        rng,
        PureRNGs.FAMILY_NORMAL,
        PureRNGs._position_block(position),
        position.bit,
        Val(PureRNGs._normal_bits(T)),
    )
    midpoint = PureRNGs._normal_midpoint(T, raw)
    q = midpoint - T(0.5)
    radius = sqrt(-log(ifelse(q < zero(T), midpoint, one(T) - midpoint)))
    return raw, midpoint, radius
end

function _normal_components(rng::REACTANT_EXT._ReactantRNG, ::Type{T}) where {T}
    width = PureRNGs._normal_bits(T)
    raw = REACTANT_EXT._raw(rng, PureRNGs.FAMILY_NORMAL, Val(width))
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    midpoint = REACTANT_EXT._cast_scalar(T, (raw * UInt64(2)) | UInt64(1)) * scale
    q = midpoint - T(0.5)
    radius = sqrt(-log(ifelse(q < zero(T), midpoint, one(T) - midpoint)))
    return raw, midpoint, radius
end

_normal_observation(rng, ::Type{T}, value) where {T} =
    (_normal_components(rng, T)..., value)

function _normal_at_observation(rng::AbstractPureRNG, ::Type{T}, index) where {T}
    addressed = PureRNGs._addressed_rng(rng, PureRNGs._normal_bits(T), index)
    return _normal_observation(addressed, T, randnat(rng, T, index))
end

function _normal_at_observation(rng::REACTANT_EXT._ReactantRNG, ::Type{T}, index) where {T}
    addressed = PureRNGs._addressed_rng(rng, PureRNGs._normal_bits(T), index)
    return _normal_observation(addressed, T, randnat(rng, T, index))
end

function _exponential_components(rng::AbstractPureRNG, ::Type{T}) where {T}
    position = rng.position
    width = PureRNGs._exponential_bits(T)
    raw = PureRNGs._extract_bits_unchecked(
        rng,
        PureRNGs.FAMILY_EXP,
        PureRNGs._position_block(position),
        position.bit,
        Val(width),
    )
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = T(raw) * scale
    return raw, u, one(T) - u, randexp(rng, T)
end

function _exponential_components(rng::REACTANT_EXT._ReactantRNG, ::Type{T}) where {T}
    width = PureRNGs._exponential_bits(T)
    raw = REACTANT_EXT._raw(rng, PureRNGs.FAMILY_EXP, Val(width))
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = REACTANT_EXT._cast_scalar(T, raw) * scale
    return raw, u, one(T) - u, randexp(rng, T)
end

function _exponential_at_components(rng::AbstractPureRNG, ::Type{T}, index) where {T}
    addressed = PureRNGs._addressed_rng(rng, PureRNGs._exponential_bits(T), index)
    return _exponential_components(addressed, T)
end

function _exponential_at_components(
    rng::REACTANT_EXT._ReactantRNG,
    ::Type{T},
    index,
) where {T}
    addressed = PureRNGs._addressed_rng(rng, PureRNGs._exponential_bits(T), index)
    return _exponential_components(addressed, T)
end

_exponential_probe(rng) =
    (_exponential_components(rng, Float32), _exponential_components(rng, Float64))

function _snapshot(rng)
    range = UInt16(2):UInt16(3):UInt16(74)
    linrange = LinRange{Int64}(Int64(1) << 53, (Int64(1) << 53) + Int64(4), 5)
    pure = (
        rand(rng, Bool),
        rand(rng, UInt32),
        rand(rng, Int32),
        rand(rng, UInt64),
        rand(rng, Int64),
        rand(rng, Float32),
        rand(rng, Float64),
        randexp(rng, Float32),
        randexp(rng, Float64),
        rand(rng, range),
        rand(rng, linrange),
        randat(rng, UInt64, 3),
        randat(rng, Int32, 3),
        randat(rng, Int64, 3),
        randexpat(rng, Float32, 3),
        randexpat(rng, Float64, 3),
    )
    normals = (
        _normal_observation(rng, Float32, randn(rng, Float32)),
        _normal_observation(rng, Float64, randn(rng, Float64)),
        _normal_at_observation(rng, Float32, 3),
        _normal_at_observation(rng, Float64, 3),
    )

    next_rng, bool_value = rand_next(rng, Bool)
    next_rng, uint32_value = rand_next(next_rng, UInt32)
    next_rng, int32_value = rand_next(next_rng, Int32)
    next_rng, uint64_value = rand_next(next_rng, UInt64)
    next_rng, int64_value = rand_next(next_rng, Int64)
    next_rng, float32_value = rand_next(next_rng, Float32)
    next_rng, float64_value = rand_next(next_rng, Float64)
    normal32_rng = next_rng
    next_rng, normal32_value = randn_next(normal32_rng, Float32)
    normal64_rng = next_rng
    next_rng, normal64_value = randn_next(normal64_rng, Float64)
    continuation_normals = (
        _normal_observation(normal32_rng, Float32, normal32_value),
        _normal_observation(normal64_rng, Float64, normal64_value),
    )
    next_rng, exponential32_value = randexp_next(next_rng, Float32)
    next_rng, exponential64_value = randexp_next(next_rng, Float64)
    next_rng, range_value = rand_next(next_rng, range)
    next_rng, linrange_value = rand_next(next_rng, linrange)
    continuation = (
        next_rng,
        bool_value,
        uint32_value,
        int32_value,
        uint64_value,
        int64_value,
        float32_value,
        float64_value,
        exponential32_value,
        exponential64_value,
        range_value,
        linrange_value,
    )

    derivation = (splitrng(rng, Val(3)), subrng(rng, 0x0123456789abcdef))
    nonnormal = (pure, continuation, derivation, _exponential_probe(rng))
    return nonnormal, (normals..., continuation_normals...)
end

_positioned(rng, block::UInt64, bit::UInt16) = _positioned(rng, block, UInt64(2), bit)

function _positioned(rng, block::UInt64, high::UInt64, bit::UInt16)
    position =
        rng.position isa PureRNGs._Position64 ? PureRNGs._Position64(block, bit) :
        PureRNGs._Position128(block, high, bit)
    return PureRNGs._rebuild(rng, position, rng.device)
end

function _range_snapshot(rng, range)
    next_rng, continuation = rand_next(rng, range)
    return rand(rng, range), next_rng, continuation
end

_ordinal_range_snapshot(rng) = _range_snapshot(rng, UInt16(2):UInt16(3):UInt16(74))
_linrange_snapshot(rng) =
    _range_snapshot(rng, LinRange{Int64}(Int64(1) << 53, (Int64(1) << 53) + Int64(4), 5))

function _normal_probe64(rng)
    addressed = PureRNGs._addressed_rng(rng, UInt16(52), 3)
    raw = REACTANT_EXT._raw(addressed, PureRNGs.FAMILY_NORMAL, Val(52))
    midpoint =
        REACTANT_EXT._cast_scalar(Float64, (raw * UInt64(2)) | UInt64(1)) * Float64(0x1p-53)
    q = midpoint - 0.5
    radius = sqrt(-log(ifelse(q < 0.0, midpoint, 1.0 - midpoint)))
    return raw, midpoint, radius, randnat(rng, Float64, 3)
end

function _normal_tail_from_radius(radius::T, midpoint::T) where {T<:AbstractFloat}
    _, _, C, D, E, F = PureRNGs._as241_coefficients(T)
    if radius <= T(5)
        reduced = radius - T(1.6)
        value =
            PureRNGs._as241_horner(reduced, C) /
            PureRNGs._as241_horner(reduced, D)
    else
        reduced = radius - T(5)
        value =
            PureRNGs._as241_horner(reduced, E) /
            PureRNGs._as241_horner(reduced, F)
    end
    return midpoint < T(0.5) ? -value : value
end

function _same_normal_observation(got, expected)
    T = typeof(expected[2])
    _same_value(got[1], expected[1]) || return false
    _same_value(got[2], expected[2]) || return false
    midpoint = T(got[2])
    expected_value = if abs(midpoint - T(0.5)) <= T(0.425)
        expected[4]
    else
        _normal_tail_from_radius(T(got[3]), midpoint)
    end
    return _same_value(got[4], expected_value)
end

function _same_snapshot(got, expected)
    return _same_value(got[1], expected[1]) &&
           all(_same_normal_observation.(got[2], expected[2]))
end

_distribution_primitive(rng, ::Uniform{T}) where {T} = rand(rng, T)
_distribution_primitive(rng, ::Exponential{T}) where {T} = _exponential_components(rng, T)
_distribution_primitive(rng, ::Bernoulli{T}) where {T} = rand(rng, T)
_distribution_primitive(rng, distribution::DiscreteUniform) =
    rand(rng, distribution.a:distribution.b)

_distribution_primitive_at(rng, ::Uniform{T}, index) where {T} = randat(rng, T, index)
_distribution_primitive_at(rng, ::Exponential{T}, index) where {T} =
    _exponential_at_components(rng, T, index)
_distribution_primitive_at(rng, ::Bernoulli{T}, index) where {T} = randat(rng, T, index)

function _distribution_primitive_at(rng, distribution::DiscreteUniform, index)
    width = REACTANT_DISTRIBUTIONS_EXT._distribution_span(distribution)
    addressed = PureRNGs._addressed_rng(rng, width, index)
    return rand(addressed, distribution.a:distribution.b)
end

_distribution_pair(rng, distribution) =
    (_distribution_primitive(rng, distribution), rand(rng, distribution))
_distribution_pair_at(rng, distribution, index) =
    (_distribution_primitive_at(rng, distribution, index), randat(rng, distribution, index))

_distribution_chain(rng, ::Tuple{}) = (rng, ())
function _distribution_chain(rng, distributions::Tuple)
    distribution = first(distributions)
    primitive = _distribution_primitive(rng, distribution)
    next_rng, value = rand_next(rng, distribution)
    final_rng, values = _distribution_chain(next_rng, Base.tail(distributions))
    return final_rng, ((primitive, value), values...)
end

_normal_distribution_pair(rng, d::Normal{T}) where {T} =
    (_normal_observation(rng, T, randn(rng, T)), rand(rng, d))
_normal_distribution_pair_at(rng, d::Normal{T}, index) where {T} =
    (_normal_at_observation(rng, T, index), randat(rng, d, index))
_normal_distribution_type(::Normal{T}) where {T} = T

_normal_distribution_chain(rng, ::Tuple{}) = (rng, ())
function _normal_distribution_chain(rng, distributions::Tuple)
    distribution = first(distributions)
    T = _normal_distribution_type(distribution)
    primitive = _normal_observation(rng, T, randn(rng, T))
    next_rng, value = rand_next(rng, distribution)
    final_rng, values = _normal_distribution_chain(next_rng, Base.tail(distributions))
    return final_rng, ((primitive, value), values...)
end

function _distribution_snapshot(rng)
    normal_distributions =
        (Normal{Float32}(Float32(1.25), Float32(0.75)), Normal{Float64}(1.25, 0.75))
    distributions = (
        Uniform{Float32}(Float32(-2), Float32(3)),
        Uniform{Float64}(-2, 3),
        Exponential{Float32}(Float32(1.5)),
        Exponential{Float64}(1.5),
        Bernoulli{Float32}(Float32(0.25)),
        Bernoulli{Float64}(0.25),
        DiscreteUniform(-17, 29),
    )
    pure = map(distribution -> _distribution_pair(rng, distribution), distributions)
    addressed =
        map(distribution -> _distribution_pair_at(rng, distribution, 3), distributions)
    next_rng, continuation = _distribution_chain(rng, distributions)
    normal_pure = map(
        distribution -> _normal_distribution_pair(rng, distribution),
        normal_distributions,
    )
    normal_addressed = map(
        distribution -> _normal_distribution_pair_at(rng, distribution, 3),
        normal_distributions,
    )
    normal_next_rng, normal_continuation =
        _normal_distribution_chain(rng, normal_distributions)
    return (
        pure,
        addressed,
        next_rng,
        continuation,
        normal_pure,
        normal_addressed,
        normal_next_rng,
        normal_continuation,
    )
end

function _same_normal_mapping(got, expected, distribution::Normal{T}) where {T}
    _same_normal_observation(got[1], expected[1]) || return false
    primitive = T(got[1][4])
    mapped = T(got[2])
    mapped_expected = fma(distribution.σ, primitive, distribution.μ)
    return _same_value(mapped, mapped_expected)
end

function _same_distribution_mapping(got, expected, distribution::Exponential{T}) where {T}
    got_primitive = got[1]
    expected_primitive = expected[1]
    _same_value(got_primitive[1:3], expected_primitive[1:3]) || return false
    return _same_value(T(got[2]), distribution.θ * T(got_primitive[4]))
end

function _same_distribution_mapping(got, expected, distribution::Uniform{T}) where {T}
    primitive = T(got[1])
    _same_value(primitive, expected[1]) || return false
    width = distribution.b - distribution.a
    scaled = PureRNGs._rounded_product(width, primitive, primitive - T(0.5))
    return _same_value(T(got[2]), distribution.a + scaled)
end

function _same_distribution_mapping(got, expected, distribution::Bernoulli{T}) where {T}
    primitive = T(got[1])
    return _same_value(primitive, expected[1]) &&
           _same_value(Bool(got[2]), primitive < distribution.p)
end

function _same_distribution_mapping(got, expected, ::DiscreteUniform)
    primitive = Int(got[1])
    return _same_value(primitive, expected[1]) && _same_value(Int(got[2]), primitive)
end

function _same_distribution_snapshot(got, expected)
    normal_distributions =
        (Normal{Float32}(Float32(1.25), Float32(0.75)), Normal{Float64}(1.25, 0.75))
    distributions = (
        Uniform{Float32}(Float32(-2), Float32(3)),
        Uniform{Float64}(-2, 3),
        Exponential{Float32}(Float32(1.5)),
        Exponential{Float64}(1.5),
        Bernoulli{Float32}(Float32(0.25)),
        Bernoulli{Float64}(0.25),
        DiscreteUniform(-17, 29),
    )
    return all(_same_distribution_mapping.(got[1], expected[1], distributions)) &&
           all(_same_distribution_mapping.(got[2], expected[2], distributions)) &&
           _same_value(got[3], expected[3]) &&
           all(_same_distribution_mapping.(got[4], expected[4], distributions)) &&
           all(_same_normal_mapping.(got[5], expected[5], normal_distributions)) &&
           all(_same_normal_mapping.(got[6], expected[6], normal_distributions)) &&
           _same_value(got[7], expected[7]) &&
           all(_same_normal_mapping.(got[8], expected[8], normal_distributions))
end

function _primitive_step(rng)
    next_rng, value = rand_next(rng, UInt32)
    return next_rng, value, rand(rng, UInt64), randat(rng, UInt32, 3)
end

_large_addressed_uint64(rng) = randat(rng, UInt64, (big(1) << 122) + 1)

function _last_bit_rng(::Type{F}) where {F}
    rng = F(0x123456)
    block_bits = PureRNGs._block_bits(rng)
    position = if rng.position isa PureRNGs._Position64
        PureRNGs._Position64(PureRNGs._max_block(rng), block_bits - UInt16(1))
    else
        PureRNGs._Position128(typemax(UInt64), typemax(UInt64), block_bits - UInt16(1))
    end
    return PureRNGs._rebuild(rng, position, rng.device)
end

Reactant.set_default_backend("cpu")

@testset "Reactant distribution extension loads" begin
    @test REACTANT_EXT !== nothing
    @test REACTANT_DISTRIBUTIONS_EXT !== nothing
end

@testset "R42 wide addressed index" begin
    index = (big(1) << 122) + 1
    eager = Philox4x64(0x123456)
    carrier = Reactant.to_rarray(eager)
    @test REACTANT_EXT._address_offset(index, UInt64(64)) ==
          (UInt64(0), UInt64(0), UInt64(1))
    compiled = Reactant.@compile sync = true _large_addressed_uint64(carrier)
    @test UInt64(compiled(carrier)) == _large_addressed_uint64(eager)
end

@testset "R43 Reactant preserves eager normal arithmetic" begin
    compile_rng = _positioned(Philox2x64(0x123456), UInt64(3), UInt16(17))
    central_rng = _positioned(Philox2x64(0x654321), UInt64(7), UInt16(29))
    tail_rng = Philox2x64(0x5)
    compile_carrier = Reactant.to_rarray(compile_rng)
    compiled = Reactant.@compile sync = true _normal_probe64(compile_carrier)

    central_raw, central_midpoint, _, central_value =
        compiled(Reactant.to_rarray(central_rng))
    @test UInt64(central_raw) == 0x000174208c39ebcd
    @test reinterpret(UInt64, Float64(central_midpoint)) == 0x3fb74208c39ebcd8
    @test reinterpret(UInt64, Float64(central_value)) == 0xbff55e55782ee12e

    tail_raw, tail_midpoint, tail_radius, tail_value =
        compiled(Reactant.to_rarray(tail_rng))
    tail_midpoint_host = Float64(tail_midpoint)
    expected_tail = _normal_tail_from_radius(Float64(tail_radius), tail_midpoint_host)
    @test UInt64(tail_raw) == 0x000114ac2a562bb5
    @test reinterpret(UInt64, tail_midpoint_host) == 0x3fb14ac2a562bb58
    @test reinterpret(UInt64, Float64(tail_value)) == reinterpret(UInt64, expected_tail)
end

@testset "R42 Reactant compiled values equal eager values" begin
    @test Base.pkgversion(Reactant) == v"0.2.280"
    for F in FAMILIES
        @testset "$F" begin
            first = _positioned(F(0x123456), UInt64(3), UInt64(2), UInt16(17))
            second = _positioned(F(0x654321), UInt64(7), UInt64(5), UInt16(29))
            first_carrier = Reactant.to_rarray(first)
            second_carrier = Reactant.to_rarray(second)
            compiled = Reactant.@compile sync = true _snapshot(first_carrier)
            @test _same_snapshot(compiled(first_carrier), _snapshot(first))
            @test _same_snapshot(compiled(second_carrier), _snapshot(second))
        end
    end
end

@testset "R42 public HLO keeps state dynamic and omits preflight" begin
    cases = ((Philox2x32, 4), (Philox4x32, 5), (Philox4x64, 6))
    forbidden = (
        "stablehlo.custom_call",
        "func.call",
        "call @",
        "scf.if",
        "cf.cond_br",
        "stablehlo.case",
        "stablehlo.while",
        "ArgumentError",
        "throw",
        "callback",
        "exhaust",
    )

    for (F, state_length) in cases
        carrier = Reactant.to_rarray(_positioned(F(0x123456), UInt64(3), UInt16(17)))
        hlo = String(Reactant.@code_hlo optimize = false rand_next(carrier, UInt64))
        state_type = "tensor<$(state_length)xui64>"

        @test occursin("func.func @main(%arg0: $state_type", hlo)
        @test !occursin("%arg1", hlo)
        @test length(findall("sizes = [1]", hlo)) >= state_length
        @test length(findall(state_type, hlo)) >= 2
        @test all(pattern -> !occursin(pattern, hlo), forbidden)
    end
end

@testset "R42 fixed distributions" begin
    for F in FAMILIES
        first = _positioned(F(0x123456), UInt64(3), UInt16(17))
        second = _positioned(F(0x654321), UInt64(7), UInt16(29))
        first_carrier = Reactant.to_rarray(first)
        second_carrier = Reactant.to_rarray(second)
        compiled = Reactant.@compile sync = true _distribution_snapshot(first_carrier)
        @test _same_distribution_snapshot(
            compiled(first_carrier),
            _distribution_snapshot(first),
        )
        @test _same_distribution_snapshot(
            compiled(second_carrier),
            _distribution_snapshot(second),
        )
    end
end

@testset "R42 integer range method surface" begin
    first = _positioned(Philox4x32(0x123456), UInt64(3), UInt16(17))
    second = _positioned(Philox4x32(0x654321), UInt64(7), UInt16(29))
    first_carrier = Reactant.to_rarray(first)
    second_carrier = Reactant.to_rarray(second)
    unsupported = NegativeIntegerRange(-5, -2, 4)

    @test !applicable(rand, first_carrier, unsupported)
    @test !applicable(rand_next, first_carrier, unsupported)

    for range in (UInt16(2):UInt16(3):UInt16(74), LinRange{Int64}(-20, 20, 5))
        @test applicable(rand, first_carrier, range)
        @test applicable(rand_next, first_carrier, range)
    end

    compiled_ordinal = Reactant.@compile sync = true _ordinal_range_snapshot(first_carrier)
    @test _same_value(compiled_ordinal(first_carrier), _ordinal_range_snapshot(first))
    @test _same_value(compiled_ordinal(second_carrier), _ordinal_range_snapshot(second))

    compiled_linrange = Reactant.@compile sync = true _linrange_snapshot(first_carrier)
    @test _same_value(compiled_linrange(first_carrier), _linrange_snapshot(first))
    @test _same_value(compiled_linrange(second_carrier), _linrange_snapshot(second))
end

@testset "R42 exact-end continuation" begin
    for F in (Philox2x32, Philox4x32, Philox4x64)
        eager = _last_bit_rng(F)
        eager_next, eager_value = rand_next(eager, Bool)
        carrier = Reactant.to_rarray(eager)
        compiled = Reactant.@compile sync = true rand_next(carrier, Bool)
        next_carrier, value = compiled(carrier, Bool)
        terminal_carrier = Reactant.to_rarray(eager_next)
        @test value == eager_value
        @test Array(next_carrier.state) == Array(terminal_carrier.state)
    end
end

@testset "R42 eager exhaustion remains checked with Reactant loaded" begin
    for F in FAMILIES
        last = _last_bit_rng(F)
        terminal, value = rand_next(last, Bool)
        terminal_state = Reactant.to_rarray(terminal).state |> Array

        @test value == rand(last, Bool)
        @test_throws ArgumentError rand(terminal, Bool)
        @test_throws ArgumentError rand_next(terminal, Bool)
        @test Array(Reactant.to_rarray(terminal).state) == terminal_state
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
    for (carrier, eager) in
        ((first_carrier, first), (second_carrier, second), (advanced_carrier, advanced))
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
