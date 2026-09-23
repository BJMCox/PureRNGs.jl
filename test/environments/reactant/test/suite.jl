# Reactant reserves three quarters of the GPU per process by default. The suite
# uses a few hundred megabytes, so it allocates on demand and leaves the device
# to other processes on the same host.
get!(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")

using PureRNGs
using Distributions
using Random
using Reactant
using Test

include(joinpath(@__DIR__, "..", "..", "..", "fixtures.jl"))

function _select_generators(names)
    isempty(names) && return GENERATOR_TYPES
    return Tuple(map(names) do name
        index = findfirst(F -> string(nameof(F)) == name, GENERATOR_TYPES)
        index === nothing && throw(ArgumentError("unknown generator: $name"))
        return GENERATOR_TYPES[index]
    end)
end

const SELECTED_GENERATORS = _select_generators(ARGS)
const REACTANT_EXT = Base.get_extension(PureRNGs, :PureRNGsReactantExt)
const REACTANT_DISTRIBUTIONS_EXT =
    Base.get_extension(PureRNGs, :PureRNGsReactantDistributionsExt)
const NORMAL_DISTRIBUTIONS = (
    Normal{Float32}(Float32(1.25), Float32(0.75)),
    Normal{Float64}(1.25, 0.75),
    Normal{Float32}(zero(Float32), nextfloat(zero(Float32))),
    Normal{Float64}(zero(Float64), nextfloat(zero(Float64))),
)
const CONTINUOUS_DISTRIBUTIONS = (
    LogNormal{Float32}(Float32(0.25), Float32(0.75)),
    LogNormal{Float64}(0.25, 0.75),
    Weibull{Float32}(Float32(1.75), Float32(0.75)),
    Weibull{Float64}(1.75, 0.75),
    Rayleigh{Float32}(Float32(0.75)),
    Rayleigh{Float64}(0.75),
    Laplace{Float32}(Float32(0.25), Float32(0.75)),
    Laplace{Float64}(0.25, 0.75),
)
const FIXED_DISTRIBUTIONS = (
    Uniform{Float32}(Float32(-2), Float32(3)),
    Uniform{Float64}(-2, 3),
    Exponential{Float32}(Float32(1.5)),
    Exponential{Float64}(1.5),
    Bernoulli{Float32}(Float32(0.25)),
    Bernoulli{Float64}(0.25),
    DiscreteUniform(-17, 29),
)
const SUBNORMAL_DISTRIBUTIONS = (
    Uniform{Float32}(nextfloat(zero(Float32)), 2 * nextfloat(zero(Float32))),
    Uniform{Float64}(nextfloat(zero(Float64)), 2 * nextfloat(zero(Float64))),
    Exponential{Float32}(nextfloat(zero(Float32))),
    Exponential{Float64}(nextfloat(zero(Float64))),
    Bernoulli{Float32}(nextfloat(zero(Float32))),
    Bernoulli{Float64}(nextfloat(zero(Float64))),
)
const CANCELLATION_NORMAL_DISTRIBUTIONS = (
    Normal{Float32}(
        reinterpret(Float32, UInt32(0x40517866)),
        reinterpret(Float32, UInt32(0x40490fdb)),
    ),
    Normal{Float64}(
        reinterpret(Float64, UInt64(0x3ff0147ba1729ded)),
        reinterpret(Float64, UInt64(0x400921fb54442d18)),
    ),
)
const SUBNORMAL_BERNOULLI = Bernoulli{Float32}(nextfloat(zero(Float32)))

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
_same_value(got, expected::PureRNGs._ReactantRNG) =
    Array(got.state) == Array(expected.state)
_same_value(got::Reactant.ConcretePJRTNumber, expected::Reactant.ConcretePJRTNumber) =
    _same_value(Reactant.to_number(got), Reactant.to_number(expected))
_same_value(got::Tuple, expected::Tuple) =
    length(got) == length(expected) && all(_same_value.(got, expected))

function _same_transform_class(got, expected::T) where {T<:Union{Float32,Float64}}
    value = T(got)
    isfinite(value) && isfinite(expected) || return false
    iszero(expected) && return _same_value(value, expected)
    return !iszero(value) && signbit(value) == signbit(expected)
end

# On the CPU backend XLA may contract a multiply-add that the host evaluates in
# two roundings, so the final transform may differ by one ulp (R43). Raw bits
# and midpoints are still compared exactly.
function _same_transform_value(got, expected::T) where {T<:Union{Float32,Float64}}
    REACTANT_TEST_BACKEND == "cpu" || return _same_transform_class(got, expected)
    value = T(got)
    return value === expected ||
           value === nextfloat(expected) ||
           value === prevfloat(expected)
end

function _normal_components(rng::AbstractPureRNG, ::Type{T}) where {T}
    position = rng.position
    raw = PureRNGs._extract_bits_unchecked(
        rng,
        PureRNGs._position_block(position),
        position.bit,
        Val(PureRNGs._normal_bits(T)),
    )
    midpoint = PureRNGs._open_midpoint(T, raw)
    return raw, midpoint
end

function _normal_components(rng::REACTANT_EXT._ReactantRNG, ::Type{T}) where {T}
    width = PureRNGs._normal_bits(T)
    raw = REACTANT_EXT._raw(rng, Val(width))
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    midpoint = REACTANT_EXT._convert(T, (raw * UInt64(2)) | UInt64(1)) * scale
    return raw, midpoint
end

_normal_observation(rng, ::Type{T}, value) where {T} =
    (_normal_components(rng, T)..., value)

function _normal_at_observation(rng::AbstractPureRNG, ::Type{T}, index) where {T}
    addressed = PureRNGs._addressed_rng(rng, PureRNGs._normal_bits(T), index)
    return _normal_observation(addressed, T, randn_at(rng, T, index))
end

function _normal_at_observation(rng::REACTANT_EXT._ReactantRNG, ::Type{T}, index) where {T}
    addressed = PureRNGs._addressed_rng(rng, PureRNGs._normal_bits(T), index)
    return _normal_observation(addressed, T, randn_at(rng, T, index))
end

function _exponential_components(rng::AbstractPureRNG, ::Type{T}) where {T}
    position = rng.position
    width = PureRNGs._exponential_bits(T)
    raw = PureRNGs._extract_bits_unchecked(
        rng,
        PureRNGs._position_block(position),
        position.bit,
        Val(width),
    )
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = T(2raw + 1) * scale
    return raw, u, one(T) - u, randexp(rng, T)
end

function _exponential_components(rng::REACTANT_EXT._ReactantRNG, ::Type{T}) where {T}
    width = PureRNGs._exponential_bits(T)
    raw = REACTANT_EXT._raw(rng, Val(width))
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = REACTANT_EXT._convert(T, (raw * UInt64(2)) | UInt64(1)) * scale
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
    # A span above 2^32 takes the K=128 reduction, which reads two words.
    wide_range = UInt64(0):(UInt64(1)<<40)
    linrange = LinRange{Int64}(Int64(1) << 53, (Int64(1) << 53) + Int64(4), 5)
    pure_values = (
        rand(rng, Bool),
        rand(rng, UInt32),
        rand(rng, Int32),
        rand(rng, UInt64),
        rand(rng, Int64),
        rand(rng, Float32),
        rand(rng, Float64),
        rand(rng, wide_range),
        first(rand_next(rng, wide_range)),
    )
    pure_exponentials = (randexp(rng, Float32), randexp(rng, Float64))
    pure_addressed = (
        rand(rng, range),
        rand(rng, linrange),
        rand_at(rng, UInt64, 3),
        rand_at(rng, Int32, 3),
        rand_at(rng, Int64, 3),
    )
    pure_addressed_exponentials = (randexp_at(rng, Float32, 3), randexp_at(rng, Float64, 3))
    pure = (pure_values, pure_exponentials, pure_addressed, pure_addressed_exponentials)
    normals = (
        _normal_observation(rng, Float32, randn(rng, Float32)),
        _normal_observation(rng, Float64, randn(rng, Float64)),
        _normal_at_observation(rng, Float32, 3),
        _normal_at_observation(rng, Float64, 3),
    )

    bool_value, next_rng = rand_next(rng, Bool)
    uint32_value, next_rng = rand_next(next_rng, UInt32)
    int32_value, next_rng = rand_next(next_rng, Int32)
    uint64_value, next_rng = rand_next(next_rng, UInt64)
    int64_value, next_rng = rand_next(next_rng, Int64)
    float32_value, next_rng = rand_next(next_rng, Float32)
    float64_value, next_rng = rand_next(next_rng, Float64)
    normal32_rng = next_rng
    normal32_value, next_rng = randn_next(normal32_rng, Float32)
    normal64_rng = next_rng
    normal64_value, next_rng = randn_next(normal64_rng, Float64)
    continuation_normals = (
        _normal_observation(normal32_rng, Float32, normal32_value),
        _normal_observation(normal64_rng, Float64, normal64_value),
    )
    exponential32_value, next_rng = randexp_next(next_rng, Float32)
    exponential64_value, next_rng = randexp_next(next_rng, Float64)
    range_value, next_rng = rand_next(next_rng, range)
    linrange_value, next_rng = rand_next(next_rng, linrange)
    continuation_values = (
        bool_value,
        uint32_value,
        int32_value,
        uint64_value,
        int64_value,
        float32_value,
        float64_value,
    )
    continuation_exponentials = (exponential32_value, exponential64_value)
    continuation_ranges = (range_value, linrange_value)
    continuation =
        (next_rng, continuation_values, continuation_exponentials, continuation_ranges)

    derivation = (splitrng(rng, Val(3)), subrng(rng, 0x0123456789abcdef))
    nonnormal = (pure, continuation, derivation, _exponential_probe(rng))
    return nonnormal, (normals..., continuation_normals...)
end

_positioned_at(rng, block::UInt64, bit::UInt16) = _positioned_at(rng, block, UInt64(2), bit)

function _positioned_at(rng, block::UInt64, high::UInt64, bit::UInt16)
    position =
        rng.position isa PureRNGs._Position64 ? PureRNGs._Position64(block, bit) :
        PureRNGs._Position128(block, high, bit)
    return PureRNGs._rebuild(rng, position, rng.device)
end

function _normal_probe64(rng)
    addressed = PureRNGs._addressed_rng(rng, UInt16(52), 3)
    raw = REACTANT_EXT._raw(addressed, Val(52))
    midpoint =
        REACTANT_EXT._convert(Float64, (raw * UInt64(2)) | UInt64(1)) * Float64(0x1p-53)
    return raw, midpoint, randn_at(rng, Float64, 3)
end

function _same_normal_observation(got, expected)
    _same_value(got[1], expected[1]) || return false
    _same_value(got[2], expected[2]) || return false
    return _same_transform_value(got[3], expected[3])
end

function _same_exponential_observation(got, expected)
    _same_value(got[1:3], expected[1:3]) || return false
    return _same_transform_value(got[4], expected[4])
end

function _same_nonnormal_snapshot(got, expected)
    got_pure, got_continuation, got_derivation, got_exponentials = got
    expected_pure, expected_continuation, expected_derivation, expected_exponentials =
        expected
    got_values, got_pure_exp, got_addressed, got_addressed_exp = got_pure
    expected_values, expected_pure_exp, expected_addressed, expected_addressed_exp =
        expected_pure
    got_state, got_continuation_values, got_continuation_exp, got_ranges = got_continuation
    expected_state,
    expected_continuation_values,
    expected_continuation_exp,
    expected_ranges = expected_continuation
    return _same_value(got_values, expected_values) &&
           all(_same_transform_value.(got_pure_exp, expected_pure_exp)) &&
           _same_value(got_addressed, expected_addressed) &&
           all(_same_transform_value.(got_addressed_exp, expected_addressed_exp)) &&
           _same_value(got_state, expected_state) &&
           _same_value(got_continuation_values, expected_continuation_values) &&
           all(_same_transform_value.(got_continuation_exp, expected_continuation_exp)) &&
           _same_value(got_ranges, expected_ranges) &&
           _same_value(got_derivation, expected_derivation) &&
           all(_same_exponential_observation.(got_exponentials, expected_exponentials))
end

function _same_snapshot(got, expected)
    return _same_nonnormal_snapshot(got[1], expected[1]) &&
           all(_same_normal_observation.(got[2], expected[2]))
end

function _snapshot_continuation(snapshot)
    nonnormal, _ = snapshot
    _, continuation, _, _ = nonnormal
    state, _, _, _ = continuation
    return state
end

function _snapshot_pure_exponential32(snapshot)
    nonnormal, _ = snapshot
    pure, _, _, _ = nonnormal
    _, exponentials, _, _ = pure
    return first(exponentials)
end

_distribution_primitive(rng, ::Uniform{T}) where {T} = rand(rng, T)
_distribution_primitive(rng, ::Exponential{T}) where {T} = _exponential_components(rng, T)
_distribution_primitive(rng, ::Bernoulli{T}) where {T} = rand(rng, T)
_distribution_primitive(rng, distribution::DiscreteUniform) =
    rand(rng, distribution.a:distribution.b)

_distribution_primitive_at(rng, ::Uniform{T}, index) where {T} = rand_at(rng, T, index)
_distribution_primitive_at(rng, ::Exponential{T}, index) where {T} =
    _exponential_at_components(rng, T, index)
_distribution_primitive_at(rng, ::Bernoulli{T}, index) where {T} = rand_at(rng, T, index)

function _distribution_primitive_at(rng, distribution::DiscreteUniform, index)
    width = REACTANT_DISTRIBUTIONS_EXT._distribution_span(distribution)
    addressed = PureRNGs._addressed_rng(rng, width, index)
    return rand(addressed, distribution.a:distribution.b)
end

function _distribution_formula(d::Uniform{T}, primitive) where {T}
    width = d.b - d.a
    scaled = PureRNGs._rounded_product(width, primitive, primitive - T(0.5))
    return d.a + scaled
end

_distribution_formula(d::Exponential, primitive) = d.θ * primitive[4]
_distribution_formula(d::Bernoulli, primitive) = primitive < d.p
_distribution_formula(::DiscreteUniform, primitive) = primitive

_continuous_primitive_next(rng, ::LogNormal{T}) where {T} = randn_next(rng, T)
_continuous_primitive_next(rng, ::Weibull{T}) where {T} = randexp_next(rng, T)
_continuous_primitive_next(rng, ::Rayleigh{T}) where {T} = randexp_next(rng, T)
function _continuous_primitive_next(rng, ::Laplace{T}) where {T}
    magnitude, after_magnitude = randexp_next(rng, T)
    positive, next_rng = rand_next(after_magnitude, Bool)
    return (magnitude, positive), next_rng
end

_continuous_formula(d::LogNormal, primitive) = exp(muladd(d.σ, primitive, d.μ))
_continuous_formula(d::Weibull, primitive) = d.θ * primitive^inv(d.α)
_continuous_formula(d::Rayleigh{T}, primitive) where {T} = d.σ * sqrt(T(2) * primitive)
_continuous_formula(d::Laplace, primitive) =
    muladd(ifelse(primitive[2], d.θ, -d.θ), primitive[1], d.μ)

function _continuous_distribution_snapshot(rng)
    return map(CONTINUOUS_DISTRIBUTIONS) do distribution
        primitive, expected_next = _continuous_primitive_next(rng, distribution)
        addressed = PureRNGs._addressed_rng(
            rng,
            REACTANT_DISTRIBUTIONS_EXT._distribution_span(distribution),
            3,
        )
        addressed_primitive, _ = _continuous_primitive_next(addressed, distribution)
        continued, actual_next = rand_next(rng, distribution)
        return (
            rand(rng, distribution),
            _continuous_formula(distribution, primitive),
            continued,
            _continuous_formula(distribution, primitive),
            actual_next,
            expected_next,
            rand_at(rng, distribution, 3),
            _continuous_formula(distribution, addressed_primitive),
        )
    end
end

function _same_continuous_mapping(got, expected)
    return _same_value(got[1], got[2]) &&
           _same_value(got[3], got[4]) &&
           _same_value(got[7], got[8]) &&
           _same_transform_value(got[2], expected[2]) &&
           _same_transform_value(got[4], expected[4]) &&
           _same_transform_value(got[8], expected[8]) &&
           _same_value(got[5], expected[6])
end

function _distribution_pair(rng, distribution)
    primitive = _distribution_primitive(rng, distribution)
    return (
        primitive,
        rand(rng, distribution),
        _distribution_formula(distribution, primitive),
    )
end

function _distribution_pair_at(rng, distribution, index)
    primitive = _distribution_primitive_at(rng, distribution, index)
    return (
        primitive,
        rand_at(rng, distribution, index),
        _distribution_formula(distribution, primitive),
    )
end

_distribution_chain(rng, ::Tuple{}) = (rng, ())
function _distribution_chain(rng, distributions::Tuple)
    distribution = first(distributions)
    primitive = _distribution_primitive(rng, distribution)
    value, next_rng = rand_next(rng, distribution)
    final_rng, values = _distribution_chain(next_rng, Base.tail(distributions))
    direct = _distribution_formula(distribution, primitive)
    return final_rng, ((primitive, value, direct), values...)
end

function _normal_distribution_pair(rng, d::Normal{T}) where {T}
    primitive = _normal_observation(rng, T, randn(rng, T))
    return primitive, rand(rng, d), muladd(d.σ, primitive[3], d.μ)
end

function _normal_distribution_pair_at(rng, d::Normal{T}, index) where {T}
    primitive = _normal_at_observation(rng, T, index)
    return primitive, rand_at(rng, d, index), muladd(d.σ, primitive[3], d.μ)
end

_normal_distribution_type(::Normal{T}) where {T} = T

_normal_distribution_chain(rng, ::Tuple{}) = (rng, ())
function _normal_distribution_chain(rng, distributions::Tuple)
    distribution = first(distributions)
    T = _normal_distribution_type(distribution)
    primitive = _normal_observation(rng, T, randn(rng, T))
    value, next_rng = rand_next(rng, distribution)
    final_rng, values = _normal_distribution_chain(next_rng, Base.tail(distributions))
    direct = muladd(distribution.σ, primitive[3], distribution.μ)
    return final_rng, ((primitive, value, direct), values...)
end

function _distribution_group(rng, distributions)
    pure = map(distribution -> _distribution_pair(rng, distribution), distributions)
    addressed =
        map(distribution -> _distribution_pair_at(rng, distribution, 3), distributions)
    next_rng, continuation = _distribution_chain(rng, distributions)
    return pure, addressed, next_rng, continuation
end

function _normal_distribution_group(rng, distributions)
    normal_pure =
        map(distribution -> _normal_distribution_pair(rng, distribution), distributions)
    normal_addressed = map(
        distribution -> _normal_distribution_pair_at(rng, distribution, 3),
        distributions,
    )
    normal_next_rng, normal_continuation = _normal_distribution_chain(rng, distributions)
    return normal_pure, normal_addressed, normal_next_rng, normal_continuation
end

_fixed_distribution_snapshot(rng) = _distribution_group(rng, FIXED_DISTRIBUTIONS)
_normal_distribution_snapshot(rng) = _normal_distribution_group(rng, NORMAL_DISTRIBUTIONS)
_subnormal_distribution_snapshot(rng) = _distribution_group(rng, SUBNORMAL_DISTRIBUTIONS)

function _same_normal_mapping(got, expected, distribution::Normal{T}) where {T}
    _same_normal_observation(got[1], expected[1]) || return false
    return _same_value(T(got[2]), T(got[3]))
end

function _same_distribution_mapping(got, expected, distribution::Exponential{T}) where {T}
    got_primitive = got[1]
    expected_primitive = expected[1]
    _same_value(got_primitive[1:3], expected_primitive[1:3]) || return false
    _same_transform_value(got_primitive[4], expected_primitive[4]) || return false
    return _same_value(T(got[2]), T(got[3]))
end

function _same_distribution_mapping(got, expected, ::Uniform{T}) where {T}
    primitive = T(got[1])
    _same_value(primitive, expected[1]) || return false
    return _same_value(T(got[2]), T(got[3]))
end

function _same_distribution_mapping(got, expected, ::Bernoulli{T}) where {T}
    primitive = T(got[1])
    return _same_value(primitive, expected[1]) && _same_value(Bool(got[2]), Bool(got[3]))
end

function _same_distribution_mapping(got, expected, ::DiscreteUniform)
    primitive = Int(got[1])
    return _same_value(primitive, expected[1]) &&
           _same_value(Int(got[2]), expected[2]) &&
           _same_value(Int(got[3]), primitive)
end

function _check_distribution_group(got, expected, distributions, comparator)
    got_pure, got_addressed, got_state, got_continuation = got
    expected_pure, expected_addressed, expected_state, expected_continuation = expected
    for (form, got_values, expected_values) in (
        ("pure", got_pure, expected_pure),
        ("addressed", got_addressed, expected_addressed),
        ("continuation", got_continuation, expected_continuation),
    )
        @testset "$form" begin
            for (got_value, expected_value, distribution) in
                zip(got_values, expected_values, distributions)
                @testset "$distribution" begin
                    @test comparator(got_value, expected_value, distribution)
                end
            end
        end
    end
    @testset "state" begin
        @test _same_value(got_state, expected_state)
    end
end

function _check_distribution_executable(
    group,
    compiled,
    snapshot,
    distributions,
    comparator,
    first,
    second,
    first_carrier,
    second_carrier,
)
    @testset "$group" begin
        first_got = compiled(first_carrier)
        first_expected = snapshot(first)
        @testset "first carrier" begin
            _check_distribution_group(first_got, first_expected, distributions, comparator)
        end
        @testset "exact replay" begin
            @test _same_value(compiled(first_carrier), first_got)
        end
        @testset "second carrier" begin
            _check_distribution_group(
                compiled(second_carrier),
                snapshot(second),
                distributions,
                comparator,
            )
        end

        _, _, got_next, _ = first_got
        _, _, expected_next, _ = first_expected
        @test typeof(got_next) === typeof(first_carrier)
        @testset "returned continuation carrier" begin
            _check_distribution_group(
                compiled(got_next),
                snapshot(expected_next),
                distributions,
                comparator,
            )
        end
    end
end

function _cancellation_distribution_snapshot(rng)
    float32, float64 = CANCELLATION_NORMAL_DISTRIBUTIONS
    return (
        _normal_distribution_pair_at(rng, float32, 3),
        _normal_distribution_pair(rng, float64),
    )
end

_same_cancellation_distribution_snapshot(got, expected) =
    all(_same_normal_mapping.(got, expected, CANCELLATION_NORMAL_DISTRIBUTIONS))

_subnormal_bernoulli_snapshot(rng) = _distribution_pair(rng, SUBNORMAL_BERNOULLI)

_large_addressed_uint64(rng) = rand_at(rng, UInt64, (big(1) << 122) + 1)

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

# Without an override the suite follows Reactant's default client, so a test
# runner on a GPU host needs no environment variable.
const REACTANT_TEST_BACKEND = get(ENV, "PURERNGS_REACTANT_BACKEND") do
    Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == "cpu" ? "cpu" : "gpu"
end
REACTANT_TEST_BACKEND in ("cpu", "gpu") ||
    error("PURERNGS_REACTANT_BACKEND must be cpu or gpu")
Reactant.set_default_backend(REACTANT_TEST_BACKEND)

# An extension is not a submodule of its parent, so a recursive scan that starts
# at PureRNGs never reaches it. Scan each loaded extension itself.
@testset "R1 extension ambiguities" begin
    for name in
        (:PureRNGsReactantExt, :PureRNGsReactantDistributionsExt, :PureRNGsDistributionsExt)
        extension = Base.get_extension(PureRNGs, name)
        @testset "$name" begin
            @test isempty(Test.detect_ambiguities(extension; recursive = true))
        end
    end
end

@testset "Reactant transform result class" begin
    for T in (Float32, Float64)
        expected = one(T)
        @test _same_transform_class(nextfloat(expected), expected)
        @test !_same_transform_class(-expected, expected)
        @test !_same_transform_class(zero(T), expected)
        @test _same_transform_class(zero(T), zero(T))
        @test _same_transform_class(-zero(T), -zero(T))
        @test !_same_transform_class(-zero(T), zero(T))
        @test !_same_transform_class(nextfloat(zero(T)), zero(T))
        @test !_same_transform_class(T(Inf), expected)
        @test !_same_transform_class(T(-Inf), -expected)
        @test !_same_transform_class(T(NaN), expected)
        @test _same_transform_value(nextfloat(expected), expected)
        @test _same_transform_value(nextfloat(nextfloat(expected)), expected) ==
              (REACTANT_TEST_BACKEND == "gpu")
    end
end

if Philox4x64 in SELECTED_GENERATORS
    @testset "R42 wide addressed index" begin
        index = (big(1) << 122) + 1
        eager = Philox4x64(0x123456)
        carrier = Reactant.to_rarray(eager)
        compiled = Reactant.@compile sync = true _large_addressed_uint64(carrier)
        @test UInt64(compiled(carrier)) == _large_addressed_uint64(eager)
    end
end

if Philox2x64 in SELECTED_GENERATORS
    @testset "R43 Reactant normal primitive conformance" begin
        compile_rng = _positioned_at(Philox2x64(0x123456), UInt64(3), UInt16(17))
        central_rng = _positioned_at(Philox2x64(0x654321), UInt64(7), UInt16(29))
        tail_rng = Philox2x64(0x5)
        compile_carrier = Reactant.to_rarray(compile_rng)
        compiled = Reactant.@compile sync = true _normal_probe64(compile_carrier)

        central = compiled(Reactant.to_rarray(central_rng))
        central_expected = _normal_at_observation(central_rng, Float64, 3)
        @test _same_normal_observation(central, central_expected)
        @test UInt64(central[1]) == 0x000c083e66d3d8ef
        @test reinterpret(UInt64, Float64(central[2])) == 0x3fe8107ccda7b1df
        @test reinterpret(UInt64, central_expected[3]) == 0x3fe5c96a5e314305

        tail = compiled(Reactant.to_rarray(tail_rng))
        tail_expected = _normal_at_observation(tail_rng, Float64, 3)
        @test _same_normal_observation(tail, tail_expected)
        @test UInt64(tail[1]) == 0x00009718cb653166
        @test reinterpret(UInt64, Float64(tail[2])) == 0x3fa2e3196ca62cd0
    end
end

@testset "R42 Reactant primitive and state conformance" begin
    for F in SELECTED_GENERATORS
        @testset "$F" begin
            first = _positioned_at(F(0x123456), UInt64(3), UInt64(2), UInt16(17))
            second = _positioned_at(F(0x654321), UInt64(7), UInt64(5), UInt16(29))
            first_carrier = Reactant.to_rarray(first)
            second_carrier = Reactant.to_rarray(second)
            compiled = Reactant.@compile sync = true _snapshot(first_carrier)
            first_got = compiled(first_carrier)
            first_expected = _snapshot(first)
            @test _same_snapshot(first_got, first_expected)
            @test _same_value(compiled(first_carrier), first_got)
            @test _same_snapshot(compiled(second_carrier), _snapshot(second))
            if F === Philox2x32
                endpoint = Philox2x32(UInt64(0x55b8cc))
                endpoint_expected = _snapshot(endpoint)
                # This seed extracts raw 0, the smallest draw on the lattice.
                @test reinterpret(
                    UInt32,
                    _snapshot_pure_exponential32(endpoint_expected),
                ) == 0x33800000
                @test _same_snapshot(
                    compiled(Reactant.to_rarray(endpoint)),
                    endpoint_expected,
                )
            end
            got_next = _snapshot_continuation(first_got)
            expected_next = _snapshot_continuation(first_expected)
            @test _same_snapshot(compiled(got_next), _snapshot(expected_next))
        end
    end
end

@testset "R42 public HLO keeps state dynamic and omits preflight" begin
    cases = ((Philox2x32, 4), (Philox4x32, 5), (Philox4x64, 6))
    forbidden = (
        "stablehlo.custom_call",
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
        F in SELECTED_GENERATORS || continue
        carrier = Reactant.to_rarray(_positioned_at(F(0x123456), UInt64(3), UInt16(17)))
        hlo = String(Reactant.@code_hlo optimize = false rand_next(carrier, UInt64))
        state_type = "tensor<$(state_length)xui64>"

        main = match(r"func\.func @main\([^)]*\)", hlo)
        @test main !== nothing
        @test startswith(main.match, "func.func @main(%arg0: $state_type")
        @test !occursin("%arg1", main.match)
        @test length(findall("sizes = [1]", hlo)) >= state_length
        @test length(findall(state_type, hlo)) >= 2
        @test all(pattern -> !occursin(pattern, hlo), forbidden)
        # The core body is one shared private function beside `main`, called
        # once per block of the two-block window.
        @test count("func.func", hlo) == 2
        @test count("call @", hlo) == 2
    end
end

@testset "R42 fills trace no per-element constants" begin
    # A dense literal with one entry per element grows with the fill and
    # fails Reactant's constant size cap above 13 million elements.
    for F in SELECTED_GENERATORS
        carrier = Reactant.to_rarray(_positioned_at(F(0x123456), UInt64(3), UInt16(17)))
        for draw in (
            rng -> rand(rng, Float64, 4096),
            rng -> randn(rng, Float32, 4096),
            rng -> rand(rng, Int16(-31):Int16(3):Int16(41), 4096),
        )
            hlo = String(Reactant.@code_hlo optimize = false draw(carrier))
            @test !occursin(r"dense<\[[^\]]{1024,}", hlo)
        end
    end
end

@testset "R42 fixed distributions" begin
    for F in SELECTED_GENERATORS
        @testset "$F" begin
            first = _positioned_at(F(0x123456), UInt64(3), UInt16(17))
            second = _positioned_at(F(0x654321), UInt64(7), UInt16(29))
            first_carrier = Reactant.to_rarray(first)
            second_carrier = Reactant.to_rarray(second)

            fixed =
                Reactant.@compile sync = true _fixed_distribution_snapshot(first_carrier)
            _check_distribution_executable(
                "regular",
                fixed,
                _fixed_distribution_snapshot,
                FIXED_DISTRIBUTIONS,
                _same_distribution_mapping,
                first,
                second,
                first_carrier,
                second_carrier,
            )

            normal =
                Reactant.@compile sync = true _normal_distribution_snapshot(first_carrier)
            _check_distribution_executable(
                "Normal",
                normal,
                _normal_distribution_snapshot,
                NORMAL_DISTRIBUTIONS,
                _same_normal_mapping,
                first,
                second,
                first_carrier,
                second_carrier,
            )

            subnormal = Reactant.@compile sync = true _subnormal_distribution_snapshot(
                first_carrier,
            )
            _check_distribution_executable(
                "subnormal",
                subnormal,
                _subnormal_distribution_snapshot,
                SUBNORMAL_DISTRIBUTIONS,
                _same_distribution_mapping,
                first,
                second,
                first_carrier,
                second_carrier,
            )
        end
    end
end

if Philox4x32 in SELECTED_GENERATORS
    include("distribution_transforms.jl")
    @testset "R42 added continuous fixed distributions" begin
        eager = _positioned_at(Philox4x32(0x123456), UInt64(3), UInt16(17))
        carrier = Reactant.to_rarray(eager)
        compiled = Reactant.@compile sync = true _continuous_distribution_snapshot(carrier)
        got = compiled(carrier)
        expected = _continuous_distribution_snapshot(eager)
        for (got_value, expected_value) in zip(got, expected)
            @test _same_continuous_mapping(got_value, expected_value)
        end
        @test _same_value(compiled(carrier), got)
    end
end

if Philox2x32 in SELECTED_GENERATORS
    @testset "R42 subnormal and cancellation mappings" begin
        root = _positioned_at(Philox2x32(0x123456), UInt64(3), UInt64(2), UInt16(17))
        carrier = Reactant.to_rarray(root)
        compiled =
            Reactant.@compile sync = true _cancellation_distribution_snapshot(carrier)
        got = compiled(carrier)
        expected = _cancellation_distribution_snapshot(root)
        @test _same_cancellation_distribution_snapshot(got, expected)
        @test _same_value(compiled(carrier), got)
        for (pair, distribution) in zip(expected, CANCELLATION_NORMAL_DISTRIBUTIONS)
            primitive = pair[1][3]
            fused = fma(distribution.σ, primitive, distribution.μ)
            separate = distribution.σ * primitive + distribution.μ
            @test !_same_value(fused, separate)
        end

        bernoulli_root =
            _positioned_at(Philox2x32(0x123456), UInt64(0x01852ed9), UInt64(2), UInt16(0))
        bernoulli_carrier = Reactant.to_rarray(bernoulli_root)
        bernoulli_compiled =
            Reactant.@compile sync = true _subnormal_bernoulli_snapshot(bernoulli_carrier)
        bernoulli_got = bernoulli_compiled(bernoulli_carrier)
        bernoulli_expected = _subnormal_bernoulli_snapshot(bernoulli_root)
        @test _same_value(bernoulli_expected[1], zero(Float32))
        @test _same_distribution_mapping(
            bernoulli_got,
            bernoulli_expected,
            SUBNORMAL_BERNOULLI,
        )
        @test _same_value(bernoulli_compiled(bernoulli_carrier), bernoulli_got)
    end
end

if Philox4x32 in SELECTED_GENERATORS
    @testset "R42 integer range method surface" begin
        carrier =
            Reactant.to_rarray(_positioned_at(Philox4x32(0x123456), UInt64(3), UInt16(17)))
        unsupported = NegativeIntegerRange(-5, -2, 4)
        @test !applicable(rand, carrier, unsupported)
        @test !applicable(rand_next, carrier, unsupported)
        for range in (UInt16(2):UInt16(3):UInt16(74), LinRange{Int64}(-20, 20, 5))
            @test applicable(rand, carrier, range)
            @test applicable(rand_next, carrier, range)
        end
    end
end

@testset "R42 exact-end continuation" begin
    for F in (Philox2x32, Philox4x32, Philox4x64)
        F in SELECTED_GENERATORS || continue
        eager = _last_bit_rng(F)
        eager_value, eager_next = rand_next(eager, Bool)
        carrier = Reactant.to_rarray(eager)
        compiled = Reactant.@compile sync = true rand_next(carrier, Bool)
        value, next_carrier = compiled(carrier, Bool)
        terminal_carrier = Reactant.to_rarray(eager_next)
        @test value == eager_value
        @test Array(next_carrier.state) == Array(terminal_carrier.state)
    end
end

@testset "R42 eager exhaustion remains checked with Reactant loaded" begin
    for F in SELECTED_GENERATORS
        last = _last_bit_rng(F)
        value, terminal = rand_next(last, Bool)
        terminal_state = Reactant.to_rarray(terminal).state |> Array

        @test value == rand(last, Bool)
        @test_throws StreamExhausted rand(terminal, Bool)
        @test_throws StreamExhausted rand_next(terminal, Bool)
        @test Array(Reactant.to_rarray(terminal).state) == terminal_state
    end
end

function _fill_snapshot(rng)
    uniform = (
        rand(rng, Bool, 9),
        rand(rng, UInt32, 5),
        rand(rng, Int32, 5),
        rand(rng, UInt64, 3, 2),
        rand(rng, Int64, 4),
        rand(rng, Float32, 70),
        rand(rng, Float64, 33),
    )
    normals = (randn(rng, Float32, 6), randn(rng, Float64, 2, 3))
    exponentials = (randexp(rng, Float32, 6), randexp(rng, Float64, 5))
    values, next_rng = rand_next(rng, Float64, 7)
    return uniform, normals, exponentials, values, next_rng
end

# A fill traces the transform over a vector, and XLA's vectorized
# single-precision `log` is a polynomial approximation, so tail normals of a
# `Float32` fill can differ from the eager value by a few ulps on the CPU
# backend. Every other fill value is exact.
function _same_fill_transform_value(got, expected::T) where {T<:Union{Float32,Float64}}
    REACTANT_TEST_BACKEND == "cpu" || return _same_transform_class(got, expected)
    value = T(got)
    U = Base.uinttype(T)
    distance = abs(Int128(reinterpret(U, value)) - Int128(reinterpret(U, expected)))
    return distance <= 8
end

function _same_fill_snapshot(got, expected)
    uniform = all(zip(got[1], expected[1])) do (array, reference)
        size(array) == size(reference) && Array(array) == reference
    end
    pairs = zip((got[2]..., got[3]...), (expected[2]..., expected[3]...))
    transformed = all(pairs) do (array, reference)
        size(array) == size(reference) &&
            all(_same_fill_transform_value.(Array(array), reference))
    end
    return uniform &&
           transformed &&
           Array(got[4]) == expected[4] &&
           _same_value(got[5], expected[5])
end

@testset "R42 array fills" begin
    for F in SELECTED_GENERATORS
        @testset "$F" begin
            first = _positioned_at(F(0x123456), UInt64(3), UInt64(2), UInt16(17))
            second = _positioned_at(F(0x654321), UInt64(7), UInt64(5), UInt16(29))
            first_carrier = Reactant.to_rarray(first)
            compiled = Reactant.@compile sync = true _fill_snapshot(first_carrier)
            @test _same_fill_snapshot(compiled(first_carrier), _fill_snapshot(first))
            @test _same_fill_snapshot(
                compiled(Reactant.to_rarray(second)),
                _fill_snapshot(second),
            )
            expected = rand(first, Float32, 70)
            @test expected == [rand_at(first, Float32, i) for i = 1:70]
        end
    end
end

# Range arrays, samples, and destination fills consume the stream exactly as
# the eager fills do, including the 128-bit reduction for spans above 2^32.
function _range_snapshot(rng)
    narrow = rand(rng, 1:1000, 7)
    stepped, after_stepped = rand_next(rng, Int16(-3):Int16(2):Int16(40), 2, 3)
    wide = rand(rng, UInt64(0):(UInt64(1)<<40), 5)
    linear = rand(rng, LinRange{Int64}(-20, 20, 5), 4)
    values = randsample(rng, [1.5, 2.5, 3.5], 6)
    whole, after_whole = randsample_next(rng, 10:20)
    tupled =
        (rand(rng, Float32, (2, 3)), rand(rng, 1:1000, (3, 2)), rand_next(rng, (4,))...)
    addressed = rand_at(rng, Float64, 3:7)
    return narrow,
    stepped,
    after_stepped,
    wide,
    linear,
    values,
    whole,
    after_whole,
    tupled...,
    addressed
end

_addressed_transforms(rng) = (randn_at(rng, Float32, 2:5), randexp_at(rng, Float64, 4:9))

function _destination_snapshot(rng, uniform, normal, exponential)
    _, after_uniform = rand_next!(rng, uniform)
    _, after_normal = randn_next!(after_uniform, normal)
    randexp!(after_normal, exponential)
    return uniform, normal, exponential, after_normal
end

function _same_snapshot_item(got, expected)
    expected isa AbstractPureRNG && return _same_value(got, expected)
    return size(got) == size(expected) && Array(got) == expected
end

@testset "R42 range arrays, samples, and destination fills" begin
    for F in SELECTED_GENERATORS
        @testset "$F" begin
            eager = _positioned_at(F(0x123456), UInt64(3), UInt64(2), UInt16(17))
            carrier = Reactant.to_rarray(eager)
            compiled = Reactant.@compile sync = true _range_snapshot(carrier)
            @test all(_same_snapshot_item.(compiled(carrier), _range_snapshot(eager)))
            compiled_at = Reactant.@compile sync = true _addressed_transforms(carrier)
            for (got, expected) in zip(compiled_at(carrier), _addressed_transforms(eager))
                @test all(_same_fill_transform_value.(Array(got), expected))
            end

            sizes = (zeros(Float32, 3, 4), zeros(Float64, 5), zeros(Float32, 6))
            traced = map(Reactant.to_rarray, sizes)
            compiled_fill =
                Reactant.@compile sync = true _destination_snapshot(carrier, traced...)
            got = compiled_fill(carrier, traced...)
            expected = _destination_snapshot(eager, map(copy, sizes)...)
            @test _same_snapshot_item(got[1], expected[1])
            @test all(_same_fill_transform_value.(Array(got[2]), expected[2]))
            @test all(_same_fill_transform_value.(Array(got[3]), expected[3]))
            @test _same_value(got[4], expected[4])
        end
    end
end

_empty_uniform(carrier) = rand_next(carrier, Float64, 0)
_empty_matrix(carrier) = rand_next(carrier, Float32, 0, 3)
_empty_range(carrier) = rand_next(carrier, Int32(1):Int32(6), 0)

@testset "empty traced draws compile and consume no bits" begin
    carrier = Reactant.to_rarray(Philox4x32(0x9a2))
    for (probe, expected_size) in
        ((_empty_uniform, (0,)), (_empty_matrix, (0, 3)), (_empty_range, (0,)))
        values, next_carrier = (Reactant.@compile sync = true probe(carrier))(carrier)
        @test size(values) == expected_size
        @test Array(next_carrier.state) == Array(carrier.state)
    end
end
