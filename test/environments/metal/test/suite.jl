using Distributions
using KernelAbstractions
using PureRNGs
using Metal
using MLDataDevices
using Random
using Test

const IR = PureRNGs
const METAL_32_GENERATORS = (Philox2x32, Philox4x32, Threefry2x32, Threefry4x32, ChaCha)
const METAL_64_GENERATORS = (Philox2x64, Philox4x64, Threefry2x64, Threefry4x64)
const METAL_GENERATORS = (METAL_32_GENERATORS..., METAL_64_GENERATORS...)
const METAL_FIXED_DISTRIBUTIONS = (
    (Normal{Float32}(0.5f0, 1.25f0), Float32),
    (Normal{Float64}(0.5, 1.25), Float64),
    (Uniform{Float32}(-1.0f0, 2.0f0), Float32),
    (Uniform{Float64}(-1.0, 2.0), Float64),
    (Exponential{Float32}(1.25f0), Float32),
    (Exponential{Float64}(1.25), Float64),
    (Bernoulli{Float32}(0.25f0), Bool),
    (Bernoulli{Float64}(0.25), Bool),
    (DiscreteUniform(2, 9), Int),
)
const METAL_EXPONENTIAL_GOLDEN_BLOCK = UInt64(0x00123456789abcde)
const METAL_EXPONENTIAL_GOLDEN_BIT = UInt16(61)
const METAL_EXPONENTIAL_GOLDEN_CASES = (
    (Philox2x32, (UInt32(0x01234567),), UInt32(0x79cb04)),
    (Philox4x32, (UInt32(0x01234567), UInt32(0x89abcdef)), UInt32(0x1014e4)),
    (Threefry2x32, (UInt32(0x01234567), UInt32(0x89abcdef)), UInt32(0x6a587f)),
    (
        Threefry4x32,
        (UInt32(0x01234567), UInt32(0x89abcdef), UInt32(0xfedcba98), UInt32(0x76543210)),
        UInt32(0x06bfc7),
    ),
)
const METAL_EXPONENTIAL_LATTICE_LENGTH = 1 << 23
const METAL_OFFSET_GENERATORS = (Philox4x32, Threefry2x32)
const METAL_OFFSET_BITS = (UInt16(0), UInt16(17), UInt16(31), UInt16(63))
const METAL_OFFSET_TYPES = (Float32, UInt32)
const METAL_OFFSET_LENGTH = 37

struct MetalDeviceArrayProbe{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(array::MetalDeviceArrayProbe) = size(array.data)
Base.getindex(array::MetalDeviceArrayProbe, index::Int) = array.data[index]
Base.setindex!(array::MetalDeviceArrayProbe, value, index::Int) =
    setindex!(array.data, value, index)
MLDataDevices.get_device_type(::MetalDeviceArrayProbe) = MetalDevice
MLDataDevices.get_device(::MetalDeviceArrayProbe) = MetalDevice()

# The Metal-token normal of the bits `rng` holds, evaluated on the host.
function _metal_normal(rng, ::Type{T}) where {T}
    width = T === Float32 ? Val(23) : Val(52)
    raw = IR._extract_bits_unchecked(
        rng,
        IR._position_block(rng.position),
        rng.position.bit,
        width,
    )
    return IR._normal_from_bits(IR._METAL_BACKEND, T, raw)
end

function _check_metal_error(f)
    @test_throws ArgumentError f()
    return nothing
end

function _positioned(rng, block::UInt64, bit::UInt16)
    position =
        rng.position isa IR._Position64 ? IR._Position64(block, bit) :
        IR._Position128(block, UInt64(2), bit)
    return IR._rebuild(rng, position, rng.device)
end

function _metal_exponential_golden_rng(F, key)
    base = F(key)
    position = if base.position isa IR._Position64
        IR._Position64(METAL_EXPONENTIAL_GOLDEN_BLOCK, METAL_EXPONENTIAL_GOLDEN_BIT)
    else
        IR._Position128(METAL_EXPONENTIAL_GOLDEN_BLOCK, UInt64(0), METAL_EXPONENTIAL_GOLDEN_BIT)
    end
    return MetalDevice()(IR._rebuild(base, position, base.device))
end

@inline function _metal_exponential_ulp_error(value::Float32, raw::UInt32, scale::BigFloat)
    reference = -log(one(BigFloat) - BigFloat(2raw + 1) * scale)
    return abs(BigFloat(value) - reference) / BigFloat(eps(value))
end

function _metal_exponential_max_ulp(values::Vector{Float32})
    return setprecision(BigFloat, 160) do
        scale = ldexp(one(BigFloat), -24)
        maximum_error = zero(BigFloat)
        for (index, value) in pairs(values)
            raw = UInt32(index - 1)
            maximum_error =
                max(maximum_error, _metal_exponential_ulp_error(value, raw, scale))
        end
        maximum_error
    end
end

KernelAbstractions.@kernel function _metal_exponential_lattice_kernel!(values)
    index = KernelAbstractions.@index(Global, Linear)
    raw = UInt64(index - 1)
    @inbounds values[index] = IR._exponential_from_bits(IR._METAL_BACKEND, Float32, raw)
end

# An extension is not a submodule of its parent, so a recursive scan that starts
# at PureRNGs never reaches it. Scan each loaded extension itself.
@testset "extension ambiguities" begin
    for name in
        (:PureRNGsMetalExt, :PureRNGsDistributionsExt, :PureRNGsKernelAbstractionsExt)
        extension = Base.get_extension(IR, name)
        @testset "$name" begin
            @test isempty(Test.detect_ambiguities(extension; recursive = true))
        end
    end
end

@testset "Metal public host surface" begin
    for F in METAL_GENERATORS
        cpu_rng = F(0x816)
        rng = MetalDevice()(cpu_rng)
        @test rng.device === IR._METAL_BACKEND
        for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
            @test rand(rng, T) === rand(cpu_rng, T)
            @test IR.rand_next(rng, T)[1] === IR.rand_next(cpu_rng, T)[1]
        end
        for T in (Float32, Float64)
            # The Metal token selects Giles' erfinv, so a host draw on a
            # Metal generator no longer equals the CPU normal on the same bits.
            @test randn(rng, T) === _metal_normal(cpu_rng, T)
            @test isapprox(randexp(rng, T), randexp(cpu_rng, T); rtol = 16eps(T))
        end
        @test rand(rng, UInt16(1):UInt16(3)) === rand(cpu_rng, UInt16(1):UInt16(3))
    end
end

@testset "Metal allocating exclusions" begin
    for F in METAL_32_GENERATORS
        rng = MetalDevice()(F(0x817))
        for count in (0, 1)
            _check_metal_error(() -> rand(rng, Float64, count))
            _check_metal_error(() -> IR.rand_next(rng, Float64, count))
            _check_metal_error(() -> randn(rng, Float64, count))
            _check_metal_error(() -> IR.randn_next(rng, Float64, count))
            _check_metal_error(() -> randexp(rng, Float64, count))
            _check_metal_error(() -> IR.randexp_next(rng, Float64, count))
        end
        _check_metal_error(() -> IR.rand_next(rng, 0))
        _check_metal_error(() -> IR.randn_next(rng, 0))
        _check_metal_error(() -> IR.randexp_next(rng, 0))
        _check_metal_error(() -> rand(rng, Float64, -1))
        _check_metal_error(() -> IR.rand_next(rng, Float64, -1))
    end

    for F in METAL_64_GENERATORS
        rng = MetalDevice()(F(0x818))
        for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64), count in (0, 1)
            _check_metal_error(() -> rand(rng, T, count))
            _check_metal_error(() -> IR.rand_next(rng, T, count))
        end
        for T in (Float32, Float64), count in (0, 1)
            _check_metal_error(() -> randn(rng, T, count))
            _check_metal_error(() -> IR.randn_next(rng, T, count))
            _check_metal_error(() -> randexp(rng, T, count))
            _check_metal_error(() -> IR.randexp_next(rng, T, count))
        end
    end

    for F in METAL_GENERATORS,
        T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64),
        count in (0, 1)

        rng = MetalDevice()(F(0x819))
        range = T(1):T(3)
        _check_metal_error(() -> rand(rng, range, count))
        _check_metal_error(() -> IR.rand_next(rng, range, count))
    end

    rng = MetalDevice()(Philox4x32(0x81a))
    _check_metal_error(() -> rand(rng, UInt32(2):UInt32(1), 1))
    _check_metal_error(() -> IR.rand_next(rng, UInt32(2):UInt32(1), 1))
    _check_metal_error(() -> rand(rng, UInt32(1):UInt32(3), -1))
    _check_metal_error(() -> IR.rand_next(rng, UInt32(1):UInt32(3), -1))
    @test_throws ArgumentError rand!(rng, UInt32[])
    @test_throws ArgumentError IR.rand_next!(rng, UInt32[])
    @test_throws ArgumentError randn!(rng, Float32[])
    @test_throws ArgumentError IR.randn_next!(rng, Float32[])
    @test_throws ArgumentError randexp!(rng, Float32[])
    @test_throws ArgumentError IR.randexp_next!(rng, Float32[])

    population = MetalDeviceArrayProbe(Int32[1, 2, 3])
    weights = MetalDeviceArrayProbe(Float64[1, 2, 3])
    for count in (0, 1)
        _check_metal_error(() -> randsample(rng, population, weights, count))
        _check_metal_error(() -> randsample_next(rng, population, weights, count))
    end
    _check_metal_error(() -> randsample(rng, population, weights))
    _check_metal_error(() -> randsample_next(rng, population, weights))
    table = WeightTable(Float64[1, 2, 3])
    _check_metal_error(() -> randsample(rng, population, table, 1))
    _check_metal_error(() -> randsample_next(rng, population, table, 1))

    for F in METAL_32_GENERATORS
        rng = MetalDevice()(F(0x81b))
        destination = MetalDeviceArrayProbe(Float64[])
        _check_metal_error(() -> rand!(rng, destination))
        _check_metal_error(() -> IR.rand_next!(rng, destination))
        _check_metal_error(() -> randn!(rng, destination))
        _check_metal_error(() -> IR.randn_next!(rng, destination))
        _check_metal_error(() -> randexp!(rng, destination))
        _check_metal_error(() -> IR.randexp_next!(rng, destination))
    end

    for F in METAL_64_GENERATORS
        rng = MetalDevice()(F(0x81c))
        for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
            destination = MetalDeviceArrayProbe(Vector{T}())
            _check_metal_error(() -> rand!(rng, destination))
            _check_metal_error(() -> IR.rand_next!(rng, destination))
        end
        for T in (Float32, Float64)
            destination = MetalDeviceArrayProbe(Vector{T}())
            _check_metal_error(() -> randn!(rng, destination))
            _check_metal_error(() -> IR.randn_next!(rng, destination))
            _check_metal_error(() -> randexp!(rng, destination))
            _check_metal_error(() -> IR.randexp_next!(rng, destination))
        end
    end

    for F in METAL_GENERATORS
        rng = MetalDevice()(F(0x91c))
        range = UInt32(1):UInt32(7)
        for operation in (IR.randsample, IR.randsample_next)
            _check_metal_error(() -> operation(rng, range))
            _check_metal_error(() -> operation(rng, range, 0))
            _check_metal_error(() -> operation(rng, range, 1))
        end
    end

    metal_rng = MetalDevice()(Philox4x32(0x91d))
    @test_throws ArgumentError IR.randsample(metal_rng, UInt32[1, 2, 3], -1)
end

@testset "Metal fixed-distribution surface" begin
    for F in METAL_GENERATORS, (distribution, result_type) in METAL_FIXED_DISTRIBUTIONS
        rng = MetalDevice()(F(0x91e))
        value = rand(rng, distribution)
        next_value, next_rng = IR.rand_next(rng, distribution)
        @test isequal(value, rand(rng, distribution))
        @test isequal(next_value, value)
        @test isequal(IR.rand_at(rng, distribution, 1), value)
        @test next_rng.device === IR._METAL_BACKEND

        for count in (0, 1)
            _check_metal_error(() -> rand(rng, distribution, count))
            _check_metal_error(() -> IR.rand_next(rng, distribution, count))
        end

        destination = MetalDeviceArrayProbe(Vector{result_type}())
        _check_metal_error(() -> rand!(rng, distribution, destination))
        _check_metal_error(() -> IR.rand_next!(rng, distribution, destination))
    end
end

@testset "Metal excludes added device-executing samplers" begin
    rng = MetalDevice()(Philox4x32(0x91f))
    for distribution in (
        LogNormal(0.0f0, 1.0f0),
        Weibull(2.0f0, 1.0f0),
        Rayleigh(1.0f0),
        Laplace(0.0f0, 1.0f0),
        Logistic(0.0f0, 1.0f0),
        Gumbel(0.0f0, 1.0f0),
        Pareto(2.0f0, 1.0f0),
        Frechet(2.0f0, 1.0f0),
        Cauchy(0.0f0, 1.0f0),
        TriangularDist(0.0f0, 1.0f0, 0.5f0),
        Categorical([0.25, 0.75]),
    )
        T = typeof(rand(rng, distribution))
        destination = MetalDeviceArrayProbe(T[])
        _check_metal_error(() -> rand(rng, distribution, 0))
        _check_metal_error(() -> rand_next!(rng, distribution, destination))
    end
    destination = MetalDeviceArrayProbe(Int[])
    _check_metal_error(() -> randsample!(rng, 1:3, destination))
    _check_metal_error(() -> randsample_next!(rng, 1:3, ones(3), destination))
end

if Metal.functional()
    @testset "Metal served primitive smoke" begin
        served = (
            Bool,
            UInt8,
            Int8,
            UInt16,
            Int16,
            UInt32,
            Int32,
            UInt64,
            Int64,
            Float16,
            Float32,
        )
        for F in METAL_32_GENERATORS, T in served
            cpu_rng = F(0x81b)
            rng = MetalDevice()(cpu_rng)
            values, next_rng = IR.rand_next(rng, T, 17)
            expected, expected_next = IR.rand_next(cpu_rng, T, 17)
            @test values isa Metal.MtlArray{T,1}
            @test Array(values) == expected
            @test next_rng.position == expected_next.position
        end

        for F in METAL_32_GENERATORS, T in (Float16, Float32)
            cpu_rng = F(0x81c)
            rng = MetalDevice()(cpu_rng)
            values, next_rng = IR.randn_next(rng, T, 17)
            repeated, repeat_next = IR.randn_next(rng, T, 17)
            _, expected_next = IR.randn_next(cpu_rng, T, 17)
            @test values isa Metal.MtlArray{T,1}
            @test Array(values) == Array(repeated)
            @test next_rng.position == repeat_next.position == expected_next.position
        end
    end

    @testset "Metal Float32 exponential" begin
        for (F, key, raw) in METAL_EXPONENTIAL_GOLDEN_CASES
            rng = _metal_exponential_golden_rng(F, key)
            block = IR._position_block(rng.position)
            extracted = IR._extract_bits_unchecked(rng, block, rng.position.bit, Val(23))
            @test extracted == UInt64(raw)

            device_values, next_rng = IR.randexp_next(rng, Float32, 1)
            value = only(Array(device_values))
            scale = setprecision(BigFloat, 160) do
                ldexp(one(BigFloat), -24)
            end
            @test device_values isa Metal.MtlArray{Float32,1}
            @test next_rng.position ==
                  IR._advance_position_unchecked(rng, UInt64(23), UInt64(0))
            @test setprecision(BigFloat, 160) do
                _metal_exponential_ulp_error(value, raw, scale) <= BigFloat(3)
            end
        end

        device_lattice = Metal.MtlArray{Float32}(undef, METAL_EXPONENTIAL_LATTICE_LENGTH)
        backend = KernelAbstractions.get_backend(device_lattice)
        _metal_exponential_lattice_kernel!(backend)(
            device_lattice;
            ndrange = METAL_EXPONENTIAL_LATTICE_LENGTH,
        )
        KernelAbstractions.synchronize(backend)
        lattice = Array(device_lattice)

        @test isfinite(last(lattice))
        @test all(isfinite, lattice)
        @test all(value -> value > zero(Float32), lattice)
        @test issorted(lattice)
        @test setprecision(BigFloat, 160) do
            scale = ldexp(one(BigFloat), -24)
            _metal_exponential_ulp_error(last(lattice), UInt32(0x7fffff), scale) <=
            BigFloat(3)
        end
        maximum_ulp = _metal_exponential_max_ulp(lattice)
        @test maximum_ulp <= BigFloat(3)
    end

    @testset "Metal mid-stream fill parity" begin
        for F in METAL_OFFSET_GENERATORS
            for bit in METAL_OFFSET_BITS, T in METAL_OFFSET_TYPES
                cpu_rng = _positioned(F(0x5150), UInt64(9), bit)
                rng = MetalDevice()(cpu_rng)
                expected, expected_next = IR.rand_next(cpu_rng, T, METAL_OFFSET_LENGTH)
                values, next_rng = IR.rand_next(rng, T, METAL_OFFSET_LENGTH)
                @test Array(values) == expected
                @test next_rng.position == expected_next.position
            end

            # Metal evaluates the Giles tail branch with its own Float32 log, so
            # the normals carry the same 3 ulp budget as the Metal exponential.
            cpu_rng = _positioned(F(0x5151), UInt64(9), UInt16(17))
            rng = MetalDevice()(cpu_rng)
            expected = Vector{Float32}(undef, METAL_OFFSET_LENGTH)
            cursor = cpu_rng
            for index in eachindex(expected)
                expected[index] = _metal_normal(cursor, Float32)
                _, cursor = IR.randn_next(cursor, Float32)
            end
            expected_next = cursor
            values, next_rng = IR.randn_next(rng, Float32, METAL_OFFSET_LENGTH)
            host = Array(values)
            @test all(
                index -> abs(host[index] - expected[index]) <= 3 * eps(expected[index]),
                eachindex(expected),
            )
            @test next_rng.position == expected_next.position
        end
    end
else
    @info "Metal hardware unavailable; served device execution was not run"
end
