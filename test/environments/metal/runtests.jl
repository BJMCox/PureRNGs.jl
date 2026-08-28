using Distributions
using PureRNGs
using Metal
using MLDataDevices
using Random
using Test

const IR = PureRNGs
const METAL_32_FAMILIES = (Philox2x32, Philox4x32, Threefry2x32, Threefry4x32)
const METAL_64_FAMILIES = (Philox2x64, Philox4x64, Threefry2x64, Threefry4x64)
const METAL_FAMILIES = (METAL_32_FAMILIES..., METAL_64_FAMILIES...)
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
    (Philox2x32, (UInt32(0x01234567),), UInt32(0xa05803)),
    (Philox4x32, (UInt32(0x01234567), UInt32(0x89abcdef)), UInt32(0xda96ce)),
    (Threefry2x32, (UInt32(0x01234567), UInt32(0x89abcdef)), UInt32(0x98edd2)),
    (
        Threefry4x32,
        (UInt32(0x01234567), UInt32(0x89abcdef), UInt32(0xfedcba98), UInt32(0x76543210)),
        UInt32(0xc12127),
    ),
)
const METAL_EXPONENTIAL_LATTICE_LENGTH = 1 << 24

struct MetalDeviceArrayProbe{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(array::MetalDeviceArrayProbe) = size(array.data)
Base.getindex(array::MetalDeviceArrayProbe, index::Int) = array.data[index]
Base.setindex!(array::MetalDeviceArrayProbe, value, index::Int) =
    setindex!(array.data, value, index)
MLDataDevices.get_device_type(::MetalDeviceArrayProbe) = MetalDevice
MLDataDevices.get_device(::MetalDeviceArrayProbe) = MetalDevice()

function _check_metal_error(f)
    error = try
        f()
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin("Metal", sprint(showerror, error))
    return nothing
end

@inline _is_purerngs_method(method) =
    method.module === IR || startswith(string(nameof(method.module)), "PureRNGs")

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
    reference = -log(one(BigFloat) - BigFloat(raw) * scale)
    denominator = iszero(value) ? BigFloat(floatmin(Float32)) : BigFloat(eps(value))
    return abs(BigFloat(value) - reference) / denominator
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

IR.KernelAbstractions.@kernel function _metal_exponential_lattice_kernel!(values)
    index = IR.KernelAbstractions.@index(Global, Linear)
    raw = UInt64(index - 1)
    @inbounds values[index] = IR._exponential_from_bits(IR._METAL_BACKEND, Float32, raw)
end

@testset "R37-R39 Metal extension host surface" begin
    extension_module = Base.get_extension(IR, :PureRNGsMetalExt)
    @test extension_module !== nothing

    for F in METAL_FAMILIES
        cpu_rng = F(0x816)
        rng = MetalDevice()(cpu_rng)
        @test rng.device === IR._METAL_BACKEND
        @test isbits(rng.device)
        @test sizeof(rng.device) == 0
        @test which(IR.rand_next, (typeof(rng), Int)).module === IR
        @test which(IR.randn_next, (typeof(rng), Int)).module === IR

        for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
            @test rand(rng, T) === rand(cpu_rng, T)
            @test first(IR.rand_next(rng, T)).device === IR._METAL_BACKEND
            @test last(IR.rand_next(rng, T)) === last(IR.rand_next(cpu_rng, T))
            @test which(rand, (typeof(rng), Type{T}, Int)).module === IR
            @test which(IR.rand_next, (typeof(rng), Type{T}, Int)).module === IR
        end
        for T in (Float32, Float64)
            @test randn(rng, T) === randn(cpu_rng, T)
            @test last(IR.randn_next(rng, T)) === last(IR.randn_next(cpu_rng, T))
            @test which(randn, (typeof(rng), Type{T}, Int)).module === IR
            @test which(IR.randn_next, (typeof(rng), Type{T}, Int)).module === IR

            exponential = randexp(rng, T)
            next_rng, next_exponential = IR.randexp_next(rng, T)
            @test isequal(exponential, randexp(rng, T))
            @test isequal(next_exponential, exponential)
            @test isequal(IR.randexpat(rng, T, 1), exponential)
            @test next_rng.device === IR._METAL_BACKEND
            @test next_rng.position == first(IR.randexp_next(cpu_rng, T)).position
            @test which(randexp, (typeof(rng), Type{T}, Int)).module === IR
            @test which(IR.randexp_next, (typeof(rng), Type{T}, Int)).module === IR
        end
        for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
            range = T(1):T(3)
            @test rand(rng, range) === rand(cpu_rng, range)
            @test last(IR.rand_next(rng, range)) === last(IR.rand_next(cpu_rng, range))
            @test which(rand, (typeof(rng), typeof(range), Int)).module === IR
            @test which(IR.rand_next, (typeof(rng), typeof(range), Int)).module === IR
        end
    end

    ambiguities = filter(Test.detect_ambiguities(IR, Random; recursive = true)) do pair
        any(_is_purerngs_method, pair)
    end
    @test isempty(ambiguities)
end

@testset "R41 Metal static serviceability and ownership" begin
    extension_module = Base.get_extension(IR, :PureRNGsMetalExt)
    @test extension_module !== nothing

    for F in METAL_32_FAMILIES
        rng = MetalDevice()(F(0x916))
        for T in (Bool, UInt32, Int32, UInt64, Int64, Float32)
            method = which(IR._check_serviceability, (typeof(rng), Type{T}))
            @test method.module === extension_module
            @test IR._check_serviceability(rng, T) === nothing
        end
    end

    for F in METAL_64_FAMILIES
        rng = MetalDevice()(F(0x917))
        for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
            method = which(IR._check_serviceability, (typeof(rng), Type{T}))
            @test method.module === extension_module
            _check_metal_error(() -> IR._check_serviceability(rng, T))
        end
    end

    rng = MetalDevice()(Philox4x32(0x918))
    @test which(IR._check_serviceability, (typeof(rng), Type{Float64})).module ===
          extension_module
    @test which(IR._check_serviceability, (typeof(rng), UnitRange{UInt32})).module ===
          extension_module
    @test which(IR._check_sampling_serviceability, (typeof(rng),)).module ===
          extension_module
    @test which(IR._allocate_array, (IR._MetalBackend, Type{UInt32}, Tuple{Int})).module ===
          extension_module
end

@testset "R41 Metal allocating exclusions" begin
    for F in METAL_32_FAMILIES
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

    for F in METAL_64_FAMILIES
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

    for F in METAL_FAMILIES,
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
    @test_throws TypeError rand!(rng, UInt32[]; threaded = 1)

    population = MetalDeviceArrayProbe(Int32[1, 2, 3])
    weights = MetalDeviceArrayProbe(Float64[1, 2, 3])
    for count in (0, 1)
        _check_metal_error(() -> randsample(rng, population, weights, count))
        _check_metal_error(() -> randsample_next(rng, population, weights, count))
    end
    _check_metal_error(() -> randsample(rng, population, weights))
    _check_metal_error(() -> randsample_next(rng, population, weights))

    for F in METAL_32_FAMILIES
        rng = MetalDevice()(F(0x81b))
        destination = MetalDeviceArrayProbe(Float64[])
        _check_metal_error(() -> rand!(rng, destination))
        _check_metal_error(() -> IR.rand_next!(rng, destination))
        _check_metal_error(() -> randn!(rng, destination))
        _check_metal_error(() -> IR.randn_next!(rng, destination))
        _check_metal_error(() -> randexp!(rng, destination))
        _check_metal_error(() -> IR.randexp_next!(rng, destination))
    end

    for F in METAL_64_FAMILIES
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

    for F in METAL_FAMILIES
        rng = MetalDevice()(F(0x91c))
        range = UInt32(1):UInt32(7)
        for operation in (IR.randsample, IR.randsample_next)
            _check_metal_error(() -> operation(rng, range))
            _check_metal_error(() -> operation(rng, range, 0))
            _check_metal_error(() -> operation(rng, range, 1))
        end
    end

    metal_rng = MetalDevice()(Philox4x32(0x91d))
    error = try
        IR.randsample(metal_rng, UInt32[1, 2, 3], -1)
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin("device", sprint(showerror, error))
    @test !occursin("Metal", sprint(showerror, error))
end

@testset "R41-R64 Metal fixed-distribution surface" begin
    extension_module = Base.get_extension(IR, :PureRNGsDistributionsExt)
    @test extension_module !== nothing

    for F in METAL_FAMILIES, (distribution, result_type) in METAL_FIXED_DISTRIBUTIONS
        rng = MetalDevice()(F(0x91e))
        value = rand(rng, distribution)
        next_rng, next_value = IR.rand_next(rng, distribution)
        @test value isa result_type
        @test isequal(value, rand(rng, distribution))
        @test isequal(next_value, value)
        @test isequal(IR.randat(rng, distribution, 1), value)
        @test next_rng.device === IR._METAL_BACKEND
        @test which(rand, (typeof(rng), typeof(distribution))).module === extension_module
        @test which(IR.rand_next, (typeof(rng), typeof(distribution))).module ===
              extension_module
        @test which(IR.randat, (typeof(rng), typeof(distribution), Int)).module ===
              extension_module
        @test which(rand, (typeof(rng), typeof(distribution), Int)).module ===
              extension_module
        @test which(IR.rand_next, (typeof(rng), typeof(distribution), Int)).module ===
              extension_module

        for count in (0, 1)
            _check_metal_error(() -> rand(rng, distribution, count))
            _check_metal_error(() -> IR.rand_next(rng, distribution, count))
        end

        destination = MetalDeviceArrayProbe(Vector{result_type}())
        @test which(rand!, (typeof(rng), typeof(distribution), typeof(destination))).module ===
              extension_module
        @test which(
            IR.rand_next!,
            (typeof(rng), typeof(distribution), typeof(destination)),
        ).module === extension_module
        _check_metal_error(() -> rand!(rng, distribution, destination))
        _check_metal_error(() -> IR.rand_next!(rng, distribution, destination))
    end
end

if Metal.functional()
    @testset "R41 Metal served primitive smoke" begin
        for F in METAL_32_FAMILIES, T in (Bool, UInt32, Int32, UInt64, Int64, Float32)
            cpu_rng = F(0x81b)
            rng = MetalDevice()(cpu_rng)
            next_rng, values = IR.rand_next(rng, T, 17)
            expected_next, expected = IR.rand_next(cpu_rng, T, 17)
            @test values isa Metal.MtlArray{T,1}
            @test Array(values) == expected
            @test next_rng.position == expected_next.position
        end

        for F in METAL_32_FAMILIES
            cpu_rng = F(0x81c)
            rng = MetalDevice()(cpu_rng)
            next_rng, values = IR.randn_next(rng, Float32, 17)
            repeat_next, repeated = IR.randn_next(rng, Float32, 17)
            expected_next, _ = IR.randn_next(cpu_rng, Float32, 17)
            @test values isa Metal.MtlArray{Float32,1}
            @test Array(values) == Array(repeated)
            @test next_rng.position == repeat_next.position == expected_next.position
        end
    end

    @testset "R41-R43-R63 Metal Float32 exponential" begin
        for (F, key, raw) in METAL_EXPONENTIAL_GOLDEN_CASES
            rng = _metal_exponential_golden_rng(F, key)
            block = IR._position_block(rng.position)
            extracted = IR._extract_bits_unchecked(
                rng,
                IR.FAMILY_EXP,
                block,
                rng.position.bit,
                Val(24),
            )
            @test extracted == UInt64(raw)

            next_rng, device_values = IR.randexp_next(rng, Float32, 1)
            value = only(Array(device_values))
            scale = setprecision(BigFloat, 160) do
                ldexp(one(BigFloat), -24)
            end
            @test device_values isa Metal.MtlArray{Float32,1}
            @test next_rng.position ==
                  IR._advance_position_unchecked(rng, UInt64(24), UInt64(0))
            @test setprecision(BigFloat, 160) do
                _metal_exponential_ulp_error(value, raw, scale) <= BigFloat(2)
            end
        end

        device_lattice = Metal.MtlArray{Float32}(undef, METAL_EXPONENTIAL_LATTICE_LENGTH)
        backend = IR._fill_backend(device_lattice)
        _metal_exponential_lattice_kernel!(backend)(
            device_lattice;
            ndrange = METAL_EXPONENTIAL_LATTICE_LENGTH,
        )
        IR.KernelAbstractions.synchronize(backend)
        lattice = Array(device_lattice)

        @test isequal(first(lattice), -zero(Float32))
        @test isfinite(last(lattice))
        @test last(lattice) > zero(Float32)
        @test all(isfinite, lattice)
        @test all(value -> value >= zero(Float32), lattice)
        @test issorted(lattice)
        @test setprecision(BigFloat, 160) do
            scale = ldexp(one(BigFloat), -24)
            _metal_exponential_ulp_error(last(lattice), UInt32(0xffffff), scale) <=
            BigFloat(2)
        end
        maximum_ulp = _metal_exponential_max_ulp(lattice)
        @test maximum_ulp <= BigFloat(2)
    end
else
    @info "Metal hardware unavailable; served device execution was not run"
end
