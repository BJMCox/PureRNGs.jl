using Distributions
using ForwardDiff
using KernelAbstractions
using PureRNGs
using Metal
using MLDataDevices
using Random
using Test

const IR = PureRNGs
const METAL_GENERATORS = (
    Philox2x32,
    Philox4x32,
    Threefry2x32,
    Threefry4x32,
    ChaCha,
    Philox2x64,
    Philox4x64,
    Threefry2x64,
    Threefry4x64,
)
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

# Metal has no Float64 or 128-bit integer arithmetic, so those results and the
# samplers that fold Float64 weights throw before touching the device.
@testset "Metal rejects Float64, 128-bit, and weighted draws" begin
    for F in METAL_GENERATORS
        rng = MetalDevice()(F(0x817))
        for T in (Float64, ComplexF64, UInt128, Int128), count in (0, 1)
            _check_metal_error(() -> rand(rng, T, count))
            _check_metal_error(() -> IR.rand_next(rng, T, count))
            _check_metal_error(() -> rand!(rng, MetalDeviceArrayProbe(Vector{T}())))
        end
        for T in (Float64, ComplexF64), count in (0, 1)
            _check_metal_error(() -> randn(rng, T, count))
            _check_metal_error(() -> IR.randn_next(rng, T, count))
        end
        _check_metal_error(() -> randexp(rng, Float64, 1))
        _check_metal_error(() -> IR.randexp_next!(rng, MetalDeviceArrayProbe(Float64[])))
        _check_metal_error(() -> IR.rand_next(rng, 0))
        _check_metal_error(() -> rand(rng, Int128(1):Int128(3), 1))
        _check_metal_error(() -> rand(rng, Float64, -1))
    end

    rng = MetalDevice()(Philox4x32(0x81a))
    _check_metal_error(() -> rand(rng, UInt32(2):UInt32(1), 1))
    _check_metal_error(() -> rand(rng, UInt32(1):UInt32(3), -1))
    @test_throws ArgumentError rand!(rng, UInt32[])
    @test_throws ArgumentError randn!(rng, Float32[])
    @test_throws ArgumentError IR.randsample(rng, UInt32[1, 2, 3], -1)

    population = MetalDeviceArrayProbe(Int32[1, 2, 3])
    weights = MetalDeviceArrayProbe(Float32[1, 2, 3])
    for count in (0, 1), replace in (true, false)
        _check_metal_error(() -> randsample(rng, population, weights, count; replace))
        _check_metal_error(() -> randsample_next(rng, population, weights, count; replace))
    end
    _check_metal_error(() -> randsample(rng, population, weights))
    _check_metal_error(
        () -> randsample_next!(rng, 1:3, ones(3), MetalDeviceArrayProbe(Int[])),
    )
    _check_metal_error(() -> randsample(rng, population, WeightTable(Float64[1, 2, 3]), 1))
    _check_metal_error(() -> WeightTable(MetalDeviceArrayProbe(Float32[1, 2, 3])))
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
        result_type === Float64 || continue
        for count in (0, 1)
            _check_metal_error(() -> rand(rng, distribution, count))
            _check_metal_error(() -> IR.rand_next(rng, distribution, count))
        end
        destination = MetalDeviceArrayProbe(Vector{result_type}())
        _check_metal_error(() -> IR.rand_next!(rng, distribution, destination))
    end
    rng = MetalDevice()(Philox4x32(0x91f))
    dual = ForwardDiff.Dual(0.5, 1.0)
    for distribution in (
        Categorical([0.25, 0.75]),
        Gamma(2.0, 1.0),
        Dirichlet([0.5, 2.0]),
        Normal(dual, one(dual)),
        Gamma(dual, one(dual)),
    )
        _check_metal_error(() -> rand(rng, distribution, 1))
    end
    destination = MetalDeviceArrayProbe(Vector{typeof(dual)}())
    _check_metal_error(() -> rand!(rng, Normal(dual, one(dual)), destination))
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
            ComplexF16,
            ComplexF32,
            Char,
        )
        for F in METAL_GENERATORS, T in served
            cpu_rng = F(0x81b)
            rng = MetalDevice()(cpu_rng)
            values, next_rng = IR.rand_next(rng, T, 17)
            expected, expected_next = IR.rand_next(cpu_rng, T, 17)
            @test values isa Metal.MtlArray{T,1}
            @test Array(values) == expected
            @test next_rng.position == expected_next.position
        end

        for F in METAL_GENERATORS, T in (Float16, Float32, ComplexF32)
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

    # Uniform-based draws equal the CPU draws. Transforms evaluate Metal's own
    # log and erfinv, and a Gamma acceptance can flip where they round, so
    # nearly all transformed values match.
    @testset "Metal ranges, sampling, and distributions track the CPU" begin
        near(x, y) = isapprox(x, y; rtol = 1.0f-4, atol = 1.0f-6)
        for F in (Philox4x32, Threefry2x64, ChaCha)
            cpu_rng = F(0x81d, 5)
            rng = MetalDevice()(cpu_rng)
            for range in (Int32(-3):Int32(9), UInt64(1):(UInt64(10)^12))
                values, next_rng = IR.rand_next(rng, range, 1000)
                expected, expected_next = IR.rand_next(cpu_rng, range, 1000)
                @test values isa Metal.MtlArray
                @test Array(values) == expected
                @test next_rng.position == expected_next.position
            end
            population = Float32.(1:100)
            for replace in (true, false)
                @test Array(randsample(rng, Metal.MtlArray(population), 50; replace)) ==
                      randsample(cpu_rng, population, 50; replace)
            end
            @test Array(randsample(rng, 1:100, 50)) == randsample(cpu_rng, 1:100, 50)
            for d in (
                Normal(1.0f0, 2.0f0),
                Logistic(0.0f0, 1.0f0),
                Laplace(0.0f0, 1.0f0),
                Bernoulli(0.3f0),
                DiscreteUniform(1, 6),
                Gamma(2.5f0, 1.0f0),
                Gamma(0.3f0, 2.0f0),
                Beta(0.5f0, 0.7f0),
                TDist(3.0f0),
            )
                values, next_rng = IR.rand_next(rng, d, 1000)
                expected, expected_next = IR.rand_next(cpu_rng, d, 1000)
                @test values isa Metal.MtlArray
                @test count(near.(Array(values), expected)) >= 998
                @test next_rng.position == expected_next.position
            end
            dual = ForwardDiff.Dual(2.5f0, 1.0f0)
            for d in (Normal(dual, 1.0f0), Gamma(dual, 1.0f0))
                values = Array(rand(rng, d, 1000))
                expected = rand(cpu_rng, d, 1000)
                @test count(
                    near.(ForwardDiff.value.(values), ForwardDiff.value.(expected)),
                ) >= 998
                @test count(
                    near.(
                        ForwardDiff.partials.(values, 1),
                        ForwardDiff.partials.(expected, 1),
                    ),
                ) >= 998
            end
            for d in (
                Dirichlet(Float32[0.3, 2, 5]),
                MvNormal(Float32[1, 2], Float32[2 0.5; 0.5 1]),
            )
                values, next_rng = IR.rand_next(rng, d, 100)
                expected, expected_next = IR.rand_next(cpu_rng, d, 100)
                @test values isa Metal.MtlArray{Float32,2}
                @test count(near.(Array(values), expected)) >= 0.998 * length(expected)
                @test next_rng.position == expected_next.position
            end
        end
    end

    @testset "Metal permutations equal the CPU permutations" begin
        for F in (Philox4x32, ChaCha), n in (0, 1, 1000)
            cpu_rng = F(0x81e, 3)
            rng = MetalDevice()(cpu_rng)
            permutation, next_rng = randperm_next(rng, n)
            expected, expected_next = randperm_next(cpu_rng, n)
            @test permutation isa Metal.MtlArray{Int,1}
            @test Array(permutation) == expected
            @test next_rng.position == expected_next.position
            @test Array(first(randcycle_next(rng, n))) == first(randcycle_next(cpu_rng, n))
            values = Float32.(1:n)
            @test Array(first(shuffle_next(rng, Metal.MtlArray(values)))) ==
                  first(shuffle_next(cpu_rng, values))
        end
        # Equal keys are resolved on the device as on the CPU, whatever order
        # the device sort leaves them in.
        keys = first(rand_next(Philox4x32(0x81f), UInt64(1):UInt64(300), 20_000))
        host = sortperm(keys)
        IR._resolve_key_ties!(host, keys, Philox4x32(0x81f))
        device = Metal.MtlArray(sortperm(collect(zip(keys, length(keys):-1:1))))
        IR._resolve_key_ties!(
            device,
            Metal.MtlArray(keys),
            MetalDevice()(Philox4x32(0x81f)),
        )
        @test Array(device) == host
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
