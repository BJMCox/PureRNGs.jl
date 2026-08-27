using PureRNGs
using Metal
using MLDataDevices
using Random
using Test

const IR = PureRNGs
const METAL_32_FAMILIES = (Philox2x32, Philox4x32, Threefry2x32, Threefry4x32)
const METAL_64_FAMILIES = (Philox2x64, Philox4x64, Threefry2x64, Threefry4x64)
const METAL_FAMILIES = (METAL_32_FAMILIES..., METAL_64_FAMILIES...)

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
        end
        for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
            range = T(1):T(3)
            @test rand(rng, range) === rand(cpu_rng, range)
            @test last(IR.rand_next(rng, range)) === last(IR.rand_next(cpu_rng, range))
            @test which(rand, (typeof(rng), typeof(range), Int)).module === IR
            @test which(IR.rand_next, (typeof(rng), typeof(range), Int)).module === IR
        end
    end

    @test isempty(Test.detect_ambiguities(IR, Random; recursive = true))
end

@testset "R41 Metal allocating exclusions" begin
    for F in METAL_32_FAMILIES
        rng = MetalDevice()(F(0x817))
        for count in (0, 1)
            _check_metal_error(() -> rand(rng, Float64, count))
            _check_metal_error(() -> IR.rand_next(rng, Float64, count))
            _check_metal_error(() -> randn(rng, Float64, count))
            _check_metal_error(() -> IR.randn_next(rng, Float64, count))
        end
        _check_metal_error(() -> IR.rand_next(rng, 0))
        _check_metal_error(() -> IR.randn_next(rng, 0))
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
    end

    for F in METAL_64_FAMILIES
        rng = MetalDevice()(F(0x81c))
        uniform = MetalDeviceArrayProbe(UInt32[])
        normal = MetalDeviceArrayProbe(Float32[])
        _check_metal_error(() -> rand!(rng, uniform))
        _check_metal_error(() -> IR.rand_next!(rng, uniform))
        _check_metal_error(() -> randn!(rng, normal))
        _check_metal_error(() -> IR.randn_next!(rng, normal))
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
else
    @info "Metal hardware unavailable; served device execution was not run"
end
