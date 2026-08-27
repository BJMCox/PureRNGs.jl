using AMDGPU
using PureRNGs
using MLDataDevices
using Random
using Test

const IR = PureRNGs
const AMDGPU_FAMILIES = (
    Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
)

@testset "R37 AMDGPU extension surface" begin
    extension_module = Base.get_extension(IR, :PureRNGsAMDGPUExt)
    @test extension_module !== nothing

    for F in AMDGPU_FAMILIES
        cpu_rng = F(0x814)
        rng = AMDGPUDevice(:discarded)(cpu_rng)
        @test rng.device === IR._AMDGPU_BACKEND
        @test isbits(rng.device)
        @test sizeof(rng.device) == 0
        @test which(IR.rand_next, (typeof(rng), Int)).module === IR
        @test which(IR.randn_next, (typeof(rng), Int)).module === IR
        @test applicable(IR.randsample, rng, UInt32(1):UInt32(3), 2)
        @test applicable(IR.randsample_next, rng, UInt32(1):UInt32(3), 2)

        for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
            @test rand(rng, T) === rand(cpu_rng, T)
            @test which(rand, (typeof(rng), Type{T}, Int)).module === IR
            @test which(IR.rand_next, (typeof(rng), Type{T}, Int)).module === IR
        end
        for T in (Float32, Float64)
            @test randn(rng, T) === randn(cpu_rng, T)
            @test which(randn, (typeof(rng), Type{T}, Int)).module === IR
            @test which(IR.randn_next, (typeof(rng), Type{T}, Int)).module === IR
        end
        for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
            range = T(1):T(3)
            @test rand(rng, range) === rand(cpu_rng, range)
            @test which(rand, (typeof(rng), typeof(range), Int)).module === IR
            @test which(IR.rand_next, (typeof(rng), typeof(range), Int)).module === IR
        end
    end

    @test isempty(Test.detect_ambiguities(IR, Random; recursive = true))
end

if AMDGPU.functional()
    @testset "R39 AMDGPU allocation smoke" begin
        for F in AMDGPU_FAMILIES,
            T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)

            cpu_rng = F(0x815)
            rng = AMDGPUDevice()(cpu_rng)
            next_rng, values = IR.rand_next(rng, T, 17)
            expected_next, expected = IR.rand_next(cpu_rng, T, 17)
            @test values isa AMDGPU.ROCArray{T,1}
            @test Array(values) == expected
            @test next_rng.position == expected_next.position
        end

        cpu_rng = Philox4x32(0x8151)
        rng = AMDGPUDevice()(cpu_rng)
        population = Int32[11, 12, 13, 14]
        weights = Float64[1, 0, 4, 2]
        next_rng, values =
            randsample_next(rng, AMDGPU.ROCArray(population), AMDGPU.ROCArray(weights), 9)
        expected_next, expected = randsample_next(cpu_rng, population, weights, 9)
        @test values isa AMDGPU.ROCArray{Int32,1}
        @test Array(values) == expected
        @test next_rng.position == expected_next.position
    end

    @testset "R56-R58 AMDGPU unweighted sampling smoke" begin
        for F in AMDGPU_FAMILIES
            cpu_rng = F(0x91b)
            rng = AMDGPUDevice()(cpu_rng)
            host_population = collect(Int32(-5):Int32(17))
            population = AMDGPU.ROCArray(host_population)
            next_rng, values = IR.randsample_next(rng, population, 17)
            expected_next, expected = IR.randsample_next(cpu_rng, host_population, 17)
            @test values isa AMDGPU.ROCArray{Int32,1}
            @test Array(values) == expected
            @test next_rng.position == expected_next.position
        end
    end
else
    @info "AMDGPU hardware unavailable; device execution was not run"
end
