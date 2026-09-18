using AMDGPU
using Distributions
using Enzyme
using KernelAbstractions
using PureRNGs
using MLDataDevices
using Random
using Test

include(joinpath(@__DIR__, "..", "..", "..", "fixtures.jl"))

function _fixed_distributions(::Type{T}) where {T}
    return (
        Normal(T(1.25), T(0.75)),
        Uniform(T(-1.5), T(2.25)),
        Exponential(T(1.75)),
        Bernoulli(T(0.375)),
        DiscreteUniform(-7, 13),
    )
end

_result_type(::Normal{T}) where {T} = T
_result_type(::Uniform{T}) where {T} = T
_result_type(::Exponential{T}) where {T} = T
_result_type(::Bernoulli) = Bool
_result_type(::DiscreteUniform) = Int

@inline function _exponential_raw(rng, ::Type{Float32})
    return IR._extract_bits_unchecked(
        rng,
        IR._position_block(rng.position),
        rng.position.bit,
        Val(24),
    )
end

@inline function _exponential_raw(rng, ::Type{Float64})
    return IR._extract_bits_unchecked(
        rng,
        IR._position_block(rng.position),
        rng.position.bit,
        Val(53),
    )
end

@inline _position_words(position::IR._Position64) =
    (position.block, UInt64(0), UInt64(position.bit))
@inline _position_words(position::IR._Position128) =
    (position.lo, position.hi, UInt64(position.bit))

@inline function _store_position!(destination, offset, position)
    words = _position_words(position)
    @inbounds for index in eachindex(words)
        destination[offset+index] = words[index]
    end
    return nothing
end

function _signed_exponential_kernel!(signed32, signed64, exp32, exp64, raw, states, rng)
    if AMDGPU.workitemIdx().x == 1
        continued_signed32, next_signed32 = rand_next(rng, Int32)
        continued_signed64, next_signed64 = rand_next(rng, Int64)
        continued_exp32, next_exp32 = randexp_next(rng, Float32)
        continued_exp64, next_exp64 = randexp_next(rng, Float64)
        @inbounds begin
            signed32[1] = rand(rng, Int32)
            signed32[2] = continued_signed32
            signed32[3] = rand(next_signed32, Int32)
            signed32[4] = rand_at(rng, Int32, 1)
            signed64[1] = rand(rng, Int64)
            signed64[2] = continued_signed64
            signed64[3] = rand(next_signed64, Int64)
            signed64[4] = rand_at(rng, Int64, 1)
            exp32[1] = randexp(rng, Float32)
            exp32[2] = continued_exp32
            exp32[3] = randexp_at(rng, Float32, 1)
            exp32[4] = randexp(next_exp32, Float32)
            exp32[5] = randexp_at(rng, Float32, 2)
            exp64[1] = randexp(rng, Float64)
            exp64[2] = continued_exp64
            exp64[3] = randexp_at(rng, Float64, 1)
            exp64[4] = randexp(next_exp64, Float64)
            exp64[5] = randexp_at(rng, Float64, 2)
            raw[1] = _exponential_raw(rng, Float32)
            raw[2] = _exponential_raw(rng, Float64)
        end
        _store_position!(states, 0, next_signed32.position)
        _store_position!(states, 3, next_signed64.position)
        _store_position!(states, 6, next_exp32.position)
        _store_position!(states, 9, next_exp64.position)
    end
    return nothing
end

function _distribution_kernel!(values, state, rng, distribution)
    if AMDGPU.workitemIdx().x == 1
        continued, next_rng = rand_next(rng, distribution)
        @inbounds begin
            values[1] = rand(rng, distribution)
            values[2] = continued
            values[3] = rand_at(rng, distribution, 1)
            values[4] = rand(next_rng, distribution)
            values[5] = rand_at(rng, distribution, 2)
        end
        _store_position!(state, 0, next_rng.position)
    end
    return nothing
end

function _pure_fill_result!(fill_function, rng, destination)
    return fill_function(rng, destination; threaded = true)
end

function _pure_fill_objective!(fill_function, rng, destination, scale)
    result = fill_function(rng, destination; threaded = true)
    values = result isa Tuple ? result[2] : result
    return scale * sum(values)
end

function _device_id(array)
    device = MLDataDevices.get_device(array)
    @assert device isa AMDGPUDevice{<:AMDGPU.HIPDevice}
    return AMDGPU.device_id(device.device)
end

function _software_identity()
    root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    source_status = readchomp(`git -C $root status --short`)
    hip = getfield(AMDGPU, :HIP)
    return (
        source = readchomp(`git -C $root rev-parse HEAD`),
        source_clean = isempty(source_status),
        source_status,
        julia = string(VERSION),
        machine = Sys.MACHINE,
        kernel = Sys.KERNEL,
        amdgpu = string(Base.pkgversion(AMDGPU)),
        kernelabstractions = string(Base.pkgversion(KernelAbstractions)),
        mldatadevices = string(Base.pkgversion(MLDataDevices)),
        distributions = string(Base.pkgversion(Distributions)),
        enzyme = string(Base.pkgversion(Enzyme)),
        gpucompiler = string(Base.pkgversion(getfield(AMDGPU, :GPUCompiler))),
        llvm = string(Base.pkgversion(getfield(AMDGPU, :LLVM))),
        libllvm = string(Base.libllvm_version),
        hip = string(hip.runtime_version()),
        device_libraries = getfield(AMDGPU, :libdevice_libs),
        device = sprint(show, AMDGPU.device()),
        architecture = hip.gcn_arch(AMDGPU.device()),
        backend = sprint(AMDGPU.versioninfo),
    )
end

function _enzyme_cases()
    return (
        (Random.rand!, rand_next!, rand_next),
        (Random.randn!, randn_next!, randn_next),
        (Random.randexp!, randexp_next!, randexp_next),
    )
end

function _check_distribution_preview(F, distribution, active_device)
    cpu_rng = F(0x817)
    rng = AMDGPUDevice()(cpu_rng)
    T = _result_type(distribution)
    kernel_values = AMDGPU.ROCArray{T}(undef, 5)
    kernel_state = AMDGPU.ROCArray{UInt64}(undef, 3)
    AMDGPU.@sync AMDGPU.@roc groupsize = 1 gridsize = 1 _distribution_kernel!(
        kernel_values,
        kernel_state,
        rng,
        distribution,
    )

    values = rand(rng, distribution, 5)
    continued, next_rng = rand_next(rng, distribution, 5)
    destination = similar(continued)
    returned, fill_next = rand_next!(rng, distribution, destination)
    _, scalar_next = rand_next(rng, distribution)
    host_values = Array(values)
    @test values isa AMDGPU.ROCArray{T,1}
    @test continued isa AMDGPU.ROCArray{T,1}
    @test _device_id(values) == _device_id(continued) == active_device
    @test isequal(host_values, Array(continued))
    @test Array(kernel_values) ==
          [host_values[1], host_values[1], host_values[1], host_values[2], host_values[2]]
    @test Tuple(Array(kernel_state)) == _position_words(scalar_next.position)
    @test returned === destination
    @test isequal(Array(destination), host_values)
    @test next_rng.position == fill_next.position

    if distribution isa Union{Uniform,Bernoulli,DiscreteUniform}
        expected, expected_next = rand_next(cpu_rng, distribution, 5)
        @test host_values == expected
        @test next_rng.position == expected_next.position
    end
    return nothing
end

@testset "R37 AMDGPU public host surface" begin
    for F in GENERATOR_TYPES
        cpu_rng = F(0x814)
        rng = AMDGPUDevice(:discarded)(cpu_rng)
        @test rng.device === IR._AMDGPU_BACKEND
        @test rand(rng, UInt64) === rand(cpu_rng, UInt64)
        @test randn(rng, Float64) === randn(cpu_rng, Float64)
        @test isapprox(randexp(rng, Float64), randexp(cpu_rng, Float64); rtol = 16eps())
        @test rand(rng, UInt16(1):UInt16(3)) === rand(cpu_rng, UInt16(1):UInt16(3))
    end
    rng = AMDGPUDevice()(Philox4x32(0x814))
    @test applicable(IR.randsample, rng, UInt32(1):UInt32(3), 2)
    @test applicable(IR.randsample_next, rng, UInt32(1):UInt32(3), 2)
    @test isempty(Test.detect_ambiguities(IR, Random; recursive = true))
end

# An extension is not a submodule of its parent, so a recursive scan that starts
# at PureRNGs never reaches it. Scan each loaded extension itself.
@testset "R1 extension ambiguities" begin
    for name in
        (:PureRNGsAMDGPUExt, :PureRNGsDistributionsExt, :PureRNGsKernelAbstractionsExt)
        extension = Base.get_extension(IR, name)
        @testset "$name" begin
            @test isempty(Test.detect_ambiguities(extension; recursive = true))
        end
    end
end

if AMDGPU.functional()
    AMDGPU.allowscalar(false)
    active_device = AMDGPU.device_id()
    @info "AMDGPU preview software identity" identity = _software_identity()

    @testset "R39 AMDGPU allocation smoke" begin
        for F in GENERATOR_TYPES,
            T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)

            cpu_rng = F(0x815)
            rng = AMDGPUDevice()(cpu_rng)
            values, next_rng = IR.rand_next(rng, T, 17)
            expected, expected_next = IR.rand_next(cpu_rng, T, 17)
            @test values isa AMDGPU.ROCArray{T,1}
            @test Array(values) == expected
            @test next_rng.position == expected_next.position
        end

        cpu_rng = Philox4x32(0x8151)
        rng = AMDGPUDevice()(cpu_rng)
        population = Int32[11, 12, 13, 14]
        weights = Float64[1, 0, 4, 2]
        values, next_rng =
            randsample_next(rng, AMDGPU.ROCArray(population), AMDGPU.ROCArray(weights), 9)
        expected, expected_next = randsample_next(cpu_rng, population, weights, 9)
        @test values isa AMDGPU.ROCArray{Int32,1}
        @test Array(values) == expected
        @test next_rng.position == expected_next.position
    end

    @testset "R25, R30, R43, and R63 AMDGPU signed and exponential probes" begin
        for F in GENERATOR_TYPES
            cpu_rng = F(0x816)
            rng = AMDGPUDevice()(cpu_rng)
            signed32 = AMDGPU.ROCArray{Int32}(undef, 4)
            signed64 = AMDGPU.ROCArray{Int64}(undef, 4)
            exp32 = AMDGPU.ROCArray{Float32}(undef, 5)
            exp64 = AMDGPU.ROCArray{Float64}(undef, 5)
            raw = AMDGPU.ROCArray{UInt64}(undef, 2)
            states = AMDGPU.ROCArray{UInt64}(undef, 12)

            AMDGPU.@sync AMDGPU.@roc groupsize = 1 gridsize = 1 _signed_exponential_kernel!(
                signed32,
                signed64,
                exp32,
                exp64,
                raw,
                states,
                rng,
            )

            for (T, values) in ((Int32, signed32), (Int64, signed64))
                U = unsigned(T)
                continued_unsigned, next_unsigned = rand_next(cpu_rng, U)
                expected = T[
                    reinterpret(T, rand(cpu_rng, U)),
                    reinterpret(T, continued_unsigned),
                    reinterpret(T, rand(next_unsigned, U)),
                    reinterpret(T, rand_at(cpu_rng, U, 1)),
                ]
                @test Array(values) == expected
                allocated, next_rng = rand_next(rng, T, 17)
                expected_allocated, expected_next = rand_next(cpu_rng, T, 17)
                @test allocated isa AMDGPU.ROCArray{T,1}
                @test _device_id(allocated) == active_device
                @test Array(allocated) == expected_allocated
                @test next_rng.position == expected_next.position
            end

            host_states = Array(states)
            for (offset, draw, T) in (
                (0, rand_next, Int32),
                (3, rand_next, Int64),
                (6, randexp_next, Float32),
                (9, randexp_next, Float64),
            )
                _, next_rng = draw(rng, T)
                @test Tuple(host_states[(offset+1):(offset+3)]) ==
                      _position_words(next_rng.position)
            end

            @test Array(raw) == UInt64[
                _exponential_raw(cpu_rng, Float32),
                _exponential_raw(cpu_rng, Float64),
            ]
            for (T, kernel_values) in ((Float32, exp32), (Float64, exp64))
                values = randexp(rng, T, 2)
                continued, next_rng = randexp_next(rng, T, 17)
                repeated, repeated_next = randexp_next(rng, T, 17)
                _, expected_next = randexp_next(cpu_rng, T, 17)
                destination = similar(continued)
                returned, fill_next = randexp_next!(rng, destination)
                first_two = Array(values)
                @test Array(kernel_values) ==
                      [first_two[1], first_two[1], first_two[1], first_two[2], first_two[2]]
                @test values isa AMDGPU.ROCArray{T,1}
                @test continued isa AMDGPU.ROCArray{T,1}
                @test _device_id(values) == _device_id(continued) == active_device
                @test isequal(Array(continued), Array(repeated))
                @test returned === destination
                @test isequal(Array(destination), Array(continued))
                @test next_rng.position ==
                      repeated_next.position ==
                      fill_next.position ==
                      expected_next.position
            end
        end
    end

    @testset "R43 and R64 AMDGPU fixed-distribution probes" begin
        for F in GENERATOR_TYPES, distribution in _fixed_distributions(Float32)
            _check_distribution_preview(F, distribution, active_device)
        end
        for distribution in _fixed_distributions(Float64)
            _check_distribution_preview(Philox4x32, distribution, active_device)
        end
    end

    @testset "R65 AMDGPU immutable Enzyme preview" begin
        for T in (Float32, Float64),
            (fill_function, next_fill_function, next_draw) in _enzyme_cases()

            rng = AMDGPUDevice()(Philox4x32(0x818))
            expected_values, expected_rng = next_draw(rng, T, 17)
            expected_host = Array(expected_values)

            for function_under_test in (fill_function, next_fill_function)
                destination = AMDGPU.zeros(T, 17)
                shadow = AMDGPU.fill(T(4), 17)
                shadow_result, primal_result = autodiff(
                    ForwardWithPrimal,
                    _pure_fill_result!,
                    Duplicated,
                    Const(function_under_test),
                    Const(rng),
                    Duplicated(destination, shadow),
                )
                primal_destination =
                    primal_result isa Tuple ? primal_result[2] : primal_result
                shadow_destination =
                    shadow_result isa Tuple ? shadow_result[2] : shadow_result
                @test primal_destination === destination
                @test shadow_destination === shadow
                @test Array(destination) == expected_host
                @test iszero(Array(shadow))
                @test _device_id(destination) == _device_id(shadow) == active_device
                if primal_result isa Tuple
                    @test primal_result[1] === expected_rng
                end
            end

            reverse_values = AMDGPU.zeros(T, 17)
            reverse_shadow = AMDGPU.fill(T(5), 17)
            scale = T(1.25)
            reverse_derivative = only(
                autodiff(
                    Reverse,
                    _pure_fill_objective!,
                    Active,
                    Const(next_fill_function),
                    Const(rng),
                    Duplicated(reverse_values, reverse_shadow),
                    Active(scale),
                ),
            )
            @test Array(reverse_values) == expected_host
            @test iszero(Array(reverse_shadow))
            @test reverse_derivative[4] ≈ sum(expected_host)

            batched_values = AMDGPU.zeros(T, 17)
            shadow_one = AMDGPU.fill(T(6), 17)
            shadow_two = AMDGPU.fill(T(7), 17)
            batched_derivative = only(
                autodiff(
                    Forward,
                    _pure_fill_objective!,
                    Const(next_fill_function),
                    Const(rng),
                    BatchDuplicated(batched_values, (shadow_one, shadow_two)),
                    BatchDuplicated(T(2), (one(T), T(2))),
                ),
            )
            @test Array(batched_values) == expected_host
            @test iszero(Array(shadow_one))
            @test iszero(Array(shadow_two))
            @test _device_id(shadow_one) == _device_id(shadow_two) == active_device
            @test batched_derivative[1] ≈ sum(expected_host)
            @test batched_derivative[2] ≈ T(2) * sum(expected_host)
        end
    end

    @testset "R56-R58 AMDGPU unweighted sampling smoke" begin
        for F in GENERATOR_TYPES
            cpu_rng = F(0x91b)
            rng = AMDGPUDevice()(cpu_rng)
            host_population = collect(Int32(-5):Int32(17))
            population = AMDGPU.ROCArray(host_population)
            values, next_rng = IR.randsample_next(rng, population, 17)
            expected, expected_next = IR.randsample_next(cpu_rng, host_population, 17)
            @test values isa AMDGPU.ROCArray{Int32,1}
            @test Array(values) == expected
            @test next_rng.position == expected_next.position
        end
    end
else
    @info "AMDGPU preview hardware unavailable; device execution remains pending"
end
