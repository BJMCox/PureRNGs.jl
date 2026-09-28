using Adapt
using KernelAbstractions
using PureRNGs
using Random
using Test

include(joinpath(@__DIR__, "..", "..", "..", "fixtures.jl"))

# The KernelAbstractions CPU backend runs the same kernels a GPU backend launches,
# so the device fill plans can be checked on any machine. A device token routes
# the generator through the device launchers; the destination stays on the host.
_device_rng(rng) = IR._rebuild(rng, rng.position, IR._AMDGPU_BACKEND)

@testset "device fill plans write every element of an offset-axis destination" begin
    n = 1001
    for F in (Philox4x32, Threefry2x32, ChaCha, Philox4x64),
        T in (Float32, Float64, UInt32, Bool, UInt8, Int16, Float16),
        plan in (nothing, (Val(:grouped), Val(4)))

        rng = F(7, 3)
        expected, _ = rand_next(rng, T, n)
        destination = ZeroBasedVector(zeros(T, n))
        IR._launch_device_fill!(
            KernelAbstractions.CPU(),
            _device_rng(rng),
            destination,
            T,
            Val(:uniform),
            plan,
        )
        @test destination.data == expected
    end
end

# Device-token generators allocate their scratch through the backend extension,
# which is not loaded here; host arrays stand in for device memory.
IR._allocate_array(::IR._AMDGPUBackend, ::Type{T}, dims::Tuple) where {T} =
    Array{T}(undef, dims)

@testset "device weighted scan matches the CPU fold and samples" begin
    rng = Philox4x32(5, 17)
    device_rng = _device_rng(rng)
    # One weight vector fits in one fold pass; the other needs two (1024 lanes).
    for count in (3, 1025)
        weights = [mod(7index, 11) + 0.5 for index = 1:count]
        weights[2] = 0.0
        population = collect(1:count)
        expected, _ = randsample_next(rng, population, weights, 777)
        _, cpu_total, cpu_cumulative = IR._prepare_weight_scan(rng, weights, false)
        for agnostic in (true, false)
            _, total, cumulative = IR._prepare_weight_scan(device_rng, weights, agnostic)
            @test cumulative == cpu_cumulative
            @test only(total) == cpu_total
            destination = Vector{Int}(undef, 777)
            IR._fill_weighted_samples!(
                KernelAbstractions.CPU(),
                device_rng,
                population,
                nothing,
                total,
                cumulative,
                destination,
            )
            @test destination == expected
        end
    end
    @test_throws ArgumentError IR._prepare_weight_scan(device_rng, [1.0, -1.0], false)
end

@testset "device weighted samples without replacement equal the CPU samples" begin
    rng = Philox4x32(5, 17)
    population = collect(1:1025)
    weights = [mod(7index, 11) + 0.5 for index = 1:1025]
    weights[2] = 0.0
    expected, expected_next =
        randsample_next(rng, population, weights, 300; replace = false)
    for agnostic in (true, false)
        sample, next_rng = IR._weighted_unique_sample(
            _device_rng(rng),
            population,
            weights,
            agnostic,
            300,
            false,
        )
        @test sample == expected
        @test next_rng.position == expected_next.position
    end
    # A device sort need not order equal keys by index, so a kernel orders each
    # tied run that reaches the leading keys; a permutation shuffles its runs.
    keys = first(rand_next(Philox4x32(0x5a1), UInt64(1):UInt64(300), 5000))
    reversed_ties = sortperm(collect(zip(keys, length(keys):-1:1)))
    device_rng = _device_rng(Philox4x32(0x5a1))
    for count in (1, 1000, 5000)
        @test IR._race_order(device_rng, view(keys, :), count) == sortperm(keys)[1:count]
        order = copy(reversed_ties)
        IR._order_device_key_runs!(
            KernelAbstractions.CPU(),
            order,
            view(keys, :),
            count,
            nothing,
        )
        @test order[1:count] == sortperm(keys)[1:count]
    end
    host = sortperm(keys)
    IR._resolve_key_ties!(host, keys, Philox4x32(0x5a1))
    order = copy(reversed_ties)
    IR._resolve_key_ties!(order, view(keys, :), device_rng)
    @test order == host
end

# A device column fill runs one workitem per component when the columns are
# fewer than the resident workitems, and one per column otherwise. The CPU
# backend reports no limit, so a stand-in limit reaches both branches.
IR._device_workitems(::KernelAbstractions.CPU) = 4

@testset "device Dirichlet column fills equal the host column fill" begin
    rng = Philox4x32(11, 5)
    codec = IR._DirichletCodec([0.3, 2.0, 5.0])
    for columns in (3, 4, 9), threaded in (false, true)
        host = zeros(3, columns)
        IR._fill_columns!(IR._CPU_BACKEND, rng, host, false, codec)
        device = zeros(3, columns)
        IR._fill_columns!(KernelAbstractions.CPU(), rng, device, threaded, codec)
        @test device == host
        @test all(≈(1), sum(device; dims = 1))
    end
end

# A kernel argument converts only at its top level, so a codec that holds a
# device array hands that field to the adaptor itself.
@testset "codecs adapt their array fields" begin
    dirichlet = Adapt.adapt(Array{Float32}, IR._DirichletCodec([0.3, 2.0]))
    @test dirichlet.alpha isa Vector{Float32}
    @test dirichlet.alpha == Float32[0.3, 2.0]
    population =
        Adapt.adapt(Array{Float32}, IR._PopulationCodec([1.0, 2.0, 3.0], UInt64(3)))
    @test population.population isa Vector{Float32}
    @test population.population == Float32[1, 2, 3]
    @test population.cardinality == UInt64(3)
end
