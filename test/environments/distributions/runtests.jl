using Distributions
using PureRNGs
using Random
using Test

include(joinpath(@__DIR__, "..", "..", "fixtures.jl"))

const EXT = Base.get_extension(PureRNGs, :PureRNGsDistributionsExt)

function fixed_distributions(::Type{T}) where {T}
    return (
        Normal(T(1.25), T(0.75)),
        Uniform(T(-1.5), T(2.25)),
        Exponential(T(1.75)),
        Bernoulli(T(0.375)),
        DiscreteUniform(-7, 13),
    )
end

fixed_result_type(::Normal{T}) where {T} = T
fixed_result_type(::Uniform{T}) where {T} = T
fixed_result_type(::Exponential{T}) where {T} = T
fixed_result_type(::Bernoulli) = Bool
fixed_result_type(::DiscreteUniform) = Int

function fixed_distribution_array_cases()
    representative = Normal{Float64}(1.25, 0.75)
    return (
        (
            (Philox4x32, distribution) for T in (Float32, Float64) for
            distribution in fixed_distributions(T)
        )...,
        ((F, representative) for F in GENERATOR_TYPES if F !== Philox4x32)...,
    )
end

function primitive_next(rng, d::Normal{T}) where {T}
    z, next_rng = randn_next(rng, T)
    return fma(d.σ, z, d.μ), next_rng
end

function primitive_next(rng, d::Uniform{T}) where {T}
    u, next_rng = rand_next(rng, T)
    width = d.b - d.a
    scaled = width * u
    return d.a + scaled, next_rng
end

function primitive_next(rng, d::Exponential{T}) where {T}
    x, next_rng = randexp_next(rng, T)
    return d.θ * x, next_rng
end

function primitive_next(rng, d::Bernoulli{T}) where {T}
    u, next_rng = rand_next(rng, T)
    return u < d.p, next_rng
end

primitive_next(rng, d::DiscreteUniform) = rand_next(rng, d.a:d.b)

function primitive_at(rng, d::Normal{T}, index) where {T}
    return fma(d.σ, randnat(rng, T, index), d.μ)
end

function primitive_at(rng, d::Uniform{T}, index) where {T}
    width = d.b - d.a
    scaled = width * randat(rng, T, index)
    return d.a + scaled
end

primitive_at(rng, d::Exponential{T}, index) where {T} = d.θ * randexpat(rng, T, index)

primitive_at(rng, d::Bernoulli{T}, index) where {T} = randat(rng, T, index) < d.p

function primitive_at(rng, d::DiscreteUniform, index)
    range = d.a:d.b
    span = IR._range_span(range)
    width = IR._range_bits(span)
    addressed = IR._addressed_rng(rng, width, index)
    return IR._draw_range_unchecked(addressed, range, span)
end

function primitive_chain(rng, distribution, count)
    values = Vector{fixed_result_type(distribution)}(undef, count)
    cursor = rng
    for index in eachindex(values)
        values[index], cursor = primitive_next(cursor, distribution)
    end
    return cursor, values
end

invalid_error(f) = @test_throws ArgumentError f()

function distribution_allocations(rng, distribution, destination)
    rand(rng, distribution)
    rand_next(rng, distribution)
    randat(rng, distribution, 2)
    rand_next!(rng, distribution, destination; threaded = false)
    return (
        @allocated(rand(rng, distribution)),
        @allocated(rand_next(rng, distribution)),
        @allocated(randat(rng, distribution, 2)),
        @allocated(rand_next!(rng, distribution, destination; threaded = false)),
    )
end

@testset "R64 fixed distribution scalar forms" begin
    for F in GENERATOR_TYPES,
        T in (Float32, Float64),
        distribution in fixed_distributions(T)

        rng = F(0x901)
        expected, expected_next = primitive_next(rng, distribution)

        @test rand(rng, distribution) === expected
        actual, actual_next = rand_next(rng, distribution)
        @test actual === expected
        @test actual_next === expected_next
        @test randat(rng, distribution, 4) === primitive_at(rng, distribution, 4)
        @test rand(rng, distribution) === rand(rng, distribution)
    end
end

@testset "R64 fixed distribution arrays and fills" begin
    for (F, distribution) in fixed_distribution_array_cases()
        rng = F(0x902)
        expected_next, expected = primitive_chain(rng, distribution, 12)
        result_type = eltype(expected)

        allocated = rand(rng, distribution, 3, 4)
        @test size(allocated) == (3, 4)
        @test eltype(allocated) === result_type
        @test vec(allocated) == expected

        continued, allocated_next = rand_next(rng, distribution, 3, 4)
        @test continued == allocated
        @test allocated_next === expected_next

        serial = Vector{result_type}(undef, 12)
        threaded = similar(serial)
        @test rand!(rng, distribution, serial; threaded = false) === serial
        @test serial == expected
        @test rand!(rng, distribution, threaded; threaded = true) === threaded
        @test threaded == expected

        returned, filled_next = rand_next!(rng, distribution, serial; threaded = false)
        @test returned === serial
        @test serial == expected
        @test filled_next === expected_next

        storage = fill(zero(result_type), 24)
        destination = @view storage[2:2:24]
        view_result, view_next =
            rand_next!(rng, distribution, destination; threaded = false)
        @test view_result === destination
        @test collect(destination) == expected
        @test view_next === expected_next
    end
end

@testset "R40 and R64 mapped fill traversal" begin
    rng = Philox4x32(0x9a2)

    distribution = Bernoulli{Float64}(0.375)
    expected_next, expected = primitive_chain(rng, distribution, 131)
    for threaded in (false, true)
        destination = BitArray(undef, length(expected))
        returned, actual_next =
            rand_next!(rng, distribution, destination; threaded = threaded)
        @test returned === destination
        @test destination == expected
        @test actual_next === expected_next
    end

    caller = current_task()
    normal = Normal{Float64}(0.25, 1.5)
    probe = TaskWriteProbe(Vector{Float64}(undef, 37))
    returned, actual_next = rand_next!(rng, normal, probe; threaded = false)
    expected_next, expected = primitive_chain(rng, normal, length(probe))
    @test returned === probe
    @test probe.data == expected
    @test actual_next === expected_next
    @test all(task -> task === caller, probe.writers)
end

@testset "R54 and R64 atomic preflight and zero sizes" begin
    for distribution in fixed_distributions(Float64)
        rng = Philox4x32(0x903)
        width = EXT._distribution_span(distribution)
        bit = IR._block_bits(rng) - width
        last = IR._rebuild(rng, IR._Position64(IR._max_block(rng), bit), rng.device)
        destination = fill(rand(last, distribution), 2)
        original = copy(destination)
        @test_throws ArgumentError rand!(last, distribution, destination)
        @test destination == original
        @test last.position.bit == bit

        exhausted = IR._reserve(last, UInt64(width), UInt64(0))
        empty = similar(destination, 0)
        @test rand!(exhausted, distribution, empty) === empty
        returned, empty_next = rand_next!(exhausted, distribution, empty)
        @test returned === empty
        @test empty_next === exhausted
        @test isempty(rand(exhausted, distribution, 0))
        allocated, allocated_next = rand_next(exhausted, distribution, 0)
        @test isempty(allocated)
        @test allocated_next === exhausted
    end
end

@testset "R64 validation and closed dispatch" begin
    rng = Philox4x32(0x904)
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    invalid = (
        Normal(Inf, 1.0; check_args = false),
        Normal(0.0, Inf; check_args = false),
        Normal(0.0, -1.0; check_args = false),
        Uniform(-Inf, 1.0; check_args = false),
        Uniform(0.0, Inf; check_args = false),
        Uniform(1.0, 1.0; check_args = false),
        Uniform(-floatmax(Float64), floatmax(Float64); check_args = false),
        Exponential(0.0; check_args = false),
        Exponential(Inf; check_args = false),
        Bernoulli(-0.1; check_args = false),
        Bernoulli(1.1; check_args = false),
        Bernoulli(NaN; check_args = false),
        DiscreteUniform(2, 1; check_args = false),
    )

    for distribution in invalid
        invalid_error(() -> rand(exhausted, distribution))
    end

    distribution = first(invalid)
    result_type = fixed_result_type(distribution)
    destination = Vector{result_type}(undef, 0)
    invalid_error(() -> rand_next(exhausted, distribution))
    invalid_error(() -> randat(exhausted, distribution, 0))
    invalid_error(() -> rand(exhausted, distribution, -1))
    invalid_error(() -> rand_next(exhausted, distribution, -1))
    invalid_error(() -> rand!(exhausted, distribution, destination))
    invalid_error(() -> rand_next!(exhausted, distribution, destination))

    @test_throws MethodError rand(rng, Normal(Float16(0), Float16(1)))
    @test_throws MethodError rand(rng, Beta())
    @test_throws MethodError rand!(rng, Normal(), Float32[])

    distribution = Normal()
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    @test_throws ArgumentError rand(exhausted, distribution)
    @test_throws ArgumentError rand_next(exhausted, distribution)
    @test_throws ArgumentError randat(rng, distribution, 0)
    width = EXT._distribution_span(distribution)
    last = IR._rebuild(
        rng,
        IR._Position64(IR._max_block(rng), IR._block_bits(rng) - width),
        rng.device,
    )
    @test_throws ArgumentError randat(last, distribution, 2)
end

@testset "R1 and R64 allocations and ambiguity freedom" begin
    rng = Philox4x32(0x905)
    normal = Normal{Float32}(0.0f0, 1.0f0)
    destination = Vector{Float32}(undef, 17)
    @test distribution_allocations(rng, normal, destination) == (0, 0, 0, 0)

    bits = BitArray(undef, 17)
    bernoulli = Bernoulli{Float32}(0.5f0)
    rand_next!(rng, bernoulli, bits; threaded = false)
    @test @allocated(rand_next!(rng, bernoulli, bits; threaded = false)) == 0

    ambiguities = Test.detect_ambiguities(PureRNGs, Random, Distributions; recursive = true)
    extension_ambiguities = filter(ambiguities) do pair
        any(method -> method.module === EXT, pair)
    end
    @test isempty(extension_ambiguities)
end

@testset "R34 Distributions StatefulRNG smoke" begin
    normal_root = Philox4x32(0x812)
    normal = Normal()
    scalar_expected, scalar_next = randn_next(normal_root, Float64)
    mutable_rng = StatefulRNG(normal_root)

    @test rand(mutable_rng, normal) === scalar_expected
    batch_expected, batch_next = randn_next(scalar_next, Float64, 11)
    @test rand(mutable_rng, normal, 11) == batch_expected
    @test parent(mutable_rng) === batch_next

    exponential_root = Philox4x32(0x813)
    exponential = Exponential()
    scalar_expected, scalar_next = randexp_next(exponential_root, Float64)
    mutable_rng = StatefulRNG(exponential_root)

    @test rand(mutable_rng, exponential) === scalar_expected
    batch_expected, batch_next = randexp_next(scalar_next, Float64, 11)
    @test rand(mutable_rng, exponential, 11) == batch_expected
    @test parent(mutable_rng) === batch_next
end

include("distribution_expansion.jl")
include("distribution_transforms.jl")
include("categorical.jl")
include("accuracy_gates.jl")
