using Distributions
using PureRNGs
using Random
using Test

const IR = PureRNGs
const EXT = Base.get_extension(PureRNGs, :PureRNGsDistributionsExt)
const FAMILY_TYPES = (
    Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
)

mutable struct TaskWriteProbe{T,A<:AbstractVector{T}} <: AbstractVector{T}
    data::A
    writers::Vector{Task}
end

TaskWriteProbe(data::AbstractVector{T}) where {T} =
    TaskWriteProbe(data, Vector{Task}(undef, length(data)))
Base.size(array::TaskWriteProbe) = size(array.data)
Base.getindex(array::TaskWriteProbe, index::Int) = array.data[index]
function Base.setindex!(array::TaskWriteProbe, value, index::Int)
    array.writers[index] = current_task()
    return setindex!(array.data, value, index)
end
IR.MLDataDevices.get_device(array::TaskWriteProbe) = IR.MLDataDevices.get_device(array.data)
IR.KernelAbstractions.get_backend(array::TaskWriteProbe) =
    IR.KernelAbstractions.get_backend(array.data)

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

function primitive_next(rng, d::Normal{T}) where {T}
    next_rng, z = randn_next(rng, T)
    return next_rng, fma(d.σ, z, d.μ)
end

function primitive_next(rng, d::Uniform{T}) where {T}
    next_rng, u = rand_next(rng, T)
    width = d.b - d.a
    scaled = width * u
    return next_rng, d.a + scaled
end

function primitive_next(rng, d::Exponential{T}) where {T}
    next_rng, x = randexp_next(rng, T)
    return next_rng, d.θ * x
end

function primitive_next(rng, d::Bernoulli{T}) where {T}
    next_rng, u = rand_next(rng, T)
    return next_rng, u < d.p
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
        cursor, values[index] = primitive_next(cursor, distribution)
    end
    return cursor, values
end

function invalid_error(f, name)
    error = try
        f()
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin(name, sprint(showerror, error))
end

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

@testset "R37 Distributions extension metadata" begin
    @test EXT !== nothing
    @test Base.pkgversion(Distributions) >= v"0.25.0"
end

@testset "R64 fixed distribution scalar forms" begin
    for F in FAMILY_TYPES, T in (Float32, Float64), distribution in fixed_distributions(T)
        rng = F(0x901)
        expected_next, expected = primitive_next(rng, distribution)

        @test rand(rng, distribution) === expected
        actual_next, actual = rand_next(rng, distribution)
        @test actual === expected
        @test actual_next === expected_next
        @test randat(rng, distribution, 4) === primitive_at(rng, distribution, 4)
        @test rand(rng, distribution) === rand(rng, distribution)
    end
end

@testset "R64 fixed distribution arrays and fills" begin
    for F in FAMILY_TYPES, T in (Float32, Float64), distribution in fixed_distributions(T)
        rng = F(0x902)
        expected_next, expected = primitive_chain(rng, distribution, 12)
        result_type = eltype(expected)

        allocated = rand(rng, distribution, 3, 4)
        @test size(allocated) == (3, 4)
        @test eltype(allocated) === result_type
        @test vec(allocated) == expected

        allocated_next, continued = rand_next(rng, distribution, 3, 4)
        @test continued == allocated
        @test allocated_next === expected_next

        serial = Vector{result_type}(undef, 12)
        threaded = similar(serial)
        @test rand!(rng, distribution, serial; threaded = false) === serial
        @test serial == expected
        @test rand!(rng, distribution, threaded; threaded = true) === threaded
        @test threaded == expected

        filled_next, returned = rand_next!(rng, distribution, serial; threaded = false)
        @test returned === serial
        @test serial == expected
        @test filled_next === expected_next

        storage = fill(zero(result_type), 24)
        destination = @view storage[2:2:24]
        view_next, view_result =
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
        actual_next, returned =
            rand_next!(rng, distribution, destination; threaded = threaded)
        @test returned === destination
        @test destination == expected
        @test actual_next === expected_next
    end

    caller = current_task()
    normal = Normal{Float64}(0.25, 1.5)
    probe = TaskWriteProbe(Vector{Float64}(undef, 37))
    actual_next, returned = rand_next!(rng, normal, probe; threaded = false)
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
        empty_next, returned = rand_next!(exhausted, distribution, empty)
        @test returned === empty
        @test empty_next === exhausted
        @test isempty(rand(exhausted, distribution, 0))
        allocated_next, allocated = rand_next(exhausted, distribution, 0)
        @test isempty(allocated)
        @test allocated_next === exhausted
    end
end

@testset "R64 validation and closed dispatch" begin
    rng = Philox4x32(0x904)
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    invalid = (
        (Normal(Inf, 1.0; check_args = false), "Normal"),
        (Normal(0.0, Inf; check_args = false), "Normal"),
        (Normal(0.0, -1.0; check_args = false), "Normal"),
        (Uniform(-Inf, 1.0; check_args = false), "Uniform"),
        (Uniform(0.0, Inf; check_args = false), "Uniform"),
        (Uniform(1.0, 1.0; check_args = false), "Uniform"),
        (Uniform(-floatmax(Float64), floatmax(Float64); check_args = false), "Uniform"),
        (Exponential(0.0; check_args = false), "Exponential"),
        (Exponential(Inf; check_args = false), "Exponential"),
        (Bernoulli(-0.1; check_args = false), "Bernoulli"),
        (Bernoulli(1.1; check_args = false), "Bernoulli"),
        (Bernoulli(NaN; check_args = false), "Bernoulli"),
        (DiscreteUniform(2, 1; check_args = false), "DiscreteUniform"),
    )

    for (distribution, name) in invalid
        invalid_error(() -> rand(exhausted, distribution), name)
        invalid_error(() -> rand_next(exhausted, distribution), name)
        invalid_error(() -> randat(exhausted, distribution, 0), name)
        invalid_error(() -> rand(exhausted, distribution, -1), name)
        result_type = fixed_result_type(distribution)
        invalid_error(
            () -> rand!(exhausted, distribution, Vector{result_type}(undef, 0)),
            name,
        )
    end

    @test_throws ArgumentError rand(rng, Normal(), -1)
    @test_throws TypeError rand!(rng, Normal(), Float64[]; threaded = 1)
    @test_throws MethodError rand(rng, Normal(Float16(0), Float16(1)))
    @test_throws MethodError rand(rng, Uniform(Float16(0), Float16(1)))
    @test_throws MethodError rand(rng, Exponential(Float16(1)))
    @test_throws MethodError rand(rng, Bernoulli(Float16(0.5)))
    @test_throws MethodError rand(rng, Beta())
    @test_throws MethodError rand!(rng, Normal(), Float32[])
    @test_throws MethodError rand!(rng, Bernoulli(), Float64[])
    @test_throws MethodError rand!(rng, DiscreteUniform(), Int32[])

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

@testset "R1 and R64 inference, allocations, and ambiguity freedom" begin
    rng = Philox4x32(0x905)
    normal = Normal{Float32}(0.0f0, 1.0f0)
    uniform = Uniform{Float64}(0.0, 1.0)
    bernoulli = Bernoulli{Float32}(0.5f0)
    destination = Vector{Float32}(undef, 17)
    @test @inferred(rand(rng, normal)) isa Float32
    @test @inferred(rand_next(rng, uniform)) isa Tuple{typeof(rng),Float64}
    @test @inferred(randat(rng, bernoulli, 2)) isa Bool
    @test @inferred(rand!(rng, normal, destination; threaded = false)) === destination
    @test @inferred(rand_next!(rng, normal, destination; threaded = false)) isa
          Tuple{typeof(rng),typeof(destination)}
    @test distribution_allocations(rng, normal, destination) == (0, 0, 0, 0)

    codec = EXT._DistributionCodec(normal, rng.device)
    @test isbits(codec)
    @test @inferred(IR._fill_width(codec, Float32)) === IR._normal_bits(Float32)
    @test @inferred(IR._fill_family(codec)) === IR.FAMILY_NORMAL
    @test EXT._scalar_store_plan((Val(:cooperative), Val(4), Val(32), Val(:packed)),) ===
          (Val(:cooperative), Val(4), Val(32))
    exponential_codec = EXT._DistributionCodec(Exponential{Float32}(1.0f0), rng.device)
    backend = IR.KernelAbstractions.get_backend(destination)
    @test IR._transformed_fill_plan(exponential_codec, backend, rng, Float32) === nothing

    bits = BitArray(undef, 17)
    bernoulli = Bernoulli{Float32}(0.5f0)
    @test @inferred(rand_next!(rng, bernoulli, bits; threaded = false)) isa
          Tuple{typeof(rng),typeof(bits)}
    rand_next!(rng, bernoulli, bits; threaded = false)
    @test @allocated(rand_next!(rng, bernoulli, bits; threaded = false)) == 0

    full = DiscreteUniform(typemin(Int), typemax(Int))
    @test EXT._discrete_span(full) === UInt64(0)
    @test EXT._distribution_span(full) === UInt16(128)
    @test @inferred(rand(rng, full)) isa Int

    ambiguities =
        Test.detect_ambiguities(PureRNGs, Random, Distributions; recursive = true)
    extension_ambiguities = filter(ambiguities) do pair
        any(method -> method.module === EXT, pair)
    end
    @test isempty(extension_ambiguities)
end

@testset "R34 Distributions StatefulRNG smoke" begin
    normal_root = Philox4x32(0x812)
    normal = Normal()
    scalar_next, scalar_expected = randn_next(normal_root, Float64)
    mutable_rng = StatefulRNG(normal_root)

    @test rand(mutable_rng, normal) === scalar_expected
    batch_next, batch_expected = randn_next(scalar_next, Float64, 11)
    @test rand(mutable_rng, normal, 11) == batch_expected
    @test mutable_rng.rng === batch_next

    exponential_root = Philox4x32(0x813)
    exponential = Exponential()
    scalar_next, scalar_expected = randexp_next(exponential_root, Float64)
    mutable_rng = StatefulRNG(exponential_root)

    @test rand(mutable_rng, exponential) === scalar_expected
    batch_next, batch_expected = randexp_next(scalar_next, Float64, 11)
    @test rand(mutable_rng, exponential, 11) == batch_expected
    @test mutable_rng.rng === batch_next
end
