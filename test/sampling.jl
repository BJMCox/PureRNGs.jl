const SamplingIR = PureRNGs
const SamplingMLD = PureRNGs.MLDataDevices
const SamplingKA = PureRNGs.KernelAbstractions

const UNWEIGHTED_OFFSET_GOLDEN = (
    Philox2x32 => Int32[104, 105, 102, 101, 105, 101, 105, 105, 106, 103, 106, 102],
    Philox4x32 => Int32[106, 105, 104, 103, 102, 105, 104, 103, 102, 104, 103, 102],
    Philox2x64 => Int32[103, 102, 106, 104, 103, 104, 105, 106, 101, 106, 104, 101],
    Philox4x64 => Int32[106, 103, 105, 104, 101, 101, 106, 104, 101, 104, 102, 101],
    Threefry2x32 => Int32[103, 103, 101, 102, 104, 101, 106, 105, 104, 103, 106, 105],
    Threefry4x32 => Int32[101, 104, 101, 102, 103, 104, 105, 103, 103, 105, 104, 106],
    Threefry2x64 => Int32[102, 102, 103, 105, 104, 104, 102, 103, 102, 101, 105, 104],
    Threefry4x64 => Int32[103, 103, 106, 103, 103, 101, 104, 102, 102, 101, 103, 106],
)

struct DeviceAgnosticIterable{T}
    values::Vector{T}
    starts::Base.RefValue{Int}
end

Base.IteratorSize(::Type{<:DeviceAgnosticIterable}) = Base.HasLength()
Base.IteratorEltype(::Type{<:DeviceAgnosticIterable}) = Base.HasEltype()
Base.eltype(::Type{DeviceAgnosticIterable{T}}) where {T} = T
Base.length(iter::DeviceAgnosticIterable) = length(iter.values)
function Base.iterate(iter::DeviceAgnosticIterable)
    iter.starts[] += 1
    return iterate(iter.values)
end
Base.iterate(iter::DeviceAgnosticIterable, state) = iterate(iter.values, state)
SamplingMLD.get_device(::DeviceAgnosticIterable) = nothing

struct DeclaredHugeIterable end

Base.IteratorSize(::Type{DeclaredHugeIterable}) = Base.HasLength()
Base.IteratorEltype(::Type{DeclaredHugeIterable}) = Base.HasEltype()
Base.eltype(::Type{DeclaredHugeIterable}) = Int
Base.length(::DeclaredHugeIterable) = big(typemax(Int)) + 1
Base.iterate(::DeclaredHugeIterable, state...) = error("population must not materialize")
SamplingMLD.get_device(::DeclaredHugeIterable) = nothing

struct UnknownDeviceIterable{T}
    values::Vector{T}
end

Base.iterate(iter::UnknownDeviceIterable, state...) = iterate(iter.values, state...)

struct SamplingCUDAProbe{T} <: AbstractVector{T}
    values::Vector{T}
end

Base.size(population::SamplingCUDAProbe) = size(population.values)
Base.getindex(population::SamplingCUDAProbe, index::Int) = population.values[index]
SamplingMLD.get_device(::SamplingCUDAProbe) = SamplingMLD.CUDADevice(:named)
SamplingMLD.get_device_type(::SamplingCUDAProbe) = SamplingMLD.CUDADevice

struct SamplingSerialProbe{T} <: AbstractVector{T}
    values::Vector{T}
end

Base.size(destination::SamplingSerialProbe) = size(destination.values)
Base.getindex(destination::SamplingSerialProbe, index::Int) = destination.values[index]
Base.setindex!(destination::SamplingSerialProbe, value, index::Int) =
    setindex!(destination.values, value, index)
SamplingMLD.get_device(::SamplingSerialProbe) = SamplingMLD.CPUDevice()
SamplingMLD.get_device_type(::SamplingSerialProbe) = SamplingMLD.CPUDevice
SamplingKA.get_backend(::SamplingSerialProbe) =
    error("serial sampling must not get a backend")

struct CountedPopulation{T} <: AbstractVector{T}
    values::Vector{T}
    reads::Base.RefValue{Int}
end

Base.size(population::CountedPopulation) = size(population.values)
function Base.getindex(population::CountedPopulation, index::Int)
    population.reads[] += 1
    return population.values[index]
end
SamplingMLD.get_device(::CountedPopulation) = SamplingMLD.CPUDevice()

function _sample_at(population::AbstractArray, ordinal::Integer)
    indices = CartesianIndices(axes(population))
    index = indices[firstindex(indices)+Int(ordinal)-1]
    return population[index]
end
_sample_at(population::AbstractRange, ordinal::Integer) = population[ordinal]
_sample_at(population::AbstractRange{<:SamplingIR._RangeInteger}, ordinal::Integer) =
    SamplingIR._range_value(population, UInt64(ordinal) - 1)

function _chained_unweighted(rng, population, count::Integer)
    values = Vector{eltype(population)}(undef, Int(count))
    cursor = rng
    cardinality = length(population) % UInt64
    for index in eachindex(values)
        ordinal, cursor = rand_next(cursor, UInt64(1):cardinality)
        values[index] = _sample_at(population, ordinal)
    end
    return cursor, values
end

@testset "R13, R57, and R58 canonical offset-axis golden vectors" begin
    population = IdentityAxesMatrix(reshape(Int32.(101:106), 2, 3))
    for (F, expected) in UNWEIGHTED_OFFSET_GOLDEN
        rng = F(0x9760)
        values, _ = randsample_next(rng, population, 12)
        @test values == expected
        @test randsample(rng, population, 12) == expected
    end
end

@testset "R56-R58 unweighted sampling values and request forms" begin
    populations = (
        collect(Int16(11):Int16(17)),
        reshape(collect(Int16(21):Int16(32)), 3, 4),
        ZeroBasedVector(collect(Int16(41):Int16(49))),
        IdentityAxesMatrix(reshape(collect(Int16(51):Int16(56)), 2, 3)),
        Int16(-17):Int16(3):Int16(31),
        Int128(-9):Int128(2):Int128(9),
        big(-7):big(2):big(7),
    )

    for population in populations
        rng = Philox4x32(0x901)
        expected_next, expected = _chained_unweighted(rng, population, 19)
        values, next_rng = randsample_next(rng, population, 19)

        @test values == expected
        @test next_rng == expected_next
    end

    rng = Philox4x32(0x901)
    population = first(populations)
    _, expected = _chained_unweighted(rng, population, 19)
    @test randsample(rng, population, 7) == expected[1:7]
    no_k, no_k_next = randsample_next(rng, population)
    chained_next, chained = _chained_unweighted(rng, population, length(population))
    @test no_k == chained
    @test no_k_next == chained_next
    @test randsample(rng, population) == chained
end

@testset "R57 device-agnostic iterable materializes once" begin
    rng = Philox4x32(0x902)
    starts = Ref(0)
    population = DeviceAgnosticIterable(collect(Int32(3):Int32(11)), starts)
    expected_next, expected = _chained_unweighted(rng, population.values, 13)
    values, next_rng = randsample_next(rng, population, 13)
    @test starts[] == 1
    @test values == expected
    @test next_rng == expected_next
end

@testset "R58 fixed work, wide cardinality, and O(k)" begin
    rng = Threefry4x64(0x903)
    small = UInt64(11):UInt64(29)
    small_values, small_next = randsample_next(rng, small, 5)
    @test small_next.position == SamplingIR._Position128(1, 0, 64)
    @test small_values == last(_chained_unweighted(rng, small, 5))

    wide = UInt64(0):(UInt64(1)<<32)
    wide_values, wide_next = randsample_next(rng, wide, 5)
    @test wide_next.position == SamplingIR._Position128(2, 0, 128)
    @test wide_values == last(_chained_unweighted(rng, wide, 5))

    reads = Ref(0)
    counted = CountedPopulation(collect(Int32(1):Int32(100)), reads)
    @test length(randsample(rng, counted, 17)) == 17
    @test reads[] == 17
end

@testset "R58 small unweighted samples" begin
    rng = Philox4x32(0x905)
    population = UInt64(0):(UInt64(1)<<32)
    count = 128
    expected_next, expected = _chained_unweighted(rng, population, count)
    values, next_rng = randsample_next(rng, population, count)

    @test values == expected
    @test next_rng == expected_next
end

@testset "R58 parallel unweighted samples preserve the scalar stream" begin
    rng = Philox4x32(0x906)
    for population in (Int32[2, 7, 19], UInt64(0):(UInt64(1)<<32))
        expected_next, expected = _chained_unweighted(rng, population, 8193)
        values, next_rng = randsample_next(rng, population, 8193)
        sync_cpu()
        @test values == expected
        @test next_rng === expected_next
    end
end

@testset "R67 threaded destination sampling uses the CPU chunk kernel" begin
    rng = Philox4x32(0x9061)
    population = Int32[2, 7, 19]
    expected, after = randsample_next(rng, population, 8193)
    destination = similar(expected)

    returned, next_rng = randsample_next!(rng, population, destination)
    sync_cpu()
    @test returned === destination
    @test destination == expected
    @test next_rng === after
end

@testset "R67 CPU destination storage does not inspect values" begin
    rng = Philox4x32(0x9062)
    symbol_population = Symbol[:red, :green, :blue]
    symbol_expected, symbol_after = randsample_next(rng, symbol_population, 33)
    symbol_destination = Vector{Symbol}(undef, 33)
    symbol_returned, symbol_next =
        randsample_next!(rng, symbol_population, symbol_destination; threaded = false)
    @test symbol_returned === symbol_destination
    @test symbol_destination == symbol_expected
    @test symbol_next === symbol_after

    symbol_storage = Vector{Symbol}(undef, 34)
    symbol_view = @view symbol_storage[2:end]
    symbol_view_returned, symbol_view_next =
        randsample_next!(rng, symbol_population, symbol_view; threaded = true)
    sync_cpu()
    @test symbol_view_returned === symbol_view
    @test symbol_view == symbol_expected
    @test symbol_view_next === symbol_after

    any_population = Any[:red, :green, :blue]
    weights = Float64[1, 2, 3]
    any_expected, any_after = randsample_next(rng, any_population, weights, 33)
    any_destination = Vector{Any}(undef, 33)
    any_returned, any_next =
        randsample_next!(rng, any_population, weights, any_destination; threaded = false)
    @test any_returned === any_destination
    @test any_destination == any_expected
    @test any_next === any_after

    any_storage = Vector{Any}(undef, 34)
    any_view = @view any_storage[2:end]
    any_view_returned, any_view_next =
        randsample_next!(rng, any_population, weights, any_view; threaded = true)
    sync_cpu()
    @test any_view_returned === any_view
    @test any_view == any_expected
    @test any_view_next === any_after

    cuda_rng = SamplingMLD.CUDADevice(:discarded)(rng)
    @test_throws ArgumentError randsample!(
        cuda_rng,
        symbol_population,
        Vector{Symbol}(undef, 1),
    )
end

@testset "R60 validation and atomic preflight" begin
    rng = Philox4x32(0x904)
    empty = Int32[]
    @test randsample(rng, empty, 0) == Int32[]
    @test randsample_next(rng, empty, 0) == (Int32[], rng)
    @test randsample(rng, empty) == Int32[]
    @test_throws ArgumentError randsample_next(rng, empty, 1)
    @test_throws ArgumentError randsample_next(rng, 1:3, -1)
    @test_throws ArgumentError randsample_next(rng, 1:3, big(typemax(Int)) + 1)
    @test_throws ArgumentError randsample(rng, UInt64(0):UInt64(typemax(Int)))
    @test_throws ArgumentError randsample(rng, big(0):(big(typemax(UInt64))+1), 0)
    @test_throws ArgumentError randsample(rng, UnknownDeviceIterable([1, 2]), 0)

    largest = typemin(Int):(typemax(Int)-1)
    largest_values, largest_next = randsample_next(rng, largest, 1)
    @test largest_values == last(_chained_unweighted(rng, largest, 1))
    @test largest_next == first(_chained_unweighted(rng, largest, 1))
    @test_throws ArgumentError randsample(rng, typemin(Int):typemax(Int), 0)
    @test_throws ArgumentError randsample(rng, UInt64(0):typemax(UInt64), 0)
    @test_throws ArgumentError randsample(rng, DeclaredHugeIterable())

    huge_generic = big(0):big(typemax(Int))
    huge_values, huge_next = randsample_next(rng, huge_generic, 1)
    @test huge_values == last(_chained_unweighted(rng, huge_generic, 1))
    @test huge_next == first(_chained_unweighted(rng, huge_generic, 1))
    wrong = SamplingCUDAProbe([1, 2, 3])
    @test_throws ArgumentError randsample(rng, wrong, -1)

    last_rng = SamplingIR._rebuild(
        rng,
        SamplingIR._Position64(typemax(UInt64), UInt16(64)),
        rng.device,
    )
    values, terminal = randsample_next(last_rng, 1:3, 1)
    @test values == last(_chained_unweighted(last_rng, 1:3, 1))
    @test terminal.position == SamplingIR._terminal64(typemax(UInt64))
    @test_throws ArgumentError randsample_next(last_rng, 1:3, 2)
    @test last_rng.position == SamplingIR._Position64(typemax(UInt64), UInt16(64))
end

@testset "R67 unweighted destination sampling" begin
    rng = Philox4x32(0x9067)
    population = Int32[11, 13, 17, 19]
    expected, after = randsample_next(rng, population, 33)
    destination = SamplingSerialProbe(similar(expected))

    returned, next_rng = randsample_next!(rng, population, destination; threaded = false)
    @test returned === destination
    @test destination.values == expected
    @test next_rng === after

    @test randsample!(rng, population, destination; threaded = false) === destination
    @test destination.values == expected
end

@testset "R67 destination axes, overlap, and preflight" begin
    rng = Philox4x32(0x9068)
    population = Int32[23, 29, 31]
    expected, after = randsample_next(rng, population, 33)
    destination = ZeroBasedVector(Vector{Int32}(undef, 33))

    returned, next_rng = randsample_next!(rng, population, destination; threaded = false)
    @test returned === destination
    @test collect(destination) == expected
    @test next_rng === after

    aliased = copy(population)
    @test_throws ArgumentError randsample!(rng, aliased, aliased; threaded = false)
    @test aliased == population

    @test_throws ArgumentError randsample!(rng, population, zeros(Int16, 3))
    @test_throws ArgumentError randsample!(
        rng,
        population,
        SamplingCUDAProbe(Int32[0, 0, 0]),
    )

    empty = Int32[]
    @test randsample_next!(rng, population, empty; threaded = false) == (empty, rng)

    terminal = SamplingIR._rebuild(
        rng,
        SamplingIR._Position64(typemax(UInt64), UInt16(64)),
        rng.device,
    )
    preserved = fill(Int32(-1), 2)
    @test_throws ArgumentError randsample_next!(terminal, population, preserved)
    @test preserved == fill(Int32(-1), 2)
end
