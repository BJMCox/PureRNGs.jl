const WeightedIR = PureRNGs

const WEIGHTED_GOLDEN = (
    Philox2x32 => Int32[40, 40, 40, 30, 30, 40, 20, 30],
    Philox4x32 => Int32[30, 40, 40, 20, 30, 20, 40, 30],
    Philox2x64 => Int32[20, 40, 40, 40, 30, 20, 30, 40],
    Philox4x64 => Int32[30, 40, 40, 10, 40, 30, 30, 30],
    Threefry2x32 => Int32[40, 40, 30, 20, 40, 10, 30, 30],
    Threefry4x32 => Int32[20, 20, 40, 30, 40, 30, 10, 40],
    Threefry2x64 => Int32[40, 30, 40, 30, 20, 40, 40, 10],
    Threefry4x64 => Int32[40, 30, 30, 30, 30, 40, 40, 30],
)

const WEIGHTED_OFFSET_GOLDEN = (
    Philox2x32 => Int32[104, 104, 105, 106, 106, 106, 103, 101, 103, 105, 106, 103],
    Philox4x32 => Int32[104, 106, 105, 104, 105, 103, 104, 105, 106, 104, 103, 103],
    Philox2x64 => Int32[103, 106, 105, 105, 103, 104, 103, 103, 106, 103, 106, 106],
    Philox4x64 => Int32[106, 105, 103, 106, 105, 103, 103, 105, 101, 105, 103, 105],
    Threefry2x32 => Int32[104, 106, 103, 104, 101, 106, 106, 106, 105, 103, 103, 106],
    Threefry4x32 => Int32[106, 101, 103, 103, 106, 106, 104, 106, 101, 103, 103, 103],
    Threefry2x64 => Int32[103, 105, 106, 101, 106, 104, 106, 106, 101, 106, 106, 106],
    Threefry4x64 => Int32[106, 106, 103, 106, 104, 106, 105, 106, 106, 106, 106, 101],
)

struct WeightWrapper{T}
    values::Vector{T}
end

mutable struct CountedWeights{T} <: AbstractVector{T}
    values::Vector{T}
    reads::Int
end

Base.size(weights::CountedWeights) = size(weights.values)
Base.getindex(weights::CountedWeights, index::Int) =
    (weights.reads += 1; weights.values[index])

function _weighted_population_value(population, ordinal::Int)
    if population isa AbstractArray
        offset = CartesianIndices(map(Base.OneTo, size(population)))[ordinal]
        index = CartesianIndex(
            ntuple(
                dimension -> first(axes(population, dimension)) + offset[dimension] - 1,
                ndims(population),
            ),
        )
        return @inbounds population[index]
    end
    return @inbounds population[ordinal]
end

function _weighted_reference(rng, population, weights, k::Int)
    converted =
        Float64[_weighted_population_value(weights, index) for index = 1:length(weights)]
    total = zero(Float64)
    for weight in converted
        total += weight
    end

    thresholds = Vector{Float64}(undef, k)
    position = rng.position
    for index in eachindex(thresholds)
        raw = WeightedIR._extract_bits_unchecked(
            rng,
            WeightedIR.FAMILY_RANGE,
            WeightedIR._position_block(position),
            position.bit,
            Val(53),
        )
        uniform = Float64(raw) * 0x1p-53
        thresholds[index] = min(uniform * total, prevfloat(total))
        position = WeightedIR._advance_position_unchecked(
            position,
            UInt64(53),
            UInt64(0),
            WeightedIR._block_shift(rng),
        )
    end

    order = sortperm(eachindex(thresholds); by = index -> (thresholds[index], index))
    result = Vector{eltype(population)}(undef, k)
    cumulative = zero(Float64)
    ordered_draw = 1
    for population_index in eachindex(converted)
        cumulative += converted[population_index]
        while ordered_draw <= k && thresholds[order[ordered_draw]] < cumulative
            original_index = order[ordered_draw]
            result[original_index] =
                _weighted_population_value(population, population_index)
            ordered_draw += 1
        end
    end
    @assert ordered_draw == k + 1
    return WeightedIR._reserve(rng, UInt64(53k), UInt64(0)), result
end

function _last_weighted_rng(F)
    rng = F(0x9751)
    bit = WeightedIR._block_bits(rng) - UInt16(53)
    position = if rng.position isa WeightedIR._Position64
        WeightedIR._Position64(WeightedIR._max_block(rng), bit)
    else
        WeightedIR._Position128(typemax(UInt64), typemax(UInt64), bit)
    end
    return WeightedIR._rebuild(rng, position, rng.device)
end

@testset "R13 and R59 weighted packed-stream golden vectors" begin
    population = Int32[10, 20, 30, 40]
    weights = Float64[1, 2, 3, 4]
    for (F, expected) in WEIGHTED_GOLDEN
        rng = F(0x9750)
        next_rng, values = randsample_next(rng, population, weights, 8)
        @test values == expected
        @test next_rng === WeightedIR._reserve(rng, UInt64(8 * 53), UInt64(0))
    end
end

@testset "R13, R57, and R59 weighted canonical offset-axis golden vectors" begin
    population = IdentityAxesMatrix(reshape(Int32.(101:106), 2, 3))
    weights = ZeroBasedVector(Float64[1, 0, 4, 2, 3, 5])
    for (F, expected) in WEIGHTED_OFFSET_GOLDEN
        rng = F(0x9761)
        _, values = randsample_next(rng, population, weights, 12)
        @test values == expected
        @test randsample(rng, population, weights, 12) == expected
    end
end

@testset "R59 strict Float64 fold golden boundaries" begin
    rng = Philox4x32(0x9750)
    population = Int32[10, 20, 30, 40]
    weights = Float64[Float64(0x000f5d057718d3b7), Float64(0x0010a2fa88e72c49), 1.0, 1.0]
    converted, total = WeightedIR._prepare_weights(rng, weights, false)
    @test reinterpret(UInt64, total) == 0x4340000000000000
    @test reinterpret(UInt64, (weights[1] + weights[2]) + (weights[3] + weights[4])) ==
          0x4340000000000001
    @test WeightedIR._weighted_threshold(rng, rng.position, total) ==
          Float64(0x000f5d057718d3b6)
    @test randsample(rng, population, weights, 1) == Int32[10]

    destination = Vector{Int32}(undef, 1)
    WeightedIR._scan_weighted!(
        population,
        Float64[0x1p53, 1.0, 1.0, 2.0],
        Float64[0x1p53],
        [1],
        destination,
    )
    @test destination == Int32[40]
    @test converted == weights
end

@testset "R56 and R59 weighted sampling surface and fixed work" begin
    populations = (
        ["a", "b", "c", "d"],
        reshape(collect(Int16(1):Int16(6)), 2, 3),
        IdentityAxesMatrix(reshape(collect(Int32(11):Int32(16)), 2, 3)),
        UInt16(10):UInt16(3):UInt16(31),
    )
    weight_sets = (
        1:4,
        Float32[1, 0, 4, 2, 3, 5],
        ZeroBasedVector(Rational{Int}[1//2, 3//2, 0//1, 4//1, 2//1, 1//1]),
        UInt8[1, 4, 2, 3, 5, 6, 7, 8],
    )

    for F in FAMILY_TYPES, (population, weights) in zip(populations, weight_sets)
        rng = F(0x9752)
        expected_next, expected = _weighted_reference(rng, population, weights, 11)
        next_rng, values = randsample_next(rng, population, weights, 11)

        @test values == expected
        @test next_rng === expected_next
        @test randsample(rng, population, weights, 11) == expected
        @test randsample(rng, population, weights, 5) == expected[1:5]
        @test rng.position == F(0x9752).position

        default_expected_next, default_expected =
            _weighted_reference(rng, population, weights, length(population))
        default_next, default_values = randsample_next(rng, population, weights)
        @test default_values == default_expected
        @test default_next === default_expected_next
        @test randsample(rng, population, weights) == default_expected
    end
end

@testset "R59 sorted batch equals chained weighted scalar sampling" begin
    population = collect('a':'f')
    weights = [0.0, 1.0, 7.0, 0.0, 2.0, 4.0]
    for F in FAMILY_TYPES
        rng = F(0x9753)
        batch_next, batch = randsample_next(rng, population, weights, 17)
        cursor = rng
        chained = similar(batch)
        for index in eachindex(chained)
            cursor, value = randsample_next(cursor, population, weights, 1)
            chained[index] = only(value)
        end
        @test batch == chained
        @test batch_next === cursor
        @test all(value -> value in ('b', 'c', 'e', 'f'), batch)
        @test randsample(rng, population, weights, 9) == batch[1:9]
    end
end

@testset "R59 and R60 weighted validation precedes generation" begin
    rng = Philox4x32(0x9754)
    population = [10, 20, 30]

    for invalid in (
        [1.0, 2.0],
        [1.0, -1.0, 2.0],
        [1.0, Inf, 2.0],
        [1.0, NaN, 2.0],
        [0.0, 0.0, 0.0],
        [floatmax(Float64), floatmax(Float64), 1.0],
    )
        @test_throws ArgumentError randsample(rng, population, invalid, 2)
        @test_throws ArgumentError randsample_next(rng, population, invalid, 2)
    end

    @test_throws ArgumentError randsample(rng, population, ones(3), -1)
    @test_throws ArgumentError randsample_next(rng, population, ones(3), -1)
    @test_throws ArgumentError randsample(rng, population, ones(3), big(typemax(Int)) + 1)
    @test_throws ArgumentError randsample_next(
        rng,
        population,
        ones(3),
        big(typemax(Int)) + 1,
    )
    @test_throws ArgumentError randsample(rng, DeclaredHugeIterable(), ones(1))

    wrong_weights = SamplingCUDAProbe([1.0, 2.0, 3.0])
    device_error = try
        randsample(rng, population, wrong_weights, -1)
        nothing
    catch caught
        caught
    end
    @test device_error isa ArgumentError
    @test occursin("weights device", sprint(showerror, device_error))

    @test !applicable(randsample, rng, population, WeightWrapper([1.0, 2.0, 3.0]))
    @test !applicable(randsample_next, rng, population, WeightWrapper([1.0, 2.0, 3.0]))
    @test !applicable(randsample, rng, population, (1.0, 2.0, 3.0))
    @test !applicable(randsample_next, rng, population, (1.0, 2.0, 3.0))

    exhausted = WeightedIR._reserve(_last_weighted_rng(Philox4x32), UInt64(53), UInt64(0))
    @test_throws ArgumentError randsample_next(exhausted, population, [1.0, -1.0, 2.0], 1)
    @test_throws ArgumentError randsample_next(exhausted, population, ones(3), 1)

    empty_next, empty = randsample_next(rng, population, ones(3), 0)
    @test empty isa Vector{Int}
    @test isempty(empty)
    @test empty_next === rng
    @test isempty(randsample(rng, population, ones(3), 0))
    @test_throws ArgumentError randsample(rng, Int[], Float64[], 0)
    @test WeightedIR._weighted_sortperm(rng.device, [0.5, 0.1, 0.5, 0.1]) == [2, 4, 1, 3]
end

@testset "R59 weight conversion is one population-order pass" begin
    rng = Philox4x64(0x9755)
    weights = CountedWeights([1.0, 0.0, 3.0, 2.0], 0)
    expected_next, expected = _weighted_reference(rng, 11:14, weights.values, 13)
    next_rng, values = randsample_next(rng, 11:14, weights, 13)
    @test values == expected
    @test next_rng === expected_next
    @test weights.reads == length(weights)
end

@testset "R54 and R60 weighted capacity is atomic" begin
    population = [:left, :right]
    weights = [1.0, 1.0]
    for F in FAMILY_TYPES
        last = _last_weighted_rng(F)
        terminal, value = randsample_next(last, population, weights, 1)
        @test length(value) == 1
        @test terminal.position.bit == WeightedIR._EXHAUSTED_BIT
        @test_throws ArgumentError randsample(last, population, weights, 2)
        @test_throws ArgumentError randsample_next(last, population, weights, 2)
        @test last.position.bit != WeightedIR._EXHAUSTED_BIT
    end
end
