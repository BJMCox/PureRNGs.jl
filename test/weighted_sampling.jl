const WeightedIR = PureRNGs

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

@testset "R59 strict Float64 fold golden boundaries" begin
    population = Int32[10, 20, 30, 40]
    weights = Float64[Float64(0x000f5d057718d3b7), Float64(0x0010a2fa88e72c49), 1.0, 1.0]
    @test randsample(Philox4x32(0x9750), population, weights, 1) == Int32[10]

    weights = [0.0, nextfloat(0.0), 0.0, nextfloat(0.0)]
    for F in GENERATOR_TYPES
        rng = F(0x9750)
        expected_next, expected = _weighted_reference(rng, population, weights, 33)
        values, next_rng = randsample_next(rng, population, weights, 33)
        @test values == expected
        @test next_rng === expected_next
    end
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

    for F in GENERATOR_TYPES, (population, weights) in zip(populations, weight_sets)
        rng = F(0x9752)
        expected_next, expected = _weighted_reference(rng, population, weights, 11)
        values, next_rng = randsample_next(rng, population, weights, 11)

        @test values == expected
        @test next_rng === expected_next
    end

    rng = Philox4x32(0x9752)
    population, weights = first(zip(populations, weight_sets))
    _, expected = _weighted_reference(rng, population, weights, 11)
    @test randsample(rng, population, weights, 5) == expected[1:5]
    default_expected_next, default_expected =
        _weighted_reference(rng, population, weights, length(population))
    default_values, default_next = randsample_next(rng, population, weights)
    @test default_values == default_expected
    @test default_next === default_expected_next
    @test randsample(rng, population, weights) == default_expected
end

@testset "R59 weighted batch equals chained weighted scalar sampling" begin
    population = collect('a':'f')
    weights = [0.0, 1.0, 7.0, 0.0, 2.0, 4.0]
    rng = Philox4x32(0x9753)
    batch, batch_next = randsample_next(rng, population, weights, 33)
    cursor = rng
    chained = similar(batch)
    for index in eachindex(chained)
        value, cursor = randsample_next(cursor, population, weights, 1)
        chained[index] = only(value)
    end
    @test batch == chained
    @test batch_next === cursor
end

@testset "R59 and R60 weighted validation precedes generation" begin
    rng = Philox4x32(0x9754)
    population = [10, 20, 30]

    invalid = (
        ([1.0, 2.0], 1),
        ([1.0, -1.0, 2.0], 1),
        ([1.0, Inf, 2.0], 1),
        ([1.0, NaN, 2.0], 1),
        (zeros(3), 1),
        (fill(floatmax(Float64), 3), 1),
        (ones(3), big(typemax(Int)) + 1),
    )
    for (weights, count) in invalid
        @test_throws ArgumentError randsample_next(rng, population, weights, count)
    end
    after_rejections, _ = randsample_next(rng, population, ones(3), 4)
    pristine, _ = randsample_next(Philox4x32(0x9754), population, ones(3), 4)
    @test after_rejections == pristine

    wrong_weights = SamplingCUDAProbe([1.0, 2.0, 3.0])
    @test_throws ArgumentError randsample(rng, population, wrong_weights, -1)

    empty, empty_next = randsample_next(rng, population, ones(3), 0)
    @test isempty(empty)
    @test empty_next === rng
    @test isempty(randsample(rng, population, ones(3), 0))
end

@testset "R59 weight conversion is one population-order pass" begin
    rng = Philox4x64(0x9755)
    weights = CountedWeights([1.0, 0.0, 3.0, 2.0], 0)
    expected_next, expected = _weighted_reference(rng, 11:14, weights.values, 13)
    values, next_rng = randsample_next(rng, 11:14, weights, 13)
    @test values == expected
    @test next_rng === expected_next
    @test weights.reads == length(weights)
end

@testset "R54 and R60 weighted capacity is atomic" begin
    population = [:left, :right]
    weights = [1.0, 1.0]
    for F in (Philox2x32, Threefry4x64)
        last = _last_weighted_rng(F)
        value, terminal = randsample_next(last, population, weights, 1)
        @test length(value) == 1
        @test terminal.position.bit == WeightedIR._EXHAUSTED_BIT
        @test_throws StreamExhausted randsample(last, population, weights, 2)
        @test_throws StreamExhausted randsample_next(last, population, weights, 2)
        again, _ = randsample_next(last, population, weights, 1)
        @test again == value
    end
end

@testset "R67 weighted destination sampling" begin
    rng = Philox4x32(0x9756)
    population = Int32[10, 20, 30, 40]
    weights = Float64[1, 2, 3, 4]
    expected, after = randsample_next(rng, population, weights, 33)
    destination = SamplingSerialProbe(similar(expected))

    returned, next_rng =
        randsample_next!(rng, population, weights, destination; threaded = false)
    @test returned === destination
    @test destination.values == expected
    @test next_rng === after

    @test randsample!(rng, population, weights, destination; threaded = false) ===
          destination
    @test destination.values == expected

    last = _last_weighted_rng(Philox4x32)
    preserved = fill(Int32(-1), 2)
    before = copy(preserved)
    @test_throws StreamExhausted randsample_next!(last, population, weights, preserved)
    @test preserved == before
end

@testset "R67 weighted destination may overlap prepared weights" begin
    rng = Philox4x32(0x9757)
    population = Float64.(1:33)
    weights = Float64.(1:33)
    expected, after = randsample_next(rng, population, weights, 33)

    returned, next_rng =
        randsample_next!(rng, population, weights, weights; threaded = false)
    @test returned === weights
    @test weights == expected
    @test next_rng === after

    empty = Float64[]
    @test randsample_next!(rng, population, ones(33), empty; threaded = false) ==
          (empty, rng)
end

@testset "R67 Cartesian destination order" begin
    rng = Philox4x32(0x9758)
    population = Int32[10, 20, 30, 40]
    weights = Float64[1, 2, 3, 4]

    for maybe_weights in (nothing, weights)
        expected, after =
            maybe_weights === nothing ? randsample_next(rng, population, 6) :
            randsample_next(rng, population, maybe_weights, 6)
        destination = IdentityAxesMatrix(Matrix{Int32}(undef, 2, 3))
        returned, next_rng =
            maybe_weights === nothing ?
            randsample_next!(rng, population, destination; threaded = false) :
            randsample_next!(rng, population, maybe_weights, destination; threaded = false)
        @test returned === destination
        @test vec(destination.data) == expected
        @test next_rng === after
    end
end

@testset "R67 threaded weighted fill matches the serial fill" begin
    rng = Philox4x32(0x77e)
    population = collect(1:64)
    weights = Float64.(1:64)
    serial = Vector{Int}(undef, 300_000)
    threaded = similar(serial)
    _, after_serial = randsample_next!(rng, population, weights, serial; threaded = false)
    _, after_threaded =
        randsample_next!(rng, population, weights, threaded; threaded = true)
    sync_cpu()
    @test serial == threaded
    @test after_serial === after_threaded
    # The serial fill allocates one cumulative vector, never per element.
    longer = Vector{Int}(undef, 900_000)
    randsample_next!(rng, population, weights, serial; threaded = false)
    randsample_next!(rng, population, weights, longer; threaded = false)
    @test @allocated(
        randsample_next!(rng, population, weights, serial; threaded = false)
    ) == @allocated(randsample_next!(rng, population, weights, longer; threaded = false))

    # A length of four whole chunks plus five leaves a final chunk under one lane width.
    ragged = Vector{Int}(undef, 4 * 2464 + 5)
    ragged_serial = similar(ragged)
    randsample_next!(rng, population, weights, ragged_serial; threaded = false)
    randsample_next!(rng, population, weights, ragged; threaded = true)
    sync_cpu()
    @test ragged == ragged_serial
end

@testset "WeightTable draws equal weight-vector draws" begin
    population = Int32[10, 20, 30, 40]
    weights = Float64[1, 2, 3, 4]
    table = WeightTable(weights)
    for F in GENERATOR_TYPES
        rng = F(0x9770)
        @test randsample(rng, population, table, 9) ==
              randsample(rng, population, weights, 9)
        a, next_a = randsample_next(rng, population, table, 9)
        b, next_b = randsample_next(rng, population, weights, 9)
        @test a == b && next_a === next_b
        dest_a = Vector{Int32}(undef, 9)
        dest_b = similar(dest_a)
        randsample!(rng, population, table, dest_a)
        randsample!(rng, population, weights, dest_b)
        @test dest_a == dest_b
        _, n1 = randsample_next!(rng, population, table, dest_a; threaded = false)
        _, n2 = randsample_next!(rng, population, weights, dest_b; threaded = false)
        @test dest_a == dest_b && n1 === n2
    end
    @test_throws ArgumentError WeightTable([1.0, -1.0])
    @test_throws ArgumentError WeightTable([1.0, NaN])
    @test_throws ArgumentError WeightTable(zeros(3))
    @test_throws ArgumentError randsample(
        Philox4x32(1),
        population,
        WeightTable([1.0, 1.0]),
        1,
    )
    # A top-level `@allocated` also counts the boxed return tuple, so measure in a
    # function where only the fill's own allocations remain.
    serial_fill_bytes(rng, population, weights, destination) = @allocated(
        randsample_next!(rng, population, weights, destination; threaded = false)
    )
    rng = Philox4x32(0x9771)
    destination = Vector{Int32}(undef, 64)
    wide_population = Int32[10, 20, 30, 40, 50, 60, 70, 80]
    wide_weights = Float64[1, 2, 3, 4, 5, 6, 7, 8]
    wide_table = WeightTable(wide_weights)
    for _ = 1:2
        serial_fill_bytes(rng, population, table, destination)
        serial_fill_bytes(rng, wide_population, wide_table, destination)
        serial_fill_bytes(rng, population, weights, destination)
        serial_fill_bytes(rng, wide_population, wide_weights, destination)
    end
    # Bounds-check and coverage flags add a fixed cost to every fill, so compare two
    # weight lengths instead of a byte count. The table carries its cumulative vector.
    @test serial_fill_bytes(rng, wide_population, wide_table, destination) ==
          serial_fill_bytes(rng, population, table, destination)
    # The weight vector allocates one Float64 cumulative entry per weight.
    @test serial_fill_bytes(rng, wide_population, wide_weights, destination) -
          serial_fill_bytes(rng, population, weights, destination) ==
          8 * (length(wide_weights) - length(weights))
end
