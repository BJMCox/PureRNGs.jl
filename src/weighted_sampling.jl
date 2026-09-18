const _WEIGHT_BITS = UInt16(53)
const _WEIGHTED_LOOKUP_LANES = 32

@noinline function _invalid_weights()
    throw(
        ArgumentError(
            "weights must be finite and non-negative with a finite positive total",
        ),
    )
end

function _collect_weights(weights)
    converted = Vector{Float64}(undef, length(weights))
    invalid = false
    @inbounds for ordinal in eachindex(converted)
        weight = Float64(_population_value(weights, UInt64(ordinal)))
        converted[ordinal] = weight
        invalid |= !isfinite(weight) || weight < zero(Float64)
    end
    invalid && _invalid_weights()
    return converted
end

@inline function _fold_weights_cpu(weights)
    cumulative = Vector{Float64}(undef, length(weights))
    total = zero(Float64)
    invalid = false
    @inbounds for ordinal = 1:length(weights)
        weight = Float64(_population_value(weights, UInt64(ordinal)))
        invalid |= !isfinite(weight) || weight < zero(Float64)
        total += weight
        cumulative[ordinal] = total
    end
    invalid |= !isfinite(total) || total <= zero(Float64)
    invalid && _invalid_weights()
    return total, cumulative
end

"""
    WeightTable(weights)

Prepared weights for repeated weighted sampling. Holds the cumulative table the
weighted forms build on every call, so a caller with fixed weights pays that cost
once. Accepted wherever a weight vector is. Draws with a table equal draws with the
weights it was built from. A table is CPU data; a device generator rejects it.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> pop = [10, 20, 30, 40];

julia> weights = [1.0, 1.0, 1.0, 7.0];

julia> table = WeightTable(weights);

julia> randsample(rng, pop, table, 6)
6-element Vector{Int64}:
 20
 20
 40
 40
 40
 10

julia> randsample(rng, pop, table, 6) == randsample(rng, pop, weights, 6)
true
```
"""
struct WeightTable
    total::Float64
    cumulative::Vector{Float64}
end

function WeightTable(weights)
    total, cumulative = _fold_weights_cpu(weights)
    return WeightTable(total, cumulative)
end

@noinline function _device_weight_table()
    throw(ArgumentError("WeightTable is a CPU value"))
end

@inline function _check_sampling_device(rng, table::WeightTable, _noun)
    rng.device isa _CPUBackend || _device_weight_table()
    return false
end

@inline _weight_count(weights) = UInt64(length(weights))
@inline _weight_count(table::WeightTable) = UInt64(length(table.cumulative))

function _prepare_weight_scan(rng::_CPUGenerators, weights, _agnostic::Bool)
    total, cumulative = _fold_weights_cpu(weights)
    return nothing, total, cumulative
end

@inline _prepare_weight_scan(rng::_CPUGenerators, table::WeightTable, _agnostic::Bool) =
    (nothing, table.total, table.cumulative)

function _transfer_weights(device, weights::Vector{Float64})
    transferred = _allocate_array(device, Float64, (length(weights),))
    copyto!(transferred, weights)
    return transferred
end

# One workgroup folds the whole weight vector. A backend whose workgroups cannot
# hold this many workitems overrides the hook.
@inline _weight_fold_lanes(_backend) = Val(1024)
@inline _weight_fold_lane_count(::Val{lanes}) where {lanes} = lanes

# [R59] the cumulative table is a strict Float64 left fold in ordinal order, so
# lane 1 alone accumulates while the other lanes only stage and store.
KernelAbstractions.@kernel function _weighted_fold_kernel!(
    source,
    total_result,
    invalid_result,
    cumulative,
    ::Val{lanes},
    ::Val{validate_elements},
) where {lanes,validate_elements}
    lane = @index(Local, Linear)
    staged = @localmem Float64 (lanes,)
    total = zero(Float64)
    invalid = false
    first = 1
    while first <= length(source)
        ordinal = first + lane - 1
        if ordinal <= length(source)
            @inbounds staged[lane] = Float64(_population_value(source, UInt64(ordinal)))
        end
        @synchronize

        if lane == 1
            last = min(lanes, length(source) - first + 1)
            @inbounds for slot = 1:last
                weight = staged[slot]
                invalid |=
                    validate_elements && (!isfinite(weight) || weight < zero(Float64))
                total += weight
                staged[slot] = total
            end
        end
        @synchronize

        if ordinal <= length(source)
            @inbounds cumulative[ordinal] = staged[lane]
        end
        @synchronize
        first += lanes
    end
    if lane == 1
        invalid |= !isfinite(total) || total <= zero(Float64)
        @inbounds begin
            total_result[1] = total
            invalid_result[1] = invalid
        end
    end
end

function _prepare_weight_scan(rng, weights, agnostic::Bool)
    source = agnostic ? _transfer_weights(rng.device, _collect_weights(weights)) : weights
    cumulative = _allocate_array(rng.device, Float64, (length(source),))
    total_result = _allocate_array(rng.device, Float64, (1,))
    invalid_result = _allocate_array(rng.device, Bool, (1,))
    backend = _fill_backend(cumulative)
    lanes = _weight_fold_lanes(backend)
    lane_count = _weight_fold_lane_count(lanes)
    _weighted_fold_kernel!(backend)(
        source,
        total_result,
        invalid_result,
        cumulative,
        lanes,
        Val(!agnostic);
        ndrange = lane_count,
        workgroupsize = lane_count,
    )
    only(Array(invalid_result)) && _invalid_weights()
    return nothing, total_result, cumulative
end

# `threshold < cumulative[end]`; odd spans overlap one index between both halves.
@inline function _weighted_cdf_index(cumulative, threshold)
    lower = 1
    span = length(cumulative)
    @inbounds while span > 1
        half = span >>> 1
        middle = lower + half - 1
        lower = ifelse(!(threshold < cumulative[middle]), middle + 1, lower)
        span -= half
    end
    return lower
end

@inline function _fill_weighted_samples_cpu_unchecked!(
    rng::_CPUGenerators,
    position,
    population,
    total::Float64,
    cumulative,
    destination,
    ordinals::UnitRange{Int},
)
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    indices = eachindex(destination)
    ordinal = first(ordinals)
    last_ordinal = last(ordinals)
    if length(ordinals) >= _WEIGHTED_LOOKUP_LANES
        thresholds = Vector{Float64}(undef, _WEIGHTED_LOOKUP_LANES)
        lower = Vector{Int}(undef, _WEIGHTED_LOOKUP_LANES)
        @inbounds while ordinal <= last_ordinal - (_WEIGHTED_LOOKUP_LANES - 1)
            for lane = 1:_WEIGHTED_LOOKUP_LANES
                raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(53))
                thresholds[lane] = _weighted_threshold_from_bits(raw, total)
                lower[lane] = 1
            end
            span = length(cumulative)
            while span > 1
                half = span >>> 1
                @inbounds @simd for lane = 1:_WEIGHTED_LOOKUP_LANES
                    middle = lower[lane] + half - 1
                    lower[lane] = ifelse(
                        !(thresholds[lane] < cumulative[middle]),
                        middle + 1,
                        lower[lane],
                    )
                end
                span -= half
            end
            for lane = 1:_WEIGHTED_LOOKUP_LANES
                population_index = lower[lane]
                index = _sampling_destination_index(indices, ordinal + lane - 1)
                destination[index] = _population_value(population, UInt64(population_index))
            end
            ordinal += _WEIGHTED_LOOKUP_LANES
        end
    end
    @inbounds while ordinal <= last_ordinal
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(53))
        threshold = _weighted_threshold_from_bits(raw, total)
        population_index = _weighted_cdf_index(cumulative, threshold)
        index = _sampling_destination_index(indices, ordinal)
        destination[index] = _population_value(population, UInt64(population_index))
        ordinal += 1
    end
    return destination
end

@inline function _fill_weighted_samples!(
    ::KernelAbstractions.CPU,
    rng::_CPUGenerators,
    population,
    _weights,
    total::Float64,
    cumulative,
    destination,
)
    # Lane-aligned chunks keep every workitem on the vectorized lookup path.
    chunk_elements = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_WEIGHT_BITS))
    chunk_elements -= chunk_elements % _WEIGHTED_LOOKUP_LANES
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), _WEIGHT_BITS)
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        _fill_weighted_samples_cpu_unchecked!(
            rng,
            position,
            population,
            total,
            cumulative,
            destination,
            first:last,
        )
    end
    return destination
end

@inline function _fill_weighted_samples!(
    backend,
    rng,
    population,
    _weights,
    total,
    cumulative,
    destination,
)
    thresholds = _allocate_array(rng.device, Float64, (length(destination),))
    _fill_weighted_thresholds!(backend, rng, total, thresholds)
    return _launch_weighted_scan!(backend, population, cumulative, thresholds, destination)
end

@inline function _weighted_threshold_from_bits(raw::UInt64, total::Float64)
    uniform = _from_bits(Float64, raw)
    return min(uniform * total, prevfloat(total))
end

@inline function _weighted_threshold(rng, position, total::Float64)
    raw = _extract_bits_unchecked(rng, _position_block(position), position.bit, Val(53))
    return _weighted_threshold_from_bits(raw, total)
end

KernelAbstractions.@kernel function _weighted_threshold_kernel!(
    rng,
    total_result,
    thresholds,
)
    index = @index(Global, Linear)
    bits_lo, bits_hi = _bit_span(UInt64(index - 1), _WEIGHT_BITS)
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    total = @inbounds total_result[1]
    @inbounds thresholds[index] = _weighted_threshold(rng, position, total)
end

@inline function _fill_weighted_thresholds!(
    ::KernelAbstractions.CPU,
    rng,
    total::Float64,
    thresholds,
)
    isempty(thresholds) && return thresholds
    cursor = _dense_cursor(rng, _position_block(rng.position), rng.position.bit)
    @inbounds for index in eachindex(thresholds)
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(53))
        thresholds[index] = _weighted_threshold_from_bits(raw, total)
    end
    return thresholds
end

@inline function _fill_weighted_thresholds!(backend, rng, total_result, thresholds)
    _weighted_threshold_kernel!(backend)(
        rng,
        total_result,
        thresholds;
        ndrange = length(thresholds),
    )
    return thresholds
end

KernelAbstractions.@kernel function _weighted_binary_search_kernel!(
    population,
    cumulative,
    thresholds,
    destination,
)
    index = @index(Global, Linear)
    threshold = @inbounds thresholds[index]
    lower = 1
    upper = length(cumulative)
    @inbounds while lower < upper
        middle = lower + ((upper - lower) >>> 1)
        if threshold < cumulative[middle]
            upper = middle
        else
            lower = middle + 1
        end
    end
    indices = eachindex(destination)
    @inbounds begin
        destination_index = _sampling_destination_index(indices, index)
        destination[destination_index] = _population_value(population, UInt64(lower))
    end
end

@inline function _launch_weighted_scan!(
    backend,
    population,
    cumulative,
    thresholds,
    destination,
)
    _weighted_binary_search_kernel!(backend)(
        population,
        cumulative,
        thresholds,
        destination;
        ndrange = length(destination),
    )
    return destination
end

function _randsample_next_weighted(rng, population, weights, requested_count)
    population_agnostic = _check_population_device(rng, population)
    weights_agnostic = _check_sampling_device(rng, weights, "weights")
    _check_sampling_serviceability(rng)

    count = requested_count === nothing ? nothing : _sampling_count(requested_count)
    _prevalidate_sampling_cardinality(population, count)
    indexed = _prepare_population(rng.device, population, population_agnostic)
    cardinality = _sampling_cardinality(indexed)
    count === nothing && (count = _sampling_count(cardinality, nothing))
    count > 0 && iszero(cardinality) && _empty_sampling_population()
    _weight_count(weights) == cardinality ||
        throw(ArgumentError("weight length differs from the population cardinality"))

    converted, total, cumulative = _prepare_weight_scan(rng, weights, weights_agnostic)
    next_rng = _sampling_reservation(rng, count, _WEIGHT_BITS)
    destination = _allocate_sampling_result(rng, indexed, count)
    isempty(destination) && return destination, next_rng

    backend = _fill_backend(destination)
    _fill_weighted_samples!(
        backend,
        rng,
        indexed,
        converted,
        total,
        cumulative,
        destination,
    )
    return destination, next_rng
end

function _randsample_next_weighted!(rng, population, weights, destination, threaded::Bool)
    _check_sampling_fill_device(rng, destination)
    _check_sampling_serviceability(rng)
    population_agnostic = _check_population_device(rng, population)
    weights_agnostic = _check_sampling_device(rng, weights, "weights")
    _check_sampling_population_overlap(destination, population)
    _prevalidate_sampling_cardinality(population, length(destination))
    indexed = _prepare_population(rng.device, population, population_agnostic)
    cardinality = _sampling_cardinality(indexed)
    _weight_count(weights) == cardinality ||
        throw(ArgumentError("weight length differs from the population cardinality"))
    _check_sampling_destination_eltype(destination, indexed)
    !isempty(destination) && iszero(cardinality) && _empty_sampling_population()

    converted, total, cumulative = _prepare_weight_scan(rng, weights, weights_agnostic)
    next_rng = _sampling_reservation(rng, length(destination), _WEIGHT_BITS)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_weighted_samples_cpu_unchecked!(
            rng,
            rng.position,
            indexed,
            total,
            cumulative,
            destination,
            1:length(destination),
        )
        return destination, next_rng
    end
    _fill_weighted_samples!(
        _fill_backend(destination),
        rng,
        indexed,
        converted,
        total,
        cumulative,
        destination,
    )
    return destination, next_rng
end

@inline function randsample(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
)
    return first(_randsample_next_weighted(rng, population, weights, nothing))
end


@inline function randsample(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
    count::Integer,
)
    return first(_randsample_next_weighted(rng, population, weights, count))
end


@inline function randsample_next(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
)
    return _randsample_next_weighted(rng, population, weights, nothing)
end


@inline function randsample_next(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
    count::Integer,
)
    return _randsample_next_weighted(rng, population, weights, count)
end

@inline function randsample!(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
    destination::AbstractArray;
    threaded = true,
)
    return first(
        _randsample_next_weighted!(
            rng,
            population,
            weights,
            destination,
            _check_threaded(threaded),
        ),
    )
end

@inline function randsample_next!(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
    destination::AbstractArray;
    threaded = true,
)
    return _randsample_next_weighted!(
        rng,
        population,
        weights,
        destination,
        _check_threaded(threaded),
    )
end
