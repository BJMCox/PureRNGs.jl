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

@inline function _convert_and_fold_weights!(
    source,
    converted,
    total_result,
    invalid_result,
    cumulative,
    ::Val{validate_elements},
) where {validate_elements}
    total = zero(Float64)
    invalid = false
    @inbounds for ordinal = 1:length(source)
        weight = Float64(_population_value(source, UInt64(ordinal)))
        converted === nothing || (converted[ordinal] = weight)
        invalid |= validate_elements && (!isfinite(weight) || weight < zero(Float64))
        total += weight
        cumulative === nothing || (cumulative[ordinal] = total)
    end
    invalid |= !isfinite(total) || total <= zero(Float64)
    @inbounds begin
        total_result[1] = total
        invalid_result[1] = invalid
    end
    return converted
end

function _prepare_weight_scan(rng::_CPUGenerators, weights, _agnostic::Bool)
    cumulative = Vector{Float64}(undef, length(weights))
    total_result = Vector{Float64}(undef, 1)
    invalid_result = Vector{Bool}(undef, 1)
    _convert_and_fold_weights!(
        weights,
        nothing,
        total_result,
        invalid_result,
        cumulative,
        Val(true),
    )
    invalid_result[1] && _invalid_weights()
    return nothing, total_result[1], cumulative
end

KernelAbstractions.@kernel function _prepare_weights_kernel!(
    source,
    converted,
    total_result,
    invalid_result,
    cumulative,
    validate_elements,
)
    if @index(Global, Linear) == 1
        _convert_and_fold_weights!(
            source,
            converted,
            total_result,
            invalid_result,
            cumulative,
            validate_elements,
        )
    end
end

KernelAbstractions.@kernel function _total_weights_kernel!(
    weights,
    total_result,
    invalid_result,
)
    if @index(Global, Linear) == 1
        total = zero(Float64)
        @inbounds for ordinal = 1:length(weights)
            total += weights[ordinal]
        end
        total_result[1] = total
        invalid_result[1] = !isfinite(total) || total <= zero(Float64)
    end
end

function _transfer_weights(device, weights::Vector{Float64})
    transferred = _allocate_array(device, Float64, (length(weights),))
    copyto!(transferred, weights)
    return transferred
end

function _prepare_weights(rng, weights, agnostic::Bool)
    source = agnostic ? _transfer_weights(rng.device, _collect_weights(weights)) : weights
    converted = agnostic ? source : _allocate_array(rng.device, Float64, (length(source),))
    total_result = _allocate_array(rng.device, Float64, (1,))
    invalid_result = _allocate_array(rng.device, Bool, (1,))
    backend = _fill_backend(converted)
    if agnostic
        _total_weights_kernel!(backend)(
            converted,
            total_result,
            invalid_result;
            ndrange = 1,
        )
    else
        _prepare_weights_kernel!(backend)(
            source,
            converted,
            total_result,
            invalid_result,
            nothing,
            Val(true);
            ndrange = 1,
        )
    end
    invalid = only(Array(invalid_result))
    invalid && _invalid_weights()
    return converted, total_result
end

@inline function _prepare_weight_scan(rng, weights, agnostic::Bool)
    converted, total = _prepare_weights(rng, weights, agnostic)
    return converted, total, nothing
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

@inline function _fill_weighted_samples!(
    ::KernelAbstractions.CPU,
    rng::_CPUGenerators,
    population,
    _weights,
    total::Float64,
    cumulative,
    destination,
)
    cursor = _dense_cursor(rng, _position_block(rng.position), rng.position.bit)
    index = firstindex(destination)
    last = lastindex(destination)
    if length(destination) >= _WEIGHTED_LOOKUP_LANES
        thresholds = Vector{Float64}(undef, _WEIGHTED_LOOKUP_LANES)
        lower = Vector{Int}(undef, _WEIGHTED_LOOKUP_LANES)
        @inbounds while index <= last - (_WEIGHTED_LOOKUP_LANES - 1)
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
                destination[index+lane-1] =
                    _population_value(population, UInt64(population_index))
            end
            index += _WEIGHTED_LOOKUP_LANES
        end
    end
    @inbounds while index <= last
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(53))
        threshold = _weighted_threshold_from_bits(raw, total)
        population_index = _weighted_cdf_index(cumulative, threshold)
        destination[index] = _population_value(population, UInt64(population_index))
        index += 1
    end
    return destination
end

@inline function _fill_weighted_samples!(
    backend,
    rng,
    population,
    weights,
    total,
    cumulative,
    destination,
)
    thresholds = _allocate_array(rng.device, Float64, (length(destination),))
    _fill_weighted_thresholds!(backend, rng, total, thresholds)
    order = _weighted_sortperm(rng.device, thresholds)
    return _launch_weighted_scan!(
        rng.device,
        backend,
        population,
        weights,
        thresholds,
        order,
        cumulative,
        destination,
    )
end

@inline function _weighted_threshold_from_bits(raw::UInt64, total::Float64)
    uniform = Float64(raw) * 0x1p-53
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

# `sortperm` is stable, so equal thresholds retain their original-index order.
@inline _weighted_sortperm(device, thresholds) = sortperm(thresholds)

@inline function _scan_weighted!(population, weights, thresholds, order, destination)
    cumulative = zero(Float64)
    ordered_draw = 1
    @inbounds for population_index in eachindex(weights)
        cumulative += weights[population_index]
        while ordered_draw <= length(order) && thresholds[order[ordered_draw]] < cumulative
            original_index = order[ordered_draw]
            destination[original_index] =
                _population_value(population, UInt64(population_index))
            ordered_draw += 1
        end
    end
    return destination
end

KernelAbstractions.@kernel function _weighted_scan_kernel!(
    population,
    weights,
    thresholds,
    order,
    destination,
)
    if @index(Global, Linear) == 1
        _scan_weighted!(population, weights, thresholds, order, destination)
    end
end

@inline function _launch_weighted_scan!(
    ::KernelAbstractions.CPU,
    population,
    weights,
    thresholds,
    order,
    destination,
)
    return _scan_weighted!(population, weights, thresholds, order, destination)
end

@inline function _launch_weighted_scan!(
    _device,
    backend,
    population,
    weights,
    thresholds,
    order,
    cumulative,
    destination,
)
    return _launch_weighted_scan!(
        backend,
        population,
        weights,
        thresholds,
        order,
        destination,
    )
end

@inline function _launch_weighted_scan!(
    backend,
    population,
    weights,
    thresholds,
    order,
    destination,
)
    _weighted_scan_kernel!(backend)(
        population,
        weights,
        thresholds,
        order,
        destination;
        ndrange = 1,
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
    UInt64(length(weights)) == cardinality ||
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

@inline function randsample(
    rng::AbstractPureRNG,
    population,
    weights::AbstractVector{<:Real},
)
    return first(_randsample_next_weighted(rng, population, weights, nothing))
end


@inline function randsample(
    rng::AbstractPureRNG,
    population,
    weights::AbstractVector{<:Real},
    count::Integer,
)
    return first(_randsample_next_weighted(rng, population, weights, count))
end


@inline function randsample_next(
    rng::AbstractPureRNG,
    population,
    weights::AbstractVector{<:Real},
)
    return _randsample_next_weighted(rng, population, weights, nothing)
end


@inline function randsample_next(
    rng::AbstractPureRNG,
    population,
    weights::AbstractVector{<:Real},
    count::Integer,
)
    return _randsample_next_weighted(rng, population, weights, count)
end
