module PureRNGsKernelAbstractionsExt

import KernelAbstractions
import PureRNGs
using KernelAbstractions: @index, @localmem, @synchronize

const IR = PureRNGs

# A device destination resolves to the launchable KernelAbstractions backend.
# The core keeps the host method, which never names a KernelAbstractions type.
@inline IR._fill_backend(::IR._BackendToken, destination) =
    KernelAbstractions.get_backend(destination)

KernelAbstractions.@kernel function _uniform_fill_kernel!(
    rng,
    destination,
    ::Val{T},
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds indices[firstindex(indices)+ordinal-1]
    bits_lo, bits_hi = IR._bit_span(UInt64(ordinal - 1), IR._draw_bits(T))
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = IR._draw_unchecked(rng, position, T)
end

KernelAbstractions.@kernel function _uniform_fill_grouped_kernel!(
    rng,
    destination,
    ::Val{T},
    group::Val{N},
) where {T,N}
    workitem = @index(Global, Linear)
    first = (workitem - 1) * N + 1
    bits_lo, bits_hi = IR._bit_span(UInt64(first - 1), IR._draw_bits(T))
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    IR._fill_uniform_grouped_unchecked!(rng, position, destination, T, first, group)
end

@inline _fill_kernel(::Val{:uniform}) = _uniform_fill_kernel!
@inline _fill_grouped_kernel(::Val{:uniform}) = _uniform_fill_grouped_kernel!

@inline function IR._launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    ::Nothing,
) where {T}
    _fill_kernel(codec)(backend)(rng, destination, Val(T); ndrange = length(destination))
    return destination
end

@inline function IR._launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    plan::Tuple{Val{:grouped},Val{N}},
) where {T,N}
    group = plan[2]
    workitems = cld(length(destination), IR._fill_group_size(group))
    _fill_grouped_kernel(codec)(backend)(
        rng,
        destination,
        Val(T),
        group;
        ndrange = workitems,
    )
    return destination
end

KernelAbstractions.@kernel function _transformed_fill_kernel!(
    rng,
    destination,
    ::Val{T},
    codec,
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds indices[firstindex(indices)+ordinal-1]
    bits_lo, bits_hi = IR._bit_span(UInt64(ordinal - 1), IR._fill_width(codec, T))
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = IR._transformed_draw_unchecked(codec, rng, position, T)
end

KernelAbstractions.@kernel function _transformed_fill_grouped_kernel!(
    rng,
    destination,
    ::Val{T},
    group::Val{N},
    codec,
) where {T,N}
    workitem = @index(Global, Linear)
    first = (workitem - 1) * N + 1
    bits_lo, bits_hi = IR._bit_span(UInt64(first - 1), IR._fill_width(codec, T))
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    IR._fill_transformed_grouped_unchecked!(
        rng,
        position,
        destination,
        T,
        first,
        group,
        codec,
    )
end

@inline function IR._launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::IR._TransformedFillCodec,
    ::Nothing,
) where {T}
    _transformed_fill_kernel!(backend)(
        rng,
        destination,
        Val(T),
        codec;
        ndrange = length(destination),
    )
    return destination
end

@inline function IR._launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::IR._TransformedFillCodec,
    plan::Tuple{Val{:grouped},Val{N}},
) where {T,N}
    group = plan[2]
    workitems = cld(length(destination), IR._fill_group_size(group))
    _transformed_fill_grouped_kernel!(backend)(
        rng,
        destination,
        Val(T),
        group,
        codec;
        ndrange = workitems,
    )
    return destination
end

KernelAbstractions.@kernel function _range_fill_kernel!(rng, destination, range, span)
    index = @index(Global, Linear)
    width = IR._range_bits(span)
    bits_lo, bits_hi = IR._bit_span(UInt64(index - 1), width)
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = IR._draw_range_unchecked(rng, position, range, span)
end

KernelAbstractions.@kernel function _range_fill_grouped_kernel!(
    rng,
    destination,
    range,
    span,
    group::Val{2},
)
    workitem = @index(Global, Linear)
    first = (workitem - 1) * 2 + 1
    width = IR._range_bits(span)
    bits_lo, bits_hi = IR._bit_span(UInt64(first - 1), width)
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    IR._fill_range_grouped_unchecked!(rng, position, destination, range, span, first, group)
end

@inline function IR._launch_range!(backend, rng, destination, range, span)
    plan =
        IR._device_fill_plan(backend, rng, IR._RangeCodec(range, span), eltype(destination))
    if plan === nothing
        _range_fill_kernel!(backend)(
            rng,
            destination,
            range,
            span;
            ndrange = length(destination),
        )
        return destination
    end
    group = plan[2]
    workitems = cld(length(destination), IR._fill_group_size(group))
    _range_fill_grouped_kernel!(backend)(
        rng,
        destination,
        range,
        span,
        group;
        ndrange = workitems,
    )
    return destination
end

KernelAbstractions.@kernel function _unweighted_sample_kernel!(
    rng,
    population,
    cardinality::UInt64,
    destination,
    width::UInt16,
)
    draw_ordinal = @index(Global, Linear)
    bits_lo, bits_hi = IR._bit_span(UInt64(draw_ordinal - 1), width)
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    population_ordinal = IR._range_offset(rng, position, cardinality) + UInt64(1)
    indices = eachindex(destination)
    index = IR._sampling_destination_index(indices, draw_ordinal)
    @inbounds destination[index] = IR._population_value(population, population_ordinal)
end

KernelAbstractions.@kernel function _unweighted_sample_grouped_kernel!(
    rng,
    population,
    cardinality::UInt64,
    destination,
    group::Val{2},
)
    workitem = @index(Global, Linear)
    first_ordinal = (workitem - 1) * 2 + 1
    bits_lo, bits_hi = IR._bit_span(UInt64(first_ordinal - 1), UInt16(64))
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    IR._fill_unweighted_grouped_unchecked!(
        rng,
        position,
        population,
        cardinality,
        destination,
        first_ordinal,
        group,
    )
end

@inline function IR._launch_unweighted_sample!(
    backend,
    rng,
    population,
    cardinality::UInt64,
    destination,
    width::UInt16,
)
    plan = IR._device_fill_plan(
        backend,
        rng,
        IR._PopulationCodec(cardinality),
        eltype(destination),
    )
    if plan !== nothing
        group = plan[2]
        workitems = cld(length(destination), IR._fill_group_size(group))
        _unweighted_sample_grouped_kernel!(backend)(
            rng,
            population,
            cardinality,
            destination,
            group;
            ndrange = workitems,
        )
        return destination
    end
    _unweighted_sample_kernel!(backend)(
        rng,
        population,
        cardinality,
        destination,
        width;
        ndrange = length(destination),
    )
    return destination
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
            @inbounds staged[lane] = Float64(IR._population_value(source, UInt64(ordinal)))
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

function IR._prepare_weight_scan(rng, weights, agnostic::Bool)
    source =
        agnostic ? IR._transfer_weights(rng.device, IR._collect_weights(weights)) : weights
    cumulative = IR._allocate_array(rng.device, Float64, (length(source),))
    total_result = IR._allocate_array(rng.device, Float64, (1,))
    invalid_result = IR._allocate_array(rng.device, Bool, (1,))
    backend = IR._fill_backend(rng.device, cumulative)
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
    only(Array(invalid_result)) && IR._invalid_weights()
    return nothing, total_result, cumulative
end

KernelAbstractions.@kernel function _weighted_threshold_kernel!(
    rng,
    total_result,
    thresholds,
)
    index = @index(Global, Linear)
    bits_lo, bits_hi = IR._bit_span(UInt64(index - 1), IR._WEIGHT_BITS)
    position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
    total = @inbounds total_result[1]
    @inbounds thresholds[index] = IR._weighted_threshold(rng, position, total)
end

@inline function IR._fill_weighted_thresholds!(backend, rng, total_result, thresholds)
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
        destination_index = IR._sampling_destination_index(indices, index)
        destination[destination_index] = IR._population_value(population, UInt64(lower))
    end
end

@inline function IR._launch_weighted_scan!(
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

end
