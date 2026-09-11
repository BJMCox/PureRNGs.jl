@inline function _fill_range_cpu_unchecked!(
    rng::_CPUGenerators,
    position,
    destination::AbstractArray{T},
    range::AbstractRange{T},
    span::UInt64,
    indices,
) where {T<:_RangeInteger}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    if _range_bits(span) == UInt16(64)
        @inbounds for index in indices
            candidate, cursor =
                _take_dense_bits_unchecked(rng, cursor, Val(64))
            offset = _reduce_range_candidate(candidate, span)
            destination[index] = _range_value(range, offset)
        end
    else
        @inbounds for index in indices
            hi, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
            lo, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
            offset = _reduce_range_candidate(lo, hi, span)
            destination[index] = _range_value(range, offset)
        end
    end
    return nothing
end

@inline _device_range_fill_plan(backend, rng, span) = nothing

@inline function _fill_range_grouped_unchecked!(
    rng,
    position,
    destination,
    range,
    span::UInt64,
    first::Int,
    ::Val{2},
)
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    last = length(destination)
    @inbounds for offset = 0:1
        index = first + offset
        index > last && break
        candidate, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        destination[index] = _range_value(range, _reduce_range_candidate(candidate, span))
    end
    return nothing
end


KernelAbstractions.@kernel function _range_fill_cpu_kernel!(
    rng,
    destination,
    range,
    span,
    chunk_elements,
)
    workitem = @index(Global, Linear)
    first, last = _dense_fill_bounds(workitem, length(destination), chunk_elements)
    width = _range_bits(span)
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), width)
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_range_cpu_unchecked!(rng, position, destination, range, span, first:last)
end

KernelAbstractions.@kernel function _range_fill_kernel!(rng, destination, range, span)
    index = @index(Global, Linear)
    width = _range_bits(span)
    bits_lo, bits_hi = _bit_span(UInt64(index - 1), width)
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = _draw_range_unchecked(rng, position, range, span)
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
    width = _range_bits(span)
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), width)
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_range_grouped_unchecked!(rng, position, destination, range, span, first, group)
end

@inline function _launch_range!(backend, rng, destination, range, span)
    plan = _device_range_fill_plan(backend, rng, span)
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
    workitems = cld(length(destination), _fill_group_size(group))
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


@inline function _launch_range!(
    backend::KernelAbstractions.CPU,
    rng::_CPUGenerators,
    destination::Array,
    range,
    span,
)
    chunk_elements = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_range_bits(span)))
    workitems = cld(length(destination), chunk_elements)
    if workitems < _CPU_FILL_MIN_WORKITEMS
        _fill_range_cpu_unchecked!(
            rng,
            rng.position,
            destination,
            range,
            span,
            eachindex(destination),
        )
        return destination
    end
    _range_fill_cpu_kernel!(backend)(
        rng,
        destination,
        range,
        span,
        chunk_elements;
        ndrange = workitems,
        workgroupsize = 1,
    )
    return destination
end

@inline function _rand_next_range_array(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dims::Tuple,
) where {T<:_RangeInteger}
    _check_serviceability(rng, range)
    span = _range_span(range)
    destination = _allocate_draw_array(rng.device, T, dims)
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _range_bits(span))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return destination, next_rng
    backend = _fill_backend(destination)
    _launch_range!(backend, rng, destination, range, span)
    return destination, next_rng
end

for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
    @eval begin
        @inline function Random.rand(
            rng::_ScalarUniformGenerators,
            range::AbstractRange{$T},
            dim1::Integer,
            dims::Integer...,
        )
            destination, _ = _rand_next_range_array(rng, range, (dim1, dims...))
            return destination
        end

        @inline function rand_next(
            rng::_ScalarUniformGenerators,
            range::AbstractRange{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return _rand_next_range_array(rng, range, (dim1, dims...))
        end
    end
end
