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
            candidate, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
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
    ::KernelAbstractions.CPU,
    rng::_CPUGenerators,
    destination::AbstractArray,
    range,
    span,
)
    width = _range_bits(span)
    chunk_elements = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(width))
    indices = eachindex(destination)
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), width)
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        chunk = _chunk_indices(indices, first, last)
        _fill_range_cpu_unchecked!(rng, position, destination, range, span, chunk)
    end
    return destination
end

@inline function _rand_next_range_fill!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    range::AbstractRange{T},
    threaded::Bool,
) where {T<:_RangeInteger}
    _check_fill_device(rng, destination)
    _check_serviceability(rng, range)
    span = _range_span(range)
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _range_bits(span))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_range_cpu_unchecked!(
            rng,
            rng.position,
            destination,
            range,
            span,
            eachindex(destination),
        )
        return destination, next_rng
    end
    _launch_range!(_fill_backend(destination), rng, destination, range, span)
    return destination, next_rng
end

@inline function _rand_next_range_array(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dims::Tuple,
) where {T<:_RangeInteger}
    _check_serviceability(rng, range)
    destination = _allocate_draw_array(rng.device, T, dims)
    return _rand_next_range_fill!(rng, destination, range, true)
end

@inline function Random.rand(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dim1::Integer,
    dims::Integer...,
) where {T<:_RangeInteger}
    destination, _ = _rand_next_range_array(rng, range, (dim1, dims...))
    return destination
end
@inline Random.rand(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dims::Dims,
) where {T<:_RangeInteger} = first(_rand_next_range_array(rng, range, dims))

@inline function rand_next(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dim1::Integer,
    dims::Integer...,
) where {T<:_RangeInteger}
    return _rand_next_range_array(rng, range, (dim1, dims...))
end
@inline rand_next(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dims::Dims,
) where {T<:_RangeInteger} = _rand_next_range_array(rng, range, dims)

@inline function Random.rand!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    range::AbstractRange{T};
    threaded = true,
) where {T<:_RangeInteger}
    return first(_rand_next_range_fill!(rng, destination, range, _check_threaded(threaded)))
end

@inline function rand_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    range::AbstractRange{T};
    threaded = true,
) where {T<:_RangeInteger}
    return _rand_next_range_fill!(rng, destination, range, _check_threaded(threaded))
end
