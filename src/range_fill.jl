# A span through 2^32 reduces one 64-bit candidate; a wider span reduces a high
# word followed by a low word. Unweighted sampling reduces the same way over the
# population cardinality.
@inline function _take_range_offset(rng, cursor, span::UInt64)
    if _range_bits(span) == UInt16(64)
        candidate, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        return _reduce_range_candidate(candidate, span), cursor
    end
    hi, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
    lo, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
    return _reduce_range_candidate(lo, hi, span), cursor
end

# The range codec draws a candidate of its own width and maps the reduced offset
# through the range, so the transformed scaffold serves range fills unchanged.
struct _RangeCodec{R}
    range::R
    span::UInt64
end

@inline _fill_width(codec::_RangeCodec, ::Type) = _range_bits(codec.span)

@inline _fill_chunk_elements(codec::_RangeCodec, ::Type) =
    Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_range_bits(codec.span)))

@inline function _codec_take(codec::_RangeCodec, rng, cursor, ::Type)
    offset, cursor = _take_range_offset(rng, cursor, codec.span)
    return _range_value(codec.range, offset), cursor
end

@inline _transformed_draw_unchecked(codec::_RangeCodec, rng, position, ::Type) =
    _draw_range_unchecked(rng, position, codec.range, codec.span)

@inline function _rand_next_range_fill!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    range::AbstractRange{T},
    threaded::Bool,
) where {T<:_RangeInteger}
    _check_fill_device(rng, destination)
    _check_serviceability(rng, range)
    codec = _RangeCodec(range, _range_span(range))
    return _fill_prevalidated!(rng, destination, threaded, codec)
end

@inline function _rand_next_range_array(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dims::Tuple,
    threaded::Bool,
) where {T<:_RangeInteger}
    _check_serviceability(rng, range)
    destination = _allocate_draw_array(rng.device, T, dims)
    codec = _RangeCodec(range, _range_span(range))
    return _fill_prevalidated!(rng, destination, threaded, codec)
end

@inline function Random.rand(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) where {T<:_RangeInteger}
    destination, _ = _rand_next_range_array(rng, range, (dim1, dims...), threaded)
    return destination
end
@inline Random.rand(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:_RangeInteger} = first(_rand_next_range_array(rng, range, dims, threaded))

@inline function rand_next(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) where {T<:_RangeInteger}
    return _rand_next_range_array(rng, range, (dim1, dims...), threaded)
end
@inline rand_next(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:_RangeInteger} = _rand_next_range_array(rng, range, dims, threaded)

@inline function Random.rand!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    range::AbstractRange{T};
    threaded::Bool = false,
) where {T<:_RangeInteger}
    return first(_rand_next_range_fill!(rng, destination, range, threaded))
end

@inline function rand_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    range::AbstractRange{T};
    threaded::Bool = false,
) where {T<:_RangeInteger}
    return _rand_next_range_fill!(rng, destination, range, threaded)
end
