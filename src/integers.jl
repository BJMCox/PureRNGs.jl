const _RangeInteger = Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}

@inline _range_bits(span::UInt64) =
    span != zero(UInt64) && span <= UInt64(1) << 32 ? UInt16(64) : UInt16(128)

# This is the pinned K=64 reduction for spans through 2^32.
@inline function _mulhi_by_halves(word::UInt64, span::UInt64)
    high = word >> 32
    low = word & UInt64(0xffffffff)
    return (high * span + ((low * span) >> 32)) >> 32
end

# This is the pinned K=128 reduction for spans above 2^32.
@inline function _mulhi128_by64(lo::UInt64, hi::UInt64, span::UInt64)
    hi_high, hi_low = _mulhilo64(hi, span)
    lo_high = first(_mulhilo64(lo, span))
    sum = hi_low + lo_high
    return hi_high + UInt64(sum < hi_low)
end

@inline _reduce_range_candidate(candidate::UInt64, span::UInt64) =
    _mulhi_by_halves(candidate, span)
@inline _reduce_range_candidate(lo::UInt64, hi::UInt64, span::UInt64) =
    iszero(span) ? hi : _mulhi128_by64(lo, hi, span)

@noinline _empty_range_error() = throw(ArgumentError("range must be non-empty"))

@inline function _range_span(range::AbstractRange{T}) where {T<:_RangeInteger}
    isempty(range) && _empty_range_error()
    return length(range) % UInt64
end

@inline function _range_value(
    ::Type{T},
    base::UInt64,
    stride::UInt64,
    offset::UInt64,
) where {T<:_RangeInteger}
    bits = (base + offset * stride) % unsigned(T)
    return reinterpret(T, bits)
end

@inline function _range_value(
    range::OrdinalRange{T},
    offset::UInt64,
) where {T<:_RangeInteger}
    base = first(range) % UInt64
    stride = step(range) % UInt64
    return _range_value(T, base, stride, offset)
end

# Non-ordinal ranges use their own indexing rule, including its rounding.
@inline function _range_value(
    range::AbstractRange{T},
    offset::UInt64,
) where {T<:_RangeInteger}
    index = offset + UInt64(1)
    return iszero(index) ? last(range) : range[index]
end

@inline function _range_offset(rng::_ScalarUniformGenerators, position, span::UInt64)
    block = _position_block(position)
    if span != zero(UInt64) && span <= UInt64(1) << 32
        candidate = _extract_bits_unchecked(rng, block, position.bit, Val(64))
        return _reduce_range_candidate(candidate, span)
    end
    lo, hi = _extract_bits128_unchecked(rng, block, position.bit)
    return _reduce_range_candidate(lo, hi, span)
end

@inline _range_offset(rng::_ScalarUniformGenerators, span::UInt64) =
    _range_offset(rng, rng.position, span)

@inline function _draw_range_unchecked(
    rng::_ScalarUniformGenerators,
    position,
    range::AbstractRange{T},
    span::UInt64,
) where {T<:_RangeInteger}
    return _range_value(range, _range_offset(rng, position, span))
end

@inline function _draw_range_unchecked(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    span::UInt64,
) where {T<:_RangeInteger}
    return _draw_range_unchecked(rng, rng.position, range, span)
end

@inline function _rand_next_range(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
) where {T}
    span = _range_span(range)
    width = _range_bits(span)
    next_rng = _reserve_scalar(rng, width)
    if width == UInt16(64)
        candidate = _chain_bits(rng, next_rng, Val(64))
        return _range_value(range, _reduce_range_candidate(candidate, span)), next_rng
    end
    return _draw_range_unchecked(rng, range, span), next_rng
end

@inline _rand_range(rng::_ScalarUniformGenerators, range::AbstractRange{T}) where {T} =
    first(_rand_next_range(rng, range))

@inline Random.rand(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
) where {T<:_RangeInteger} = _rand_range(rng, range)
@inline rand_next(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
) where {T<:_RangeInteger} = _rand_next_range(rng, range)
