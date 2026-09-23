const _RangeInteger64 = Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}
const _RangeInteger = Union{_RangeInteger64,_WideInteger}

# The candidate is 64 bits wider than the span it reduces, so every outcome's
# probability is off by at most 2^-64 relative (2^-32 for spans through 2^32).
# A span that fills its type wraps to zero and takes the raw candidate.
@inline _range_bits(span::UInt64) =
    span != zero(UInt64) && span <= UInt64(1) << 32 ? UInt16(64) : UInt16(128)
@inline _range_bits(span::UInt128) =
    _narrow_span(span) ? _range_bits(span % UInt64) : UInt16(192)

# A 128-bit span through 2^64 reduces exactly as a 64-bit span does.
@inline _narrow_span(span::UInt128) = span != zero(UInt128) && span <= UInt128(1) << 64

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
@inline _reduce_range_candidate(candidate::UInt64, span::UInt128) =
    UInt128(_mulhi_by_halves(candidate, span % UInt64))
@inline _reduce_range_candidate(lo::UInt64, hi::UInt64, span::UInt64) =
    iszero(span) ? hi : _mulhi128_by64(lo, hi, span)

# The top 128 bits of the 192-bit candidate `w2:w1:w0` times the 128-bit span,
# accumulated column by column from 64-bit partial products.
@inline function _mulhi192_by128(w2::UInt64, w1::UInt64, w0::UInt64, span::UInt128)
    s0 = span % UInt64
    s1 = (span >> 64) % UInt64
    product(a, b) = ((hi, lo) = _mulhilo64(a, b); (UInt128(hi), UInt128(lo)))
    h00, _ = product(w0, s0)
    h01, l01 = product(w0, s1)
    h10, l10 = product(w1, s0)
    h11, l11 = product(w1, s1)
    h20, l20 = product(w2, s0)
    h21, l21 = product(w2, s1)
    column1 = h00 + l01 + l10
    column2 = (column1 >> 64) + h01 + h10 + l11 + l20
    column3 = (column2 >> 64) + h11 + h20 + l21
    column4 = (column3 >> 64) + h21
    return (column4 << 64) | (column3 & UInt128(typemax(UInt64)))
end

@inline _reduce_range_candidate(w2::UInt64, w1::UInt64, w0::UInt64, span::UInt128) =
    iszero(span) ? _wide_from_words(UInt128, w2, w1) : _mulhi192_by128(w2, w1, w0, span)

@noinline _empty_range_error() = throw(ArgumentError("range must be non-empty"))

@inline function _range_span(range::AbstractRange{T}) where {T<:_RangeInteger64}
    isempty(range) && _empty_range_error()
    return length(range) % UInt64
end

@inline function _range_span(range::AbstractRange{T}) where {T<:_WideInteger}
    isempty(range) && _empty_range_error()
    return length(range) % UInt128
end

@inline function _range_value(
    ::Type{T},
    base::UInt64,
    stride::UInt64,
    offset::UInt64,
) where {T<:_RangeInteger64}
    bits = (base + offset * stride) % unsigned(T)
    return reinterpret(T, bits)
end

@inline function _range_value(
    range::OrdinalRange{T},
    offset::UInt64,
) where {T<:_RangeInteger64}
    base = first(range) % UInt64
    stride = step(range) % UInt64
    return _range_value(T, base, stride, offset)
end

@inline function _range_value(
    range::OrdinalRange{T},
    offset::UInt128,
) where {T<:_WideInteger}
    bits = first(range) % UInt128 + offset * (step(range) % UInt128)
    return reinterpret(T, bits)
end

# Non-ordinal ranges use their own indexing rule, including its rounding.
@inline function _range_value(
    range::AbstractRange{T},
    offset::Union{UInt64,UInt128},
) where {T<:_RangeInteger}
    index = offset + one(offset)
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

@inline function _range_offset(rng::_ScalarUniformGenerators, position, span::UInt128)
    _narrow_span(span) && return UInt128(_range_offset(rng, position, span % UInt64))
    block = _position_block(position)
    w1, w2 = _extract_bits128_unchecked(rng, block, position.bit)
    third = _advance_position_unchecked(position, UInt64(128), UInt64(0), _block_shift(rng))
    w0 = _extract_bits_unchecked(rng, _position_block(third), third.bit, Val(64))
    return _reduce_range_candidate(w2, w1, w0, span)
end

@inline function _draw_range_unchecked(
    rng::_ScalarUniformGenerators,
    position,
    range::AbstractRange{T},
    span::Union{UInt64,UInt128},
) where {T<:_RangeInteger}
    return _range_value(range, _range_offset(rng, position, span))
end

@inline function _draw_range_unchecked(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    span::Union{UInt64,UInt128},
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

@inline function rand_at(
    rng::_ScalarUniformGenerators,
    range::AbstractRange{T},
    i::Integer,
) where {T<:_RangeInteger}
    span = _range_span(range)
    addressed = _addressed_rng(rng, _range_bits(span), i)
    return _draw_range_unchecked(addressed, range, span)
end
