const _RangeInteger = Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}
const FAMILY_RANGE = UInt32(0x00000003)

@inline _range_words(span::UInt64) =
    span != zero(UInt64) && span <= UInt64(1) << 32 ? UInt64(2) : UInt64(4)

# This is the pinned K=64 reduction for spans through 2^32.
@inline function _mulhi32limbs(word::UInt64, span::UInt64)
    high = word >> 32
    low = word & UInt64(0xffffffff)
    return (high * span + ((low * span) >> 32)) >> 32
end

# This is the pinned K=128 reduction for spans above 2^32.
@inline function _mulhi128_by64(high::UInt64, low::UInt64, span::UInt64)
    high_high = first(_mulhilo64(high, span))
    high_low = high * span
    low_high = first(_mulhilo64(low, span))
    sum = high_low + low_high
    return high_high + UInt64(sum < high_low)
end

@inline function _range_parameters(range::AbstractRange{T}) where {T<:_RangeInteger}
    isempty(range) && throw(ArgumentError("range must be non-empty"))
    base = first(range) % UInt64
    span = length(range) % UInt64
    stride = span == UInt64(1) ? UInt64(0) : range[2] % UInt64 - base
    return base, stride, span
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

@inline function _range_offset(rng::_ScalarUniform32Family, span::UInt64)
    if span == zero(UInt64)
        return _raw64(rng, FAMILY_RANGE)
    elseif span <= UInt64(1) << 32
        return _mulhi32limbs(_raw64(rng, FAMILY_RANGE), span)
    end
    high, low = _raw128(rng, FAMILY_RANGE)
    return _mulhi128_by64(high, low, span)
end

@inline function _draw_range_unchecked(
    rng::_ScalarUniform32Family,
    ::Type{T},
    base::UInt64,
    stride::UInt64,
    span::UInt64,
) where {T<:_RangeInteger}
    return _range_value(T, base, stride, _range_offset(rng, span))
end

@inline function _rand_range(rng::_ScalarUniform32Family, range::AbstractRange{T}) where {T}
    base, stride, span = _range_parameters(range)
    words = _range_words(span)
    start, _ = _reserve_aligned(rng, words, words)
    return _draw_range_unchecked(start, T, base, stride, span)
end

@inline function _rand_next_range(
    rng::_ScalarUniform32Family,
    range::AbstractRange{T},
) where {T}
    base, stride, span = _range_parameters(range)
    words = _range_words(span)
    start, next_rng = _reserve_aligned(rng, words, words)
    return next_rng, _draw_range_unchecked(start, T, base, stride, span)
end

for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
    @eval begin
        @inline Random.rand(rng::_ScalarUniform32Family, range::AbstractRange{$T}) =
            _rand_range(rng, range)
        @inline rand_next(rng::_ScalarUniform32Family, range::AbstractRange{$T}) =
            _rand_next_range(rng, range)
    end
end
