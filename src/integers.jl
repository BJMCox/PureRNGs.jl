const _RangeInteger = Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}

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
    stride = step(range) % UInt64
    span = length(range) % UInt64
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
