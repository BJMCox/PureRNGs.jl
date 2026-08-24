const FAMILY_BITS = UInt32(0x00000000)

# `_block` is the bulk-codec seam; limb extraction below is for scalar/peel/tail work.
@inline _block(rng::Philox2x32, family::UInt32, block::UInt64) = _philox2x32(
    (block % UInt32, (family << 24) | (((block >> 32) & 0x00ffffff) % UInt32)),
    rng.key,
)

@inline _block(rng::Threefry2x32, family::UInt32, block::UInt64) = _threefry2x32(
    (block % UInt32, (family << 24) | (((block >> 32) & 0x00ffffff) % UInt32)),
    rng.key,
)

@inline _block(rng::Philox4x32, family::UInt32, block::UInt64) =
    _philox4x32((block % UInt32, (block >> 32) % UInt32, family, UInt32(0)), rng.key)

@inline function _blocks4(rng::Philox4x32, family::UInt32, block::UInt64)
    high = (block >> 32) % UInt32
    a = (block % UInt32, high, family, UInt32(0))
    block += UInt64(1)
    high += UInt32(iszero(block % UInt32))
    b = (block % UInt32, high, family, UInt32(0))
    block += UInt64(1)
    high += UInt32(iszero(block % UInt32))
    c = (block % UInt32, high, family, UInt32(0))
    block += UInt64(1)
    high += UInt32(iszero(block % UInt32))
    d = (block % UInt32, high, family, UInt32(0))
    return _philox4x32_blocks4(a, b, c, d, rng.key)
end

@inline _block(rng::Threefry4x32, family::UInt32, block::UInt64) =
    _threefry4x32((block % UInt32, (block >> 32) % UInt32, family, UInt32(0)), rng.key)

@inline _block(rng::Philox2x64, family::UInt32, block::UInt64) =
    _philox2x64((block, UInt64(family)), rng.key)

@inline _block(rng::Threefry2x64, family::UInt32, block::UInt64) =
    _threefry2x64((block, UInt64(family)), rng.key)

@inline _block(rng::Philox4x64, family::UInt32, block_lo::UInt64, block_hi::UInt64) =
    _philox4x64((block_lo, block_hi, UInt64(family), UInt64(0)), rng.key)

@inline _block(rng::Threefry4x64, family::UInt32, block_lo::UInt64, block_hi::UInt64) =
    _threefry4x64((block_lo, block_hi, UInt64(family), UInt64(0)), rng.key)

@inline _stream_limbs(words::NTuple{2,UInt32}) =
    ((UInt64(words[1]) << 32) | UInt64(words[2]),)
@inline _stream_limbs(words::NTuple{4,UInt32}) = (
    (UInt64(words[1]) << 32) | UInt64(words[2]),
    (UInt64(words[3]) << 32) | UInt64(words[4]),
)
@inline _stream_limbs(words::NTuple{N,UInt64}) where {N} = words

@inline _stream_limbs(rng::_Position64Family, family::UInt32, block::UInt64) =
    _stream_limbs(_block(rng, family, block))
@inline _stream_limbs(rng::_Position128Family, family::UInt32, block::NTuple{2,UInt64}) =
    _stream_limbs(_block(rng, family, block...))

@inline _next_stream_block_unchecked(block::UInt64) = block + UInt64(1)
@inline function _next_stream_block_unchecked(block::NTuple{2,UInt64})
    lo = block[1] + UInt64(1)
    return lo, block[2] + UInt64(iszero(lo))
end

@inline function _select_tuple_value(values::Tuple{T,Vararg{T}}, lane::UInt16) where {T}
    value = values[1]
    for index = 2:length(values)
        value = ifelse(lane == index - 1, values[index], value)
    end
    return value
end

@inline function _local_dense_bits(storage, bit::Int, ::Val{W}) where {W}
    lane = (bit >> 6) + 1
    word_bit = bit & 63
    first = @inbounds storage[lane]
    available = 64 - word_bit
    if W <= available
        return (first >> (available - W)) & _low_mask(UInt16(W))
    end
    remaining = W - available
    second = @inbounds storage[lane+1]
    return ((first & _low_mask(UInt16(available))) << remaining) |
           (second >> (64 - remaining))
end

@inline _low_mask(width::UInt16) = typemax(UInt64) >> (UInt16(64) - width)

@inline function _next_stream_limb_unchecked(
    rng,
    family::UInt32,
    block,
    limbs::NTuple{N,UInt64},
    lane::UInt16,
) where {N}
    next_lane = lane + UInt16(1)
    if next_lane < UInt16(N)
        return _select_tuple_value(limbs, next_lane), block, limbs, next_lane
    end
    next_block = _next_stream_block_unchecked(block)
    next_limbs = _stream_limbs(rng, family, next_block)
    return next_limbs[1], next_block, next_limbs, UInt16(0)
end

# Caller guarantees W in 1:64, a valid bit offset, and full-span preflight.
@inline function _extract_bits_unchecked(
    rng,
    family::UInt32,
    block,
    bit::UInt16,
    ::Val{W},
) where {W}
    width = UInt16(W)
    limbs = _stream_limbs(rng, family, block)
    lane = bit >> UInt16(6)
    word_bit = bit & UInt16(63)
    first = _select_tuple_value(limbs, lane)
    available = UInt16(64) - word_bit

    if width <= available
        return (first >> (available - width)) & _low_mask(width)
    end

    remaining = width - available
    second, _, _, _ = _next_stream_limb_unchecked(rng, family, block, limbs, lane)
    return ((first & _low_mask(available)) << remaining) |
           (second >> (UInt16(64) - remaining))
end

@inline function _extract_bits128_unchecked(rng, family::UInt32, block, bit::UInt16)
    limbs = _stream_limbs(rng, family, block)
    lane = bit >> UInt16(6)
    word_bit = bit & UInt16(63)
    first = _select_tuple_value(limbs, lane)
    second, block, limbs, lane =
        _next_stream_limb_unchecked(rng, family, block, limbs, lane)
    iszero(word_bit) && return second, first

    third, _, _, _ = _next_stream_limb_unchecked(rng, family, block, limbs, lane)
    inverse = UInt16(64) - word_bit
    hi = (first << word_bit) | (second >> inverse)
    lo = (second << word_bit) | (third >> inverse)
    return lo, hi
end
