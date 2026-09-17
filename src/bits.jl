# A draw counter is the block address zero-extended to the core's counter width.
# Key derivation in derive.jl writes a nonzero tag into the extension words, so
# derived keys never coincide with a draw address. The two-word 32-bit generators
# reserve the top byte of the second word for that tag, which is why their block
# counter is 56 bits wide. The 64-bit two-word cores use a whole tag word instead.
@inline _draw_counter(::Val{N}, address::NTuple{K,T}) where {N,K,T} =
    (address..., ntuple(_ -> zero(T), Val(N - K))...)
@inline _address32(block::UInt64) = (block % UInt32, (block >> 32) % UInt32)
@inline _address32_narrow(block::UInt64) =
    (block % UInt32, ((block >> 32) & 0x00ffffff) % UInt32)

# `_core_block` runs the core for one block address. It takes the key rather than
# a generator so that a generator can decode its own block while it is built.
@inline _core_block(::Type{F}, key, block::UInt64) where {F<:Philox2x32} =
    _philox2x32(_address32_narrow(block), key, Val(_rounds(F)))
@inline _core_block(::Type{F}, key, block::UInt64) where {F<:Threefry2x32} =
    _threefry2x32(_address32_narrow(block), key, Val(_rounds(F)))
@inline _core_block(::Type{F}, key, block::UInt64) where {F<:Philox4x32} =
    _philox4x32(_draw_counter(Val(4), _address32(block)), key, Val(_rounds(F)))
@inline _core_block(::Type{F}, key, block::UInt64) where {F<:Threefry4x32} =
    _threefry4x32(_draw_counter(Val(4), _address32(block)), key, Val(_rounds(F)))
@inline _core_block(::Type{F}, key, block::UInt64) where {F<:Philox2x64} =
    _philox2x64(_draw_counter(Val(2), (block,)), key, Val(_rounds(F)))
@inline _core_block(::Type{F}, key, block::UInt64) where {F<:Threefry2x64} =
    _threefry2x64(_draw_counter(Val(2), (block,)), key, Val(_rounds(F)))
@inline _core_block(::Type{F}, key, block::NTuple{2,UInt64}) where {F<:Philox4x64} =
    _philox4x64(_draw_counter(Val(4), block), key, Val(_rounds(F)))
@inline _core_block(::Type{F}, key, block::NTuple{2,UInt64}) where {F<:Threefry4x64} =
    _threefry4x64(_draw_counter(Val(4), block), key, Val(_rounds(F)))
@inline _core_block(::Type{F}, key, block::UInt64) where {F<:ChaCha} =
    _chacha(_draw_counter(Val(4), _address32(block)), key, Val(_rounds(F)))

# CPU-bound 64-bit Philox runs through the host word operations.
@inline function _core_block(
    ::Type{F},
    key,
    block::UInt64,
) where {F<:Philox2x64{_CPUBackend}}
    counter = _host_words(_draw_counter(Val(2), (block,)), Val(2))
    return _unwrap_words(_philox2x64(counter, _host_words(key, Val(2)), Val(_rounds(F))))
end
@inline function _core_block(
    ::Type{F},
    key,
    block::NTuple{2,UInt64},
) where {F<:Philox4x64{_CPUBackend}}
    counter = _host_words(_draw_counter(Val(4), block), Val(4))
    return _unwrap_words(_philox4x64(counter, _host_words(key, Val(4)), Val(_rounds(F))))
end

# `_block` is the bulk-codec seam; word extraction below is for scalar/peel/tail work.
@inline _block(rng::_Position64Generators, block::UInt64) =
    _core_block(typeof(rng), rng.key, block)
@inline _block(rng::_Position128Generators, block_lo::UInt64, block_hi::UInt64) =
    _core_block(typeof(rng), rng.key, (block_lo, block_hi))

@inline function _blocks4(rng::Philox4x32, block::UInt64)
    counter(offset) = _draw_counter(Val(4), _address32(block + UInt64(offset)))
    rounds = Val(_rounds(typeof(rng)))
    return _philox4x32_blocks4(
        counter(0),
        counter(1),
        counter(2),
        counter(3),
        rng.key,
        rounds,
    )
end

# The block as 64-bit words, packing two 32-bit output words per block word.
@inline _block_words(output::NTuple{2,UInt32}) =
    ((UInt64(output[1]) << 32) | UInt64(output[2]),)
@inline _block_words(output::NTuple{4,UInt32}) = (
    (UInt64(output[1]) << 32) | UInt64(output[2]),
    (UInt64(output[3]) << 32) | UInt64(output[4]),
)
@inline _block_words(output::NTuple{16,UInt32}) =
    ntuple(i -> (UInt64(output[2i-1]) << 32) | UInt64(output[2i]), Val(8))
@inline _block_words(output::NTuple{N,UInt64}) where {N} = output

@inline _block_words(rng::_Position64Generators, block::UInt64) =
    _block_words(_block(rng, block))
@inline _block_words(rng::_Position128Generators, block::NTuple{2,UInt64}) =
    _block_words(_block(rng, block...))

# The block words a generator carries are the decoded block at its own position.
@inline _decoded_block_words(::Type{F}, key, position) where {F} =
    _block_words(_core_block(F, key, _position_block(position)))

# Reuse the carried block when the address matches, otherwise run the core.
@inline _block_words_at(rng::AbstractPureRNG, block) =
    block == _position_block(rng.position) ? rng.block_words : _block_words(rng, block)

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

@inline function _next_block_word_unchecked(
    rng,
    block,
    block_words::NTuple{N,UInt64},
    lane::UInt16,
) where {N}
    next_lane = lane + UInt16(1)
    if next_lane < UInt16(N)
        return _select_tuple_value(block_words, next_lane), block, block_words, next_lane
    end
    next_block = _next_stream_block_unchecked(block)
    next_block_words = _block_words(rng, next_block)
    return next_block_words[1], next_block, next_block_words, UInt16(0)
end

# Caller guarantees W in 1:64, a valid bit offset, and full-span preflight.
@inline function _extract_bits_unchecked(rng, block, bit::UInt16, ::Val{W}) where {W}
    width = UInt16(W)
    block_words = _block_words_at(rng, block)
    lane = bit >> UInt16(6)
    word_bit = bit & UInt16(63)
    first = _select_tuple_value(block_words, lane)
    available = UInt16(64) - word_bit

    if width <= available
        return (first >> (available - width)) & _low_mask(width)
    end

    remaining = width - available
    second, _, _, _ = _next_block_word_unchecked(rng, block, block_words, lane)
    return ((first & _low_mask(available)) << remaining) |
           (second >> (UInt16(64) - remaining))
end

# Bits for a scalar draw at the generator's position whose successor is known.
# When the draw straddles two blocks, the successor carries the second block.
@inline function _chain_bits(rng::AbstractPureRNG, next_rng, ::Val{W}) where {W}
    width = UInt16(W)
    block_words = rng.block_words
    bit = rng.position.bit
    lane = bit >> UInt16(6)
    available = UInt16(64) - (bit & UInt16(63))
    head = _select_tuple_value(block_words, lane)
    width <= available && return (head >> (available - width)) & _low_mask(width)

    remaining = width - available
    next_lane = lane + UInt16(1)
    tail = if next_lane < UInt16(length(block_words))
        _select_tuple_value(block_words, next_lane)
    else
        next_rng.block_words[1]
    end
    return ((head & _low_mask(available)) << remaining) |
           (tail >> (UInt16(64) - remaining))
end

@inline function _extract_bits128_unchecked(rng, block, bit::UInt16)
    block_words = _block_words_at(rng, block)
    lane = bit >> UInt16(6)
    word_bit = bit & UInt16(63)
    first = _select_tuple_value(block_words, lane)
    second, block, block_words, lane =
        _next_block_word_unchecked(rng, block, block_words, lane)
    iszero(word_bit) && return second, first

    third, _, _, _ = _next_block_word_unchecked(rng, block, block_words, lane)
    inverse = UInt16(64) - word_bit
    hi = (first << word_bit) | (second >> inverse)
    lo = (second << word_bit) | (third >> inverse)
    return lo, hi
end
