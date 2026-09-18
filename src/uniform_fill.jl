# `_run_chunks` counts draws from one. A linearly indexed destination may number
# its elements differently, so shift the chunk onto its own index range. A
# Cartesian destination takes the same ordinals through linear indexing.
@inline _chunk_indices(indices, from::Int, to::Int) = from:to
@inline function _chunk_indices(indices::AbstractUnitRange, from::Int, to::Int)
    offset = first(indices) - 1
    return (from+offset):(to+offset)
end

@inline function _fill_uniform_unchecked!(
    rng::_ScalarUniformGenerators,
    position,
    destination,
    ::Type{T},
    indices,
) where {T}
    width = UInt64(_draw_bits(T))
    shift = _block_shift(rng)
    remaining = length(indices)
    @inbounds for index in indices
        destination[index] = _draw_unchecked(rng, position, T)
        remaining -= 1
        iszero(remaining) ||
            (position = _advance_position_unchecked(position, width, UInt64(0), shift))
    end
    return nothing
end

# Fill-local cursor; public callers preflight the complete span before unchecked use.
struct _DenseBitCursor{B,L}
    block::B
    block_words::L
    lane::UInt16
    bit::UInt16
end

@inline function _dense_cursor(rng, block, bit::UInt16)
    block_words = _block_words_at(rng, block)
    return _DenseBitCursor(block, block_words, bit >> UInt16(6), bit & UInt16(63))
end

@inline function _ensure_dense_cursor(rng, cursor::_DenseBitCursor)
    cursor.lane < UInt16(length(cursor.block_words)) && return cursor
    block = _next_stream_block_unchecked(cursor.block)
    block_words = _block_words(rng, block)
    return _DenseBitCursor(block, block_words, UInt16(0), UInt16(0))
end

@inline function _take_dense_bits_unchecked(
    rng,
    cursor::_DenseBitCursor,
    ::Val{W},
) where {W}
    cursor = _ensure_dense_cursor(rng, cursor)
    width = UInt16(W)
    first = _select_tuple_value(cursor.block_words, cursor.lane)
    available = UInt16(64) - cursor.bit
    if width <= available
        value = (first >> (available - width)) & _low_mask(width)
        next_bit = cursor.bit + width
        next_lane = cursor.lane
        if next_bit == UInt16(64)
            next_bit = UInt16(0)
            next_lane += UInt16(1)
        end
        return value, _DenseBitCursor(cursor.block, cursor.block_words, next_lane, next_bit)
    end

    remaining = width - available
    second, block, block_words, lane =
        _next_block_word_unchecked(rng, cursor.block, cursor.block_words, cursor.lane)
    value =
        ((first & _low_mask(available)) << remaining) | (second >> (UInt16(64) - remaining))
    return value, _DenseBitCursor(block, block_words, lane, remaining)
end

@inline _dense_fill_group(::Type{Bool}) = 64
@inline _dense_fill_group(::Type{UInt32}) = 2
@inline _dense_fill_group(::Type{Int32}) = 2
@inline _dense_fill_group(::Type{Float32}) = 8
@inline _dense_fill_group(::Type{UInt64}) = 1
@inline _dense_fill_group(::Type{Int64}) = 1
@inline _dense_fill_group(::Type{Float64}) = 1

@inline _fill_group_size(::Val{N}) where {N} = N

@inline function _fill_grouped_cursor!(
    rng,
    position,
    destination,
    ::Type{T},
    first::Int,
    ::Val{N},
    width,
    codec,
) where {T,N}
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, width)
        destination[index] = _cooperative_value(codec, T, raw)
    end
    return nothing
end

@inline _uniform_cursor!(rng, position, destination, T, first, group) =
    _fill_grouped_cursor!(
        rng,
        position,
        destination,
        T,
        first,
        group,
        Val(_draw_bits(T)),
        Val(:uniform),
    )

@inline function _fill_uniform_grouped_unchecked!(
    rng::_ScalarUniform32Generators,
    position,
    destination,
    ::Type{Bool},
    first::Int,
    group::Val{N},
) where {N}
    word_bit = position.bit & UInt16(31)
    word_bit + UInt16(N) <= UInt16(32) ||
        return _uniform_cursor!(rng, position, destination, Bool, first, group)
    words = _block(rng, _position_block(position))
    word = _select_tuple_value(words, position.bit >> UInt16(5))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        destination[index] =
            !iszero((word >> (UInt16(31) - word_bit - UInt16(offset))) & UInt32(1))
    end
    return nothing
end

@inline _fill_uniform_grouped_unchecked!(rng, position, destination, T, first, group) =
    _uniform_cursor!(rng, position, destination, T, first, group)

@inline function _fill_uniform_grouped_unchecked!(
    rng,
    position,
    destination,
    ::Type{Bool},
    first::Int,
    group::Val{N},
) where {N}
    word_bit = position.bit & UInt16(63)
    word_bit + UInt16(N) <= UInt16(64) ||
        return _uniform_cursor!(rng, position, destination, Bool, first, group)
    block_words = _block_words(rng, _position_block(position))
    word = _select_tuple_value(block_words, position.bit >> UInt16(6))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        destination[index] = !iszero((word >> (UInt16(63) - word_bit - offset)) & 1)
    end
    return nothing
end

@inline function _fill_uniform_grouped_unchecked!(
    rng::_ScalarUniform32Generators,
    position,
    destination,
    ::Type{T},
    first::Int,
    group::Val{N},
) where {T<:_UniformInteger32,N}
    iszero(position.bit) && UInt16(32N) == _block_bits(rng) ||
        return _uniform_cursor!(rng, position, destination, T, first, group)
    words = _block(rng, _position_block(position))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        destination[index] =
            _from_bits(T, UInt64(_select_tuple_value(words, UInt16(offset))))
    end
    return nothing
end

@inline function _fill_uniform_grouped_unchecked!(
    rng,
    position,
    destination,
    ::Type{T},
    first::Int,
    group::Val{N},
) where {T<:_UniformInteger32,N}
    iszero(position.bit) && UInt16(32N) == _block_bits(rng) ||
        return _uniform_cursor!(rng, position, destination, T, first, group)
    block_words = _block_words(rng, _position_block(position))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        word = _select_tuple_value(block_words, UInt16(offset >> 1))
        raw = iseven(offset) ? word >> UInt16(32) : word
        destination[index] = _from_bits(T, raw)
    end
    return nothing
end

@inline function _fill_uniform_grouped_unchecked!(
    rng,
    position,
    destination,
    ::Type{T},
    first::Int,
    group::Val{N},
) where {T<:_UniformInteger64,N}
    iszero(position.bit) && UInt16(64N) == _block_bits(rng) ||
        return _uniform_cursor!(rng, position, destination, T, first, group)
    block_words = _block_words(rng, _position_block(position))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        destination[index] = _from_bits(T, _select_tuple_value(block_words, UInt16(offset)))
    end
    return nothing
end

@inline function _fill_dense_cursor!(
    rng,
    destination::AbstractArray{Bool},
    ::Type{Bool},
    index::Int,
    count::Int,
    cursor,
)
    last_index = index + count - 1
    while index + 63 <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        @inbounds for lane = 0:63
            destination[index+lane] = !iszero((raw >> (63 - lane)) & UInt64(1))
        end
        index += 64
    end
    @inbounds while index <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(1))
        destination[index] = _from_bits(Bool, raw)
        index += 1
    end
    return cursor
end

@inline function _fill_dense_cursor!(
    rng,
    destination::AbstractArray{T},
    ::Type{T},
    index::Int,
    count::Int,
    cursor,
) where {T<:_UniformInteger32}
    last_index = index + count - 1
    @inbounds while index + 1 <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        destination[index] = _from_bits(T, raw >> 32)
        destination[index+1] = _from_bits(T, raw)
        index += 2
    end
    @inbounds while index <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(32))
        destination[index] = _from_bits(T, raw)
        index += 1
    end
    return cursor
end

@inline function _fill_dense_cursor!(
    rng,
    destination::AbstractArray{Float32},
    ::Type{Float32},
    index::Int,
    count::Int,
    cursor,
)
    last_index = index + count - 1
    @inbounds while index + 7 <= last_index
        first, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        second, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        third, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        destination[index] = _from_bits(Float32, first >> 40)
        destination[index+1] = _from_bits(Float32, (first >> 16) & UInt64(0xffffff))
        destination[index+2] =
            _from_bits(Float32, ((first & UInt64(0xffff)) << 8) | (second >> 56))
        destination[index+3] = _from_bits(Float32, (second >> 32) & UInt64(0xffffff))
        destination[index+4] = _from_bits(Float32, (second >> 8) & UInt64(0xffffff))
        destination[index+5] =
            _from_bits(Float32, ((second & UInt64(0xff)) << 16) | (third >> 48))
        destination[index+6] = _from_bits(Float32, (third >> 24) & UInt64(0xffffff))
        destination[index+7] = _from_bits(Float32, third & UInt64(0xffffff))
        index += 8
    end
    @inbounds while index <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(24))
        destination[index] = _from_bits(Float32, raw)
        index += 1
    end
    return cursor
end

@inline function _fill_dense_cursor!(
    rng,
    destination::AbstractArray{T},
    ::Type{T},
    index::Int,
    count::Int,
    cursor,
) where {T<:_UniformInteger64}
    last_index = index + count - 1
    @inbounds while index <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        destination[index] = _from_bits(T, raw)
        index += 1
    end
    return cursor
end

# Fixed-width draws narrower than a word stream through a left-aligned bit
# buffer: `acc` holds `have` unread bits in its high end, and one block word is
# pulled whenever a draw needs more. This replaces a per-element lane search.
@inline function _fill_f64_bitbuffer!(rng, destination, index::Int, count::Int, cursor)
    width = Int(_draw_bits(Float64))
    last_index = index + count - 1
    cursor = _ensure_dense_cursor(rng, cursor)
    acc = _select_tuple_value(cursor.block_words, cursor.lane) << cursor.bit
    have = 64 - Int(cursor.bit)
    block, block_words, lane = cursor.block, cursor.block_words, cursor.lane
    @inbounds while index <= last_index
        if have >= width
            raw = acc >> (64 - width)
            acc <<= width
            have -= width
        else
            word, block, block_words, lane =
                _next_block_word_unchecked(rng, block, block_words, lane)
            short = width - have
            head = (acc >> (64 - width)) & (typemax(UInt64) << short)
            raw = head | (word >> (64 - short))
            acc = word << short
            have = 64 - short
        end
        destination[index] = _from_bits(Float64, raw)
        index += 1
    end
    # The cursor names the next unread word and bit. A drained word hands over
    # to the next lane, which `_ensure_dense_cursor` carries into the next block.
    drained = iszero(have)
    next_lane = lane + UInt16(drained)
    next_bit = drained ? UInt16(0) : UInt16(64 - have)
    return _DenseBitCursor(block, block_words, next_lane, next_bit)
end

@inline _fill_dense_cursor!(
    rng,
    destination::AbstractArray{Float64},
    ::Type{Float64},
    index::Int,
    count::Int,
    cursor,
) = _fill_f64_bitbuffer!(rng, destination, index, count, cursor)

# 53 Philox4x32 blocks hold exactly 128 Float64 draws (53 * 128 = 128 * 53 bits),
# so an aligned group extracts every value with compile-time shifts.
@inline function _blocks4_words(rng::Philox4x32, block::UInt64)
    a, b, c, d = _blocks4(rng, block)
    return (_block_words(a)..., _block_words(b)..., _block_words(c)..., _block_words(d)...)
end

# A group is 13 four-block chunks of 8 words and a final single block. Draw `i`
# covers bits `53i` to `53i + 52` of the group. The generated body decodes each
# chunk and stores the draws that end inside it, so every shift is a constant.
@generated function _store_f64_group!(destination, index, rng::Philox4x32, block::UInt64)
    chunk_of(word) = word >> 3
    word_ref(word) = :($(Symbol(:chunk_, chunk_of(word)))[$(word-8chunk_of(word)+1)])
    body = Expr[:(mask = _low_mask(UInt16(53)))]
    for chunk = 0:13
        source =
            chunk < 13 ? :(_blocks4_words(rng, block + UInt64($(4chunk)))) :
            :(_block_words(rng, block + UInt64(52)))
        push!(body, :($(Symbol(:chunk_, chunk)) = $source))
        for draw = 0:127
            offset = 53draw
            word, start = offset >> 6, offset & 63
            last_word = start <= 11 ? word : word + 1
            chunk_of(last_word) == chunk || continue
            raw = if start <= 11
                :(($(word_ref(word)) >> $(11 - start)) & mask)
            else
                head = :($(word_ref(word)) << $(start - 11))
                tail = :($(word_ref(word + 1)) >> $(75 - start))
                :(($head | $tail) & mask)
            end
            push!(body, :(@inbounds destination[index+$draw] = _from_bits(Float64, $raw)))
        end
    end
    push!(body, :(return nothing))
    return Expr(:block, body...)
end

@inline function _fill_dense_cursor!(
    rng::Philox4x32,
    destination::AbstractArray{Float64},
    ::Type{Float64},
    index::Int,
    count::Int,
    cursor,
)
    last_index = index + count - 1
    cursor = _ensure_dense_cursor(rng, cursor)
    # Draws until the stream is block aligned: 53k ≡ -offset (mod 128), and
    # 29 is the inverse of 53 modulo 128.
    offset = 64 * Int(cursor.lane) + Int(cursor.bit)
    prefix = min(((128 - offset) * 29) & 127, count)
    if prefix > 0
        cursor = _fill_f64_bitbuffer!(rng, destination, index, prefix, cursor)
        index += prefix
        cursor = _ensure_dense_cursor(rng, cursor)
    end
    block = cursor.block
    while index + 127 <= last_index && block <= _max_block(rng) - UInt64(52)
        _store_f64_group!(destination, index, rng, block)
        index += 128
        block += UInt64(53)
    end
    block == cursor.block || (cursor = _dense_cursor(rng, block, UInt16(0)))
    index > last_index && return cursor
    return _fill_f64_bitbuffer!(rng, destination, index, last_index - index + 1, cursor)
end

# The dense cursors walk `destination` by linear index, so the fast paths hold
# for every `IndexLinear` array, not only `Array`.
@inline _fill_uniform_dense_cpu!(
    rng,
    position,
    destination::AbstractArray{T},
    ::Type{T},
    indices,
) where {T} = _fill_uniform_dense_cpu!(
    IndexStyle(destination),
    rng,
    position,
    destination,
    T,
    indices,
)

@inline function _fill_uniform_dense_cpu!(
    ::IndexLinear,
    rng,
    position,
    destination::AbstractArray{T},
    ::Type{T},
    indices,
) where {T<:Union{Bool,_UniformInteger,Float32,Float64}}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    _fill_dense_cursor!(rng, destination, T, first(indices), length(indices), cursor)
    return nothing
end

@inline _fill_uniform_dense_cpu!(
    ::IndexStyle,
    rng,
    position,
    destination,
    ::Type{T},
    indices,
) where {T} = _fill_uniform_unchecked!(rng, position, destination, T, indices)

@inline function _store_bool_blocks4!(destination, index, blocks)
    @inbounds for block_lane = 1:4, word_lane = 1:4, bit_lane = 0:31
        word = blocks[block_lane][word_lane]
        offset = 128(block_lane - 1) + 32(word_lane - 1) + bit_lane
        destination[index+offset] = isodd(word >> (31 - bit_lane))
    end
    return nothing
end

@inline function _store_bits32_blocks4!(
    destination::AbstractArray{T},
    index,
    blocks,
) where {T}
    @inbounds for block_lane = 1:4, word_lane = 1:4
        destination[index+4(block_lane-1)+word_lane-1] =
            _from_bits(T, UInt64(blocks[block_lane][word_lane]))
    end
    return nothing
end

@inline function _store_bits64_blocks4!(
    destination::AbstractArray{T},
    index,
    blocks,
) where {T}
    @inbounds for block_lane = 1:4, word_lane = 1:2
        words = blocks[block_lane]
        raw = (UInt64(words[2word_lane-1]) << 32) | UInt64(words[2word_lane])
        destination[index+2(block_lane-1)+word_lane-1] = _from_bits(T, raw)
    end
    return nothing
end

@inline function _store_f32_triple!(destination, index, a, b, c)
    @inbounds begin
        destination[index] = _from_bits(Float32, UInt64(a >> 8))
        destination[index+1] =
            _from_bits(Float32, UInt64(((a & UInt32(0xff)) << 16) | (b >> 16)))
        destination[index+2] =
            _from_bits(Float32, UInt64(((b & UInt32(0xffff)) << 8) | (c >> 24)))
        destination[index+3] = _from_bits(Float32, UInt64(c & UInt32(0x00ffffff)))
    end
    return nothing
end

@inline function _store_f32_blocks3!(destination, index, words)
    @inbounds for group = 0:3
        word = 3group + 1
        _store_f32_triple!(
            destination,
            index + 4group,
            words[word],
            words[word+1],
            words[word+2],
        )
    end
    return nothing
end

@inline function _fill_aligned_blocks4!(
    rng,
    destination::AbstractArray{Bool},
    index,
    last,
    block,
)
    while index + 511 <= last && block <= _max_block(rng) - UInt64(3)
        blocks = _blocks4(rng, block)
        _store_bool_blocks4!(destination, index, blocks)
        index += 512
        block += UInt64(4)
    end
    return index, block
end
@inline function _fill_aligned_blocks4!(
    rng,
    destination::AbstractArray{T},
    index,
    last,
    block,
) where {T<:_UniformInteger32}
    while index + 15 <= last && block <= _max_block(rng) - UInt64(3)
        blocks = _blocks4(rng, block)
        _store_bits32_blocks4!(destination, index, blocks)
        index += 16
        block += UInt64(4)
    end
    return index, block
end
@inline function _fill_aligned_blocks4!(
    rng,
    destination::AbstractArray{T},
    index,
    last,
    block,
) where {T<:_UniformInteger64}
    while index + 7 <= last && block <= _max_block(rng) - UInt64(3)
        blocks = _blocks4(rng, block)
        _store_bits64_blocks4!(destination, index, blocks)
        index += 8
        block += UInt64(4)
    end
    return index, block
end
@inline function _fill_aligned_blocks4!(
    rng,
    destination::AbstractArray{Float32},
    index,
    last,
    block,
)
    while index + 63 <= last && block <= _max_block(rng) - UInt64(11)
        first = _blocks4(rng, block)
        second = _blocks4(rng, block + UInt64(4))
        third = _blocks4(rng, block + UInt64(8))
        _store_f32_blocks3!(destination, index, (first[1]..., first[2]..., first[3]...))
        _store_f32_blocks3!(
            destination,
            index + 16,
            (first[4]..., second[1]..., second[2]...),
        )
        _store_f32_blocks3!(
            destination,
            index + 32,
            (second[3]..., second[4]..., third[1]...),
        )
        _store_f32_blocks3!(
            destination,
            index + 48,
            (third[2]..., third[3]..., third[4]...),
        )
        index += 64
        block += UInt64(12)
    end
    return index, block
end

@inline function _fill_uniform_blocks4_cpu!(
    rng::Philox4x32,
    position::_Position64,
    destination::AbstractArray{T},
    ::Type{T},
    indices,
) where {T<:Union{Bool,_UniformInteger,Float32,Float64}}
    isempty(indices) && return nothing
    index = first(indices)
    last_index = last(indices)
    block = position.block
    bit = position.bit
    if iszero(bit)
        index, block = _fill_aligned_blocks4!(rng, destination, index, last_index, block)
    end
    index > last_index && return nothing
    cursor = _dense_cursor(rng, block, bit)
    _fill_dense_cursor!(rng, destination, T, index, last_index - index + 1, cursor)
    return nothing
end

# Float64 is excluded so the @generated group path above keeps the dispatch.
@inline _fill_uniform_dense_cpu!(
    ::IndexLinear,
    rng::Philox4x32,
    position::_Position64,
    destination::AbstractArray{T},
    ::Type{T},
    indices,
) where {T<:Union{Bool,_UniformInteger,Float32}} =
    _fill_uniform_blocks4_cpu!(rng, position, destination, T, indices)

@inline function _fill_uniform_dense_cpu!(
    rng,
    position,
    destination::BitArray,
    ::Type{Bool},
    indices,
)
    isempty(indices) && return nothing
    first_index = first(indices)
    iszero((first_index - 1) & 63) ||
        return _fill_uniform_unchecked!(rng, position, destination, Bool, indices)

    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    index = first_index
    last_index = last(indices)
    chunk = ((first_index - 1) >> 6) + 1
    @inbounds while index + 63 <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        # Stream bits are MSB-first; BitArray chunks store their first bit lowest.
        destination.chunks[chunk] = bitreverse(raw)
        index += 64
        chunk += 1
    end
    if index <= last_index
        raw = UInt64(0)
        offset = 0
        @inbounds while index <= last_index
            bit, cursor = _take_dense_bits_unchecked(rng, cursor, Val(1))
            raw |= bit << offset
            index += 1
            offset += 1
        end
        @inbounds destination.chunks[chunk] = raw
    end
    return nothing
end
