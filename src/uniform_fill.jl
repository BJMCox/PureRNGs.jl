@inline function _fill_uniform_unchecked!(
    rng::_ScalarUniformFamily,
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
    limbs::L
    lane::UInt16
    bit::UInt16
end

@inline function _dense_cursor(rng, family::UInt32, block, bit::UInt16)
    limbs = _stream_limbs(rng, family, block)
    return _DenseBitCursor(block, limbs, bit >> UInt16(6), bit & UInt16(63))
end

@inline function _ensure_dense_cursor(rng, family::UInt32, cursor::_DenseBitCursor)
    cursor.lane < UInt16(length(cursor.limbs)) && return cursor
    block = _next_stream_block_unchecked(cursor.block)
    limbs = _stream_limbs(rng, family, block)
    return _DenseBitCursor(block, limbs, UInt16(0), UInt16(0))
end

@inline function _take_dense_bits_unchecked(
    rng,
    family::UInt32,
    cursor::_DenseBitCursor,
    ::Val{W},
) where {W}
    cursor = _ensure_dense_cursor(rng, family, cursor)
    width = UInt16(W)
    first = _select_tuple_value(cursor.limbs, cursor.lane)
    available = UInt16(64) - cursor.bit
    if width <= available
        value = (first >> (available - width)) & _low_mask(width)
        next_bit = cursor.bit + width
        next_lane = cursor.lane
        if next_bit == UInt16(64)
            next_bit = UInt16(0)
            next_lane += UInt16(1)
        end
        return value, _DenseBitCursor(cursor.block, cursor.limbs, next_lane, next_bit)
    end

    remaining = width - available
    second, block, limbs, lane =
        _next_stream_limb_unchecked(rng, family, cursor.block, cursor.limbs, cursor.lane)
    value =
        ((first & _low_mask(available)) << remaining) | (second >> (UInt16(64) - remaining))
    return value, _DenseBitCursor(block, limbs, lane, remaining)
end

@inline _dense_fill_group(::Type{Bool}) = 64
@inline _dense_fill_group(::Type{UInt32}) = 2
@inline _dense_fill_group(::Type{Float32}) = 8
@inline _dense_fill_group(::Type{UInt64}) = 1
@inline _dense_fill_group(::Type{Float64}) = 1

@inline _fill_group_size(::Val{N}) where {N} = N

@inline _device_uniform_fill_group(rng, ::Type{Bool}) = Val(4)
@inline _device_uniform_fill_group(rng::_NarrowFamily, ::Type{UInt32}) = Val(2)
@inline _device_uniform_fill_group(rng::_Position64Family, ::Type{UInt32}) = Val(4)
@inline _device_uniform_fill_group(rng::_Position128Family, ::Type{UInt32}) = Val(8)
@inline _device_uniform_fill_group(rng::_NarrowFamily, ::Type{UInt64}) = Val(1)
@inline _device_uniform_fill_group(rng::_Position64Family, ::Type{UInt64}) = Val(2)
@inline _device_uniform_fill_group(rng::_Position128Family, ::Type{UInt64}) = Val(4)
@inline _device_uniform_fill_group(rng, ::Type{Float32}) = Val(4)
@inline _device_uniform_fill_group(rng, ::Type{Float64}) = Val(4)

@inline _cooperative_uniform_fill(::Philox4x32, ::Type{Bool}) = (Val(4096), Val(32))
@inline _cooperative_uniform_fill(::Philox4x32, ::Type{Float32}) = (Val(2048), Val(32))
@inline _cooperative_uniform_fill(::Philox4x32, ::Type{Float64}) = (Val(1024), Val(64))
@inline _cooperative_uniform_fill(rng, T) = nothing
@inline _device_uniform_fill_plan(backend, rng, T) = nothing

@inline function _fill_grouped_cursor!(
    rng,
    position,
    destination,
    ::Type{T},
    first::Int,
    ::Val{N},
    width,
    family::UInt32,
    codec,
) where {T,N}
    cursor = _dense_cursor(rng, family, _position_block(position), position.bit)
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        raw, cursor = _take_dense_bits_unchecked(rng, family, cursor, width)
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
        FAMILY_BITS,
        Val(:uniform),
    )

@inline function _fill_uniform_grouped_unchecked!(
    rng::_ScalarUniform32Family,
    position,
    destination,
    ::Type{Bool},
    first::Int,
    group::Val{N},
) where {N}
    word_bit = position.bit & UInt16(31)
    word_bit + UInt16(N) <= UInt16(32) ||
        return _uniform_cursor!(rng, position, destination, Bool, first, group)
    words = _block(rng, FAMILY_BITS, _position_block(position))
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
    limbs = _stream_limbs(rng, FAMILY_BITS, _position_block(position))
    word = _select_tuple_value(limbs, position.bit >> UInt16(6))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        destination[index] = !iszero((word >> (UInt16(63) - word_bit - offset)) & 1)
    end
    return nothing
end

@inline function _fill_uniform_grouped_unchecked!(
    rng::_ScalarUniform32Family,
    position,
    destination,
    ::Type{UInt32},
    first::Int,
    group::Val{N},
) where {N}
    iszero(position.bit) && UInt16(32N) == _block_bits(rng) ||
        return _uniform_cursor!(rng, position, destination, UInt32, first, group)
    words = _block(rng, FAMILY_BITS, _position_block(position))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        destination[index] = _select_tuple_value(words, UInt16(offset))
    end
    return nothing
end

@inline function _fill_uniform_grouped_unchecked!(
    rng,
    position,
    destination,
    ::Type{UInt32},
    first::Int,
    group::Val{N},
) where {N}
    iszero(position.bit) && UInt16(32N) == _block_bits(rng) ||
        return _uniform_cursor!(rng, position, destination, UInt32, first, group)
    limbs = _stream_limbs(rng, FAMILY_BITS, _position_block(position))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        word = _select_tuple_value(limbs, UInt16(offset >> 1))
        destination[index] = iseven(offset) ? (word >> UInt16(32)) % UInt32 : word % UInt32
    end
    return nothing
end

@inline function _fill_uniform_grouped_unchecked!(
    rng,
    position,
    destination,
    ::Type{UInt64},
    first::Int,
    group::Val{N},
) where {N}
    iszero(position.bit) && UInt16(64N) == _block_bits(rng) ||
        return _uniform_cursor!(rng, position, destination, UInt64, first, group)
    limbs = _stream_limbs(rng, FAMILY_BITS, _position_block(position))
    last = length(destination)
    @inbounds for offset = 0:(N-1)
        index = first + offset
        index > last && break
        destination[index] = _select_tuple_value(limbs, UInt16(offset))
    end
    return nothing
end

@inline function _fill_dense_cursor!(
    rng,
    destination::Array{Bool},
    ::Type{Bool},
    index::Int,
    count::Int,
    cursor,
)
    last_index = index + count - 1
    while index + 63 <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(64))
        @inbounds for lane = 0:63
            destination[index+lane] = !iszero((raw >> (63 - lane)) & UInt64(1))
        end
        index += 64
    end
    @inbounds while index <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(1))
        destination[index] = _from_bits(Bool, raw)
        index += 1
    end
    return cursor
end

@inline function _fill_dense_cursor!(
    rng,
    destination::Array{UInt32},
    ::Type{UInt32},
    index::Int,
    count::Int,
    cursor,
)
    last_index = index + count - 1
    @inbounds while index + 1 <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(64))
        destination[index] = _from_bits(UInt32, raw >> 32)
        destination[index+1] = _from_bits(UInt32, raw)
        index += 2
    end
    @inbounds while index <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(32))
        destination[index] = _from_bits(UInt32, raw)
        index += 1
    end
    return cursor
end

@inline function _fill_dense_cursor!(
    rng,
    destination::Array{Float32},
    ::Type{Float32},
    index::Int,
    count::Int,
    cursor,
)
    last_index = index + count - 1
    @inbounds while index + 7 <= last_index
        first, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(64))
        second, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(64))
        third, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(64))
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
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(24))
        destination[index] = _from_bits(Float32, raw)
        index += 1
    end
    return cursor
end

@inline function _fill_dense_cursor!(
    rng,
    destination::Array{T},
    ::Type{T},
    index::Int,
    count::Int,
    cursor,
) where {T<:Union{UInt64,Float64}}
    width = Val(_draw_bits(T))
    last_index = index + count - 1
    @inbounds while index <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, width)
        destination[index] = _from_bits(T, raw)
        index += 1
    end
    return cursor
end

@inline function _fill_uniform_dense_cpu!(
    rng,
    position,
    destination::Array{T},
    ::Type{T},
    indices,
) where {T<:Union{Bool,UInt32,UInt64,Float32,Float64}}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, FAMILY_BITS, _position_block(position), position.bit)
    _fill_dense_cursor!(rng, destination, T, first(indices), length(indices), cursor)
    return nothing
end

@inline function _store_bool_blocks4!(destination, index, blocks)
    @inbounds for block_lane = 1:4, word_lane = 1:4, bit_lane = 0:31
        word = blocks[block_lane][word_lane]
        offset = 128(block_lane - 1) + 32(word_lane - 1) + bit_lane
        destination[index+offset] = isodd(word >> (31 - bit_lane))
    end
    return nothing
end

@inline function _store_u32_blocks4!(destination, index, blocks)
    @inbounds for block_lane = 1:4, word_lane = 1:4
        destination[index+4(block_lane-1)+word_lane-1] = blocks[block_lane][word_lane]
    end
    return nothing
end

@inline function _store_u64_blocks4!(destination, index, blocks)
    @inbounds for block_lane = 1:4, word_lane = 1:2
        words = blocks[block_lane]
        destination[index+2(block_lane-1)+word_lane-1] =
            (UInt64(words[2word_lane-1]) << 32) | UInt64(words[2word_lane])
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

@inline function _fill_aligned_blocks4!(rng, destination::Array{Bool}, index, last, block)
    while index + 511 <= last && block <= _max_block(rng) - UInt64(3)
        blocks = _blocks4(rng, FAMILY_BITS, block)
        _store_bool_blocks4!(destination, index, blocks)
        index += 512
        block += UInt64(4)
    end
    return index, block
end
@inline function _fill_aligned_blocks4!(rng, destination::Array{UInt32}, index, last, block)
    while index + 15 <= last && block <= _max_block(rng) - UInt64(3)
        blocks = _blocks4(rng, FAMILY_BITS, block)
        _store_u32_blocks4!(destination, index, blocks)
        index += 16
        block += UInt64(4)
    end
    return index, block
end
@inline function _fill_aligned_blocks4!(rng, destination::Array{UInt64}, index, last, block)
    while index + 7 <= last && block <= _max_block(rng) - UInt64(3)
        blocks = _blocks4(rng, FAMILY_BITS, block)
        _store_u64_blocks4!(destination, index, blocks)
        index += 8
        block += UInt64(4)
    end
    return index, block
end
@inline function _fill_aligned_blocks4!(
    rng,
    destination::Array{Float32},
    index,
    last,
    block,
)
    while index + 63 <= last && block <= _max_block(rng) - UInt64(11)
        first = _blocks4(rng, FAMILY_BITS, block)
        second = _blocks4(rng, FAMILY_BITS, block + UInt64(4))
        third = _blocks4(rng, FAMILY_BITS, block + UInt64(8))
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

@inline _fill_aligned_blocks4!(rng, destination::Array{Float64}, index, last, block) =
    (index, block)

@inline function _fill_uniform_blocks4_cpu!(
    rng::Philox4x32,
    position::_Position64,
    destination::Array{T},
    ::Type{T},
    indices,
) where {T<:Union{Bool,UInt32,UInt64,Float32,Float64}}
    isempty(indices) && return nothing
    index = first(indices)
    last_index = last(indices)
    block = position.block
    bit = position.bit
    if iszero(bit)
        index, block = _fill_aligned_blocks4!(rng, destination, index, last_index, block)
    end
    index > last_index && return nothing
    cursor = _dense_cursor(rng, FAMILY_BITS, block, bit)
    _fill_dense_cursor!(rng, destination, T, index, last_index - index + 1, cursor)
    return nothing
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval @inline _fill_uniform_dense_cpu!(
        rng::Philox4x32,
        position::_Position64,
        destination::Array{$T},
        ::Type{$T},
        indices,
    ) = _fill_uniform_blocks4_cpu!(rng, position, destination, $T, indices)
end

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

    cursor = _dense_cursor(rng, FAMILY_BITS, _position_block(position), position.bit)
    index = first_index
    last_index = last(indices)
    chunk = ((first_index - 1) >> 6) + 1
    @inbounds while index + 63 <= last_index
        raw, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(64))
        # Stream bits are MSB-first; BitArray chunks store their first bit lowest.
        destination.chunks[chunk] = bitreverse(raw)
        index += 64
        chunk += 1
    end
    if index <= last_index
        raw = UInt64(0)
        offset = 0
        @inbounds while index <= last_index
            bit, cursor = _take_dense_bits_unchecked(rng, FAMILY_BITS, cursor, Val(1))
            raw |= bit << offset
            index += 1
            offset += 1
        end
        @inbounds destination.chunks[chunk] = raw
    end
    return nothing
end

@inline _fill_uniform_dense_cpu!(rng, position, destination, ::Type{T}, indices) where {T} =
    _fill_uniform_unchecked!(rng, position, destination, T, indices)
