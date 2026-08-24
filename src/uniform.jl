const _ScalarUniform32Family = Union{Philox2x32,Philox4x32,Threefry2x32,Threefry4x32}
const _ScalarUniform64Family = Union{Philox2x64,Philox4x64,Threefry2x64,Threefry4x64}
const _ScalarUniformFamily = Union{_ScalarUniform32Family,_ScalarUniform64Family}

@inline _draw_bits(::Type{Bool}) = UInt16(1)
@inline _draw_bits(::Type{Float32}) = UInt16(24)
@inline _draw_bits(::Type{UInt32}) = UInt16(32)
@inline _draw_bits(::Type{Float64}) = UInt16(53)
@inline _draw_bits(::Type{UInt64}) = UInt16(64)

@inline _position_block(position::_Position64) = position.block
@inline _position_block(position::_Position128) = (position.lo, position.hi)

@inline function _draw_raw(rng::_ScalarUniformFamily, position, ::Val{W}) where {W}
    return _extract_bits_unchecked(
        rng,
        FAMILY_BITS,
        _position_block(position),
        position.bit,
        Val(W),
    )
end

@inline _draw_raw(rng::_ScalarUniformFamily, width) = _draw_raw(rng, rng.position, width)

@inline _from_bits(::Type{Bool}, value::UInt64) = isone(value)
@inline _from_bits(::Type{UInt32}, value::UInt64) = value % UInt32
@inline _from_bits(::Type{UInt64}, value::UInt64) = value
@inline _from_bits(::Type{Float32}, value::UInt64) =
    Float32(value % UInt32) * Float32(0x1p-24)
@inline _from_bits(::Type{Float64}, value::UInt64) = Float64(value) * 0x1p-53

@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{Bool}) =
    _from_bits(Bool, _draw_raw(rng, Val(1)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{Float32}) =
    _from_bits(Float32, _draw_raw(rng, Val(24)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{UInt32}) =
    _from_bits(UInt32, _draw_raw(rng, Val(32)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{Float64}) =
    _from_bits(Float64, _draw_raw(rng, Val(53)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, ::Type{UInt64}) =
    _from_bits(UInt64, _draw_raw(rng, Val(64)))

@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{Bool}) =
    _from_bits(Bool, _draw_raw(rng, position, Val(1)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{Float32}) =
    _from_bits(Float32, _draw_raw(rng, position, Val(24)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{UInt32}) =
    _from_bits(UInt32, _draw_raw(rng, position, Val(32)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{Float64}) =
    _from_bits(Float64, _draw_raw(rng, position, Val(53)))
@inline _draw_unchecked(rng::_ScalarUniformFamily, position, ::Type{UInt64}) =
    _from_bits(UInt64, _draw_raw(rng, position, Val(64)))

function Random.rand(::AbstractPureRNG)
    throw(ArgumentError("untyped immutable draws are forbidden; use rand(rng, T)"))
end

@inline function _rand_scalar(rng::_ScalarUniformFamily, ::Type{T}) where {T}
    _reserve(rng, UInt64(_draw_bits(T)), UInt64(0))
    return _draw_unchecked(rng, T)
end

@inline rand_next(rng::_ScalarUniformFamily) = rand_next(rng, Float64)

@inline function _rand_next_scalar(rng::_ScalarUniformFamily, ::Type{T}) where {T}
    next_rng = _reserve(rng, UInt64(_draw_bits(T)), UInt64(0))
    return next_rng, _draw_unchecked(rng, T)
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline Random.rand(rng::_ScalarUniformFamily, ::Type{$T}) = _rand_scalar(rng, $T)
        @inline rand_next(rng::_ScalarUniformFamily, ::Type{$T}) =
            _rand_next_scalar(rng, $T)
    end
end

@noinline function _fill_device_mismatch()
    throw(ArgumentError("destination device differs from the generator device"))
end

@inline _same_fill_device(::MLDataDevices.CPUDevice, ::MLDataDevices.CPUDevice) = true
@inline _same_fill_device(generator_device, destination_device) =
    generator_device == destination_device

@inline function _check_fill_device(rng::_ScalarUniformFamily, destination)
    _same_fill_device(rng.device, MLDataDevices.get_device(destination)) ||
        _fill_device_mismatch()
    return nothing
end

@inline _check_fill_serviceability(rng, destination, ::Type) = nothing

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

@inline _fill_uniform_unchecked!(rng, destination, ::Type{T}, indices) where {T} =
    _fill_uniform_unchecked!(rng, rng.position, destination, T, indices)

@inline _fill_uniform_unchecked!(rng, destination, ::Type{T}) where {T} =
    _fill_uniform_unchecked!(rng, destination, T, eachindex(destination))

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
    first = _select_stream_limb(cursor.limbs, cursor.lane)
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

KernelAbstractions.@kernel function _uniform_fill_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    _fill_uniform_unchecked!(rng, destination, T)
end

const _CPU_FILL_CHUNK_BITS = UInt64(4096 * 32)
const _CPU_FILL_MIN_WORKITEMS = 4

@inline function _dense_fill_chunk_elements(::Type{T}) where {T}
    raw = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_draw_bits(T)))
    group = _dense_fill_group(T)
    return raw - raw % group
end
@inline function _dense_fill_bounds(workitem::Int, count::Int, chunk_elements::Int)
    first = (workitem - 1) * chunk_elements + 1
    chunk_count = min(chunk_elements, count - first + 1)
    return first, first + chunk_count - 1
end

KernelAbstractions.@kernel function _uniform_fill_dense_kernel!(
    rng,
    destination,
    ::Type{T},
    chunk_elements,
) where {T}
    workitem = @index(Global, Linear)
    first, last = _dense_fill_bounds(workitem, length(destination), chunk_elements)
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _draw_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_uniform_dense_cpu!(rng, position, destination, T, first:last)
end

KernelAbstractions.@kernel function _uniform_fill_dense_serial_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    _fill_uniform_dense_cpu!(rng, rng.position, destination, T, eachindex(destination))
end

@inline _fill_backend(destination) = KernelAbstractions.get_backend(destination)
@inline _fill_backend(destination::BitArray) =
    KernelAbstractions.get_backend(destination.chunks)

function _launch_uniform!(backend, rng, destination, ::Type{T}) where {T}
    _uniform_fill_kernel!(backend)(rng, destination, T; ndrange = 1)
    return destination
end

function _launch_uniform!(
    backend::KernelAbstractions.CPU,
    rng,
    destination::Union{Array{T},BitArray},
    ::Type{T},
) where {T}
    chunk_elements = _dense_fill_chunk_elements(T)
    workitems = cld(length(destination), chunk_elements)
    if workitems < _CPU_FILL_MIN_WORKITEMS
        _uniform_fill_dense_serial_kernel!(backend)(rng, destination, T; ndrange = 1)
        return destination
    end
    _uniform_fill_dense_kernel!(backend)(
        rng,
        destination,
        T,
        chunk_elements;
        ndrange = workitems,
        workgroupsize = 1,
    )
    return destination
end

@inline function _rand_next_fill!(
    rng::_ScalarUniformFamily,
    destination::AbstractArray{T},
    threaded::Bool,
) where {T}
    _check_fill_device(rng, destination)
    _check_fill_serviceability(rng, destination, T)
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _draw_bits(T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return next_rng, destination
    if !threaded && rng.device isa MLDataDevices.CPUDevice
        _fill_uniform_dense_cpu!(rng, rng.position, destination, T, eachindex(destination))
        return next_rng, destination
    end
    backend = _fill_backend(destination)
    _launch_uniform!(backend, rng, destination, T)
    return next_rng, destination
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline function Random.rand!(
            rng::_ScalarUniformFamily,
            destination::AbstractArray{$T};
            threaded::Bool = true,
        )
            _, result = _rand_next_fill!(rng, destination, threaded)
            return result
        end

        @inline function rand_next!(
            rng::_ScalarUniformFamily,
            destination::AbstractArray{$T};
            threaded::Bool = true,
        )
            return _rand_next_fill!(rng, destination, threaded)
        end
    end
end

const _AddressIndex64 = Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}

@noinline function _invalid_address_index()
    throw(ArgumentError("addressed draw index must be positive"))
end

@noinline function _address_capacity_error()
    throw(ArgumentError("draw exceeds the generator counter capacity"))
end

@inline function _addressed_rng_device(
    rng::_ScalarUniformFamily,
    width::UInt16,
    i::_AddressIndex64,
)
    i < 1 && _invalid_address_index()
    index = UInt64(i)
    end_lo, end_hi = _bit_span(index, width)
    _, ok = _try_advance(rng, end_lo, end_hi)
    ok || _address_capacity_error()
    start_lo, start_hi = _bit_span(index - UInt64(1), width)
    position, _ = _try_advance(rng, start_lo, start_hi)
    position == rng.position && return rng
    return _rebuild(rng, position, rng.device)
end

@inline _addressed_rng(rng::_Position64Family, width::UInt16, i::_AddressIndex64) =
    _addressed_rng_device(rng, width, i)
@inline _addressed_rng(rng::_Position128Family, width::UInt16, i::_AddressIndex64) =
    _addressed_rng_device(rng, width, i)

@noinline function _addressed_rng(rng::_Position64Family, width::UInt16, i::Integer)
    i < 1 && _invalid_address_index()
    _, valid = _try_advance(rng, UInt64(0), UInt64(0))
    valid || _address_capacity_error()
    position = rng.position
    block_bits = BigInt(_block_bits(rng))
    current = BigInt(position.block) * block_bits + BigInt(position.bit)
    capacity = (BigInt(_max_block(rng)) + 1) * block_bits
    offset = (BigInt(i) - 1) * BigInt(width)
    current + offset + BigInt(width) <= capacity || _address_capacity_error()
    block, bit = divrem(current + offset, block_bits)
    return _rebuild(rng, _Position64(UInt64(block), UInt16(bit)), rng.device)
end

@noinline function _addressed_rng(rng::_Position128Family, width::UInt16, i::Integer)
    i < 1 && _invalid_address_index()
    _, valid = _try_advance(rng, UInt64(0), UInt64(0))
    valid || _address_capacity_error()
    position = rng.position
    block_bits = BigInt(_block_bits(rng))
    block = (BigInt(position.hi) << 64) + BigInt(position.lo)
    current = block * block_bits + BigInt(position.bit)
    capacity = (BigInt(1) << 128) * block_bits
    offset = (BigInt(i) - 1) * BigInt(width)
    current + offset + BigInt(width) <= capacity || _address_capacity_error()
    start_block, bit = divrem(current + offset, block_bits)
    mask = BigInt(typemax(UInt64))
    lo = UInt64(start_block & mask)
    hi = UInt64(start_block >> 64)
    return _rebuild(rng, _Position128(lo, hi, UInt16(bit)), rng.device)
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline randat(rng::_ScalarUniformFamily, ::Type{$T}, i::Integer) =
            _draw_unchecked(_addressed_rng(rng, _draw_bits($T), i), $T)
    end
end
