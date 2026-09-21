# A codec selects the width, mapping, and fill plan through dispatch.
abstract type _MappedFillCodec end
# Both transform codecs carry the backend because the token selects the
# transform: [R28] for the normal, [R63] for the exponential.
struct _NormalCodec{B<:_BackendToken}
    backend::B
end
struct _ExponentialCodec{B<:_BackendToken}
    backend::B
end
# The codecs the two public transformed entries accept. A backend token wraps a
# codec and is not one, so the bound keeps that mistake a MethodError.
const _TransformedFillCodec =
    Union{Val{:uniform},_NormalCodec,_ExponentialCodec,_MappedFillCodec}

# The single seam a backend extension overrides to select a tuned device kernel.
# `nothing` keeps the generic one-work-item-per-element kernel.
@inline _device_fill_plan(backend, rng, codec, ::Type{T}) where {T} = nothing

@inline _transformed_draw_unchecked(::Val{:uniform}, rng, position, T) =
    _draw_unchecked(rng, position, T)
@inline _transformed_draw_unchecked(::_NormalCodec, rng, position, T) =
    _draw_normal_unchecked(rng, position, T)
@inline _transformed_draw_unchecked(::_ExponentialCodec, rng, position, T) =
    _draw_exponential_unchecked(rng, position, T)
@inline function _transformed_draw_unchecked(
    codec::_MappedFillCodec,
    rng,
    position,
    ::Type{T},
) where {T}
    raw = _extract_bits_unchecked(
        rng,
        _position_block(position),
        position.bit,
        Val(_fill_width(codec, T)),
    )
    return _cooperative_value(codec, T, raw)
end

# One element off the dense cursor. The default is one fixed-width draw through
# the codec's map; a codec whose draw is wider than a word, or whose value is an
# index rather than a converted raw, overrides it.
@inline function _codec_take(codec, rng, cursor, ::Type{T}) where {T}
    raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(_fill_width(codec, T)))
    return _cooperative_value(codec, T, raw), cursor
end

@inline function _fill_transformed_cpu!(
    rng,
    position,
    destination,
    ::Type{T},
    indices,
    codec,
) where {T}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    @inbounds for index in indices
        value, cursor = _codec_take(codec, rng, cursor, T)
        destination[index] = value
    end
    return nothing
end

# The uniform codec keeps its own dense CPU path: it owns block-aligned stores
# that no mapping codec can use, and `_from_bits` needs no per-draw arithmetic.
@inline _fill_transformed_cpu!(
    rng,
    position,
    destination,
    ::Type{T},
    indices,
    ::Val{:uniform},
) where {T} = _fill_uniform_cpu!(rng, position, destination, T, indices)

@inline function _fill_cursor!(
    rng,
    cursor,
    destination,
    ::Type{T},
    index::Int,
    count::Int,
    codec,
) where {T}
    width = Val(_fill_width(codec, T))
    @inbounds for offset = 0:(count-1)
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, width)
        destination[index+offset] = _cooperative_value(codec, T, raw)
    end
    return cursor
end

# A Philox4x32 block is 128 bits, so `128 ÷ gcd(width, 128)` draws already cover
# a whole number of blocks and every shift inside the group becomes a constant.
# Four blocks at a time is what keeps the generator vectorized, so take four
# times that many draws when it also buys whole four-block chunks and stays small.
@inline function _transformed_group_draws(width::Int)
    draws = 128 ÷ gcd(width, 128)
    return iszero((draws * width ÷ 128) % 4) || 4draws > 128 ? draws : 4draws
end

# Draws needed to carry a cursor at `offset` bits into a block up to the next
# block boundary. Widths coarser than the offset never align, hence the -1. The
# width is a type parameter so the modular inverse folds away.
@inline function _transformed_group_prefix(::Val{W}, offset::Int) where {W}
    step = gcd(W, 128)
    iszero(offset % step) || return -1
    modulus = 128 ÷ step
    return ((modulus - offset ÷ step) * invmod(W ÷ step, modulus)) % modulus
end

# Decode one aligned group of `W`-bit draws out of consecutive Philox4x32
# blocks. Blocks arrive four at a time so the counter-based generator keeps its
# vector width, and every shift is a constant.
@generated function _decode_transformed_group(
    rng::Philox4x32,
    block::UInt64,
    ::Val{W},
) where {W}
    draws = _transformed_group_draws(W)
    blocks = draws * W ÷ 128
    chunks = blocks ÷ 4
    chunk_words = 8chunks
    function word_ref(word)
        word < chunk_words && return :($(Symbol(:group_, word>>3))[$((word&7)+1)])
        rest = word - chunk_words
        return :($(Symbol(:group_, chunks+(rest>>1)))[$((rest&1)+1)])
    end
    body = Expr[:(mask = _low_mask(UInt16($W)))]
    for source = 0:(chunks+blocks%4-1)
        words =
            source < chunks ? :(_blocks4_words(rng, block + UInt64($(4source)))) :
            :(_block_words(rng, block + UInt64($(3chunks + source))))
        push!(body, :($(Symbol(:group_, source)) = $words))
    end
    raws = map(0:(draws-1)) do draw
        offset = W * draw
        word, start = offset >> 6, offset & 63
        start + W <= 64 && return :(($(word_ref(word)) >> $(64 - start - W)) & mask)
        head = :($(word_ref(word)) << $(start + W - 64))
        tail = :($(word_ref(word + 1)) >> $(128 - start - W))
        return :(($head | $tail) & mask)
    end
    push!(body, Expr(:tuple, raws...))
    return Expr(:block, body...)
end

# The decoded group stays a tuple so the map runs as a loop. Inlining one
# quantile or logarithm per draw instead costs about 1 ns per element.
@inline function _store_transformed_group!(
    destination,
    index::Int,
    rng::Philox4x32,
    block::UInt64,
    codec,
    ::Type{T},
    width::Val,
) where {T}
    raws = _decode_transformed_group(rng, block, width)
    @inbounds for draw in eachindex(raws)
        destination[index+draw-1] = _cooperative_value(codec, T, raws[draw])
    end
    return nothing
end

# The AS241 tail branch costs 0.83 ns per element even when every element is
# central, because it stops the group loop vectorizing. Pass one runs the
# central rational over the whole group with no branch, pass two overwrites the
# elements that belong in the tail. Each element still sees the same operations
# in the same order, so no value moves.
#
# `Float32` only. Pass two repeats the midpoint conversion for every element of
# the group, and at `Float64` the vector is half as wide, so the repeat costs
# more than the branch it removes: 5.21 against 5.07 ns per element.
@inline function _store_transformed_group!(
    destination,
    index::Int,
    rng::Philox4x32,
    block::UInt64,
    ::_NormalCodec{_CPUBackend},
    ::Type{Float32},
    width::Val,
)
    raws = _decode_transformed_group(rng, block, width)
    @inbounds for draw in eachindex(raws)
        destination[index+draw-1] =
            _as241_central(_open_midpoint(Float32, raws[draw]) - 0.5f0)
    end
    @inbounds for draw in eachindex(raws)
        u = _open_midpoint(Float32, raws[draw])
        q = u - 0.5f0
        abs(q) <= 0.425f0 || (destination[index+draw-1] = _as241_tail(u, q))
    end
    return nothing
end

@inline function _fill_transformed_blocks_cpu!(
    ::IndexLinear,
    rng::Philox4x32,
    position::_Position64,
    destination::AbstractArray{T},
    ::Type{T},
    indices,
    codec,
) where {T<:_UniformFloat}
    isempty(indices) && return nothing
    bits = Int(_fill_width(codec, T))
    width = Val(bits)
    group = _transformed_group_draws(bits)
    span = UInt64(group * bits ÷ 128)
    index = first(indices)
    last_index = last(indices)
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    if last_index - index + 1 >= group
        cursor = _ensure_dense_cursor(rng, cursor)
        prefix = _transformed_group_prefix(width, 64 * Int(cursor.lane) + Int(cursor.bit))
        if prefix >= 0 && last_index - index + 1 >= prefix + group
            cursor = _fill_cursor!(rng, cursor, destination, T, index, prefix, codec)
            index += prefix
            cursor = _ensure_dense_cursor(rng, cursor)
            block = cursor.block
            while index + group - 1 <= last_index &&
                  block <= _max_block(rng) - (span - UInt64(1))
                _store_transformed_group!(destination, index, rng, block, codec, T, width)
                index += group
                block += span
            end
            block == cursor.block || (cursor = _dense_cursor(rng, block, UInt16(0)))
        end
    end
    index > last_index && return nothing
    _fill_cursor!(rng, cursor, destination, T, index, last_index - index + 1, codec)
    return nothing
end

# A Cartesian destination keeps the per-element cursor: the group stores by
# linear index.
@inline function _fill_transformed_blocks_cpu!(
    ::IndexStyle,
    rng,
    position,
    destination,
    ::Type{T},
    indices,
    codec,
) where {T}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    _fill_cursor!(rng, cursor, destination, T, first(indices), length(indices), codec)
    return nothing
end

# `Val(:uniform)` is excluded because its own method already routes it to
# `_fill_uniform_cpu!`, and naming it here would make the two ambiguous.
@inline _fill_transformed_cpu!(
    rng::Philox4x32,
    position::_Position64,
    destination::AbstractArray{T},
    ::Type{T},
    indices::AbstractUnitRange,
    codec::Union{_NormalCodec,_ExponentialCodec,_MappedFillCodec},
) where {T<:_UniformFloat} = _fill_transformed_blocks_cpu!(
    IndexStyle(destination),
    rng,
    position,
    destination,
    T,
    indices,
    codec,
)

@inline function _fill_group!(
    rng,
    position,
    destination,
    ::Type{T},
    first::Int,
    ::Val{N},
    codec,
) where {T,N}
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    count = length(destination)
    indices = eachindex(destination)
    @inbounds for offset = 0:(N-1)
        ordinal = first + offset
        ordinal > count && break
        value, cursor = _codec_take(codec, rng, cursor, T)
        # [R67] the destination ordinal is an index into `eachindex`, not a raw
        # linear index, so a custom axis is written in draw order.
        destination[indices[firstindex(indices)+ordinal-1]] = value
    end
    return nothing
end

const _CPU_TRANSFORMED_FILL_CHUNK_BITS = UInt64(8192 * 32)

# The two chunk sizes were tuned separately, so the codec selects one. The
# uniform chunk also rounds down to a whole number of block stores, so no two
# work items share one store.
@inline _fill_chunk_elements(codec, ::Type{T}) where {T} =
    Int(_CPU_TRANSFORMED_FILL_CHUNK_BITS ÷ UInt64(_fill_width(codec, T)))
@inline function _fill_chunk_elements(::Val{:uniform}, ::Type{T}) where {T}
    raw = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_draw_bits(T)))
    group = _fill_store_elements(T)
    return raw - raw % group
end

function _launch_cpu!(rng, destination::AbstractArray{T}, ::Type{T}, codec) where {T}
    chunk_elements = _fill_chunk_elements(codec, T)
    indices = eachindex(destination)
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), _fill_width(codec, T))
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        chunk = _chunk_indices(indices, first, last)
        _fill_transformed_cpu!(rng, position, destination, T, chunk, codec)
    end
    return destination
end

# A BitArray chunk holds 64 elements, so a mapping codec whose chunk is not a
# multiple of 64 would have two work items write the same chunk.
function _launch_cpu!(rng, destination::BitArray, ::Type{Bool}, codec::_MappedFillCodec)
    _fill_transformed_cpu!(
        rng,
        rng.position,
        destination,
        Bool,
        eachindex(destination),
        codec,
    )
    return destination
end

# [R54] the whole span is reserved before any element is written, so a fill
# either runs or throws. Callers keep their own validation prefix ahead of this.
@inline function _fill_prevalidated!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded::Bool,
    codec,
) where {T}
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _fill_width(codec, T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_transformed_cpu!(
            rng,
            rng.position,
            destination,
            T,
            eachindex(destination),
            codec,
        )
        return destination, next_rng
    end
    backend = _fill_backend(rng.device, destination)
    if backend isa _CPUBackend
        _launch_cpu!(rng, destination, T, codec)
    else
        plan = _device_fill_plan(backend, rng, codec, T)
        _launch_device_fill!(backend, rng, destination, T, codec, plan)
    end
    return destination, next_rng
end

@inline function _rand_transformed_next_fill!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded,
    codec::_TransformedFillCodec,
) where {T}
    # The keyword is checked first, before the device, so every public fill
    # reports the same error for the same input whichever entry it came through.
    checked = _check_threaded(threaded)
    _check_fill_device(rng, destination)
    _check_serviceability(rng, T)
    return _fill_prevalidated!(rng, destination, checked, codec)
end

@inline function _rand_transformed_next_array(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Tuple,
    codec::_TransformedFillCodec,
) where {T}
    _check_serviceability(rng, T)
    destination = _allocate_draw_array(rng.device, T, dims)
    return _fill_prevalidated!(rng, destination, true, codec)
end

# A scalar draw is the one-element case of a fill: the codec's width, reserved
# and chained across the block boundary, then mapped.
@inline function _draw_next(rng::_ScalarUniformGenerators, codec, ::Type{T}) where {T}
    next_rng = _reserve_scalar(rng, _fill_width(codec, T))
    raw = _chain_bits(rng, next_rng, Val(Int(_fill_width(codec, T))))
    return _cooperative_value(codec, T, raw), next_rng
end

# An addressed draw is the held-position draw of the generator moved to that
# address, so it leaves the caller's generator alone.
@inline function _draw_at(
    rng::_ScalarUniformGenerators,
    codec,
    ::Type{T},
    i::Integer,
) where {T}
    addressed = _addressed_rng(rng, _fill_width(codec, T), i)
    return _transformed_draw_unchecked(codec, addressed, addressed.position, T)
end
