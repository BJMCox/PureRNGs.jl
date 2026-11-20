# A codec selects the width, mapping, and fill plan through dispatch.
abstract type _MappedFillCodec end
# Both transform codecs carry the backend because the token selects the normal
# and exponential transforms.
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

@inline _draw_transform(codec) = _transformed_draw_unchecked

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
    raw, cursor = _take_bits(rng, cursor, Val(Int(_fill_width(codec, T))))
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
    if destination isa AbstractArray{T} &&
       indices isa AbstractUnitRange &&
       IndexStyle(destination) isa IndexLinear &&
       _use_grouped_fill(rng, codec, T)
        return _fill_transformed_blocks_cpu!(rng, position, destination, T, indices, codec)
    end
    isempty(indices) && return nothing
    # One range check covers every store below, so the inner loops skip the
    # per-element checks. Those cost normal and exponential fills 1.1-1.5x (BenchmarkTools,
    # serial 2^16 fills).
    checkbounds(destination, indices)
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    @inbounds for index in indices
        value, cursor = _codec_take(codec, rng, cursor, T)
        destination[index] = value
    end
    return nothing
end

# The uniform codec keeps its own dense CPU path: it owns block-aligned stores
# that no mapping codec can use, and `_from_bits` needs no per-draw arithmetic.
@inline function _fill_transformed_cpu!(
    rng,
    position,
    destination,
    ::Type{T},
    indices,
    ::Val{:uniform},
) where {T}
    # The widest dense loop tests a 512-Bool group, including one last test
    # after the final store. Near typemax(Int), iterate the valid indices directly.
    if indices isa AbstractUnitRange && last(indices) > typemax(Int) - 512
        return _fill_uniform_position_cpu!(rng, position, destination, T, indices)
    end
    return _fill_uniform_cpu!(rng, position, destination, T, indices)
end

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

# A group ending at a block boundary has constant shifts throughout. Round up
# to four-block chunks for Philox4x32 when the group stays small.
@inline function _transformed_group_draws(::Type{R}, width::Int) where {R}
    block_bits = 1 << _block_shift(R)
    draws = block_bits ÷ gcd(width, block_bits)
    if R <: Philox4x32 && 4draws <= 128 && !iszero((draws * width ÷ block_bits) % 4)
        return 4draws
    end
    return draws
end

# Draws needed to carry a cursor at `offset` bits into a block up to the next
# block boundary. Widths coarser than the offset never align, hence the -1. The
# width is a type parameter so the modular inverse folds away.
@inline function _transformed_group_prefix(::Val{W}, offset::Int, block_bits::Int) where {W}
    step = gcd(W, block_bits)
    iszero(offset % step) || return -1
    modulus = block_bits ÷ step
    return ((modulus - offset ÷ step) * invmod(W ÷ step, modulus)) % modulus
end

# Decode one aligned group of `W`-bit draws from consecutive blocks.
# Philox4x32 generates four blocks at a time to keep its vector width.
@generated function _decode_transformed_group(
    rng::R,
    block,
    ::Val{W},
    ::Type{U},
) where {R<:_ScalarUniformGenerators,W,U<:Union{UInt32,UInt64}}
    block_bits = 1 << _block_shift(R)
    draws = _transformed_group_draws(R, W)
    blocks = draws * W ÷ block_bits
    chunks = R <: Philox4x32 ? blocks ÷ 4 : 0
    word_bits = 8sizeof(U)
    words_per_block = block_bits ÷ word_bits
    words_per_group = 4words_per_block
    chunk_words = chunks * words_per_group
    function word_ref(word)
        if word < chunk_words
            group, lane = divrem(word, words_per_group)
            ref = Symbol(:group_, group)
            if U === UInt32
                subblock, slot = divrem(lane, words_per_block)
                return :($ref[$(subblock+1)][$(slot+1)])
            end
            return :($ref[$(lane+1)])
        end
        tail, lane = divrem(word - chunk_words, words_per_block)
        return :($(Symbol(:group_, chunks+tail))[$(lane+1)])
    end
    # Use native UInt32 words for Float32 normals from a 32-bit core.
    group_blocks = U === UInt32 ? :_blocks4 : :_blocks4_words
    single_block = U === UInt32 ? :_block : :_block_words
    body = Expr[:(mask = _low_mask(UInt16($W)) % $U)]
    for source = 0:(blocks-3chunks-1)
        words =
            source < chunks ? :($group_blocks(rng, block + UInt64($(4source)))) :
            :($single_block(
                rng,
                _advance_stream_block_unchecked(block, UInt64($(3chunks + source))),
            ))
        push!(body, :($(Symbol(:group_, source)) = $words))
    end
    raws = map(0:(draws-1)) do draw
        word, start = divrem(W * draw, word_bits)
        start + W <= word_bits &&
            return :(($(word_ref(word)) >> $(word_bits - start - W)) & mask)
        head = :($(word_ref(word)) << $(start + W - word_bits))
        tail = :($(word_ref(word + 1)) >> $(2word_bits - start - W))
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
    rng,
    block,
    codec,
    ::Type{T},
    width::Val,
) where {T}
    raws = _decode_transformed_group(rng, block, width, UInt64)
    @inbounds for draw in eachindex(raws)
        destination[index+draw-1] = _cooperative_value(codec, T, raws[draw])
    end
    return nothing
end

# The AS241 tail branch costs 0.83 ns per element even when every element is
# central, because it stops the group loop vectorizing. Pass one runs the
# central rational over the whole group without a branch. Pass two overwrites
# the tail. Both AS241 formulas receive the same Float32 inputs as scalar draws.
#
# Keep Float64 on the one-pass path. The two-pass transform was slower with
# half-width vectors: 5.21 against 5.07 ns per element.
@inline function _store_transformed_group!(
    destination,
    index::Int,
    rng,
    block,
    ::_NormalCodec{_CPUBackend},
    ::Type{Float32},
    width::Val,
)
    word = rng isa _ScalarUniform32Generators ? UInt32 : UInt64
    raws = _decode_transformed_group(rng, block, width, word)
    # On the 23-bit midpoint lattice, q = (2k - (2^23 - 1)) * 2^-24 exactly.
    # The inclusive central interval contains center_width + 1 consecutive k.
    midpoint_span = Int32(0x7f_ffff)
    center_width = UInt32(0.425f0 * Float32(0x1p24))
    center_start = (UInt32(midpoint_span) - center_width) >> 1
    @inbounds for draw in eachindex(raws)
        q = Float32(Int32(2) * (raws[draw] % Int32) - midpoint_span) * Float32(0x1p-24)
        destination[index+draw-1] = _as241_central(q)
    end
    @inbounds for draw in eachindex(raws)
        raw = raws[draw]
        # Below center_start, unsigned subtraction wraps past center_width.
        if raw - center_start > center_width
            u = _open_midpoint(Float32, UInt64(raw))
            destination[index+draw-1] = _as241_tail(u, u - 0.5f0)
        end
    end
    return nothing
end

@inline function _fill_transformed_blocks_cpu!(
    rng,
    position,
    destination::AbstractArray{T},
    ::Type{T},
    indices,
    codec,
) where {T<:_UniformFloat}
    isempty(indices) && return nothing
    # One range check covers every store below, so the inner loops skip the
    # per-element checks. Those cost normal and exponential fills 1.1-1.5x (BenchmarkTools,
    # serial 2^16 fills).
    checkbounds(destination, indices)
    bits = Int(_fill_width(codec, T))
    width = Val(bits)
    block_bits = Int(_block_bits(rng))
    group = _transformed_group_draws(typeof(rng), bits)
    span = UInt64(group * bits ÷ block_bits)
    index = first(indices)
    remaining = length(indices)
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    if remaining >= group
        cursor = _ensure_dense_cursor(rng, cursor)
        offset = 64 * Int(cursor.lane) + Int(cursor.bit)
        prefix = _transformed_group_prefix(width, offset, block_bits)
        if prefix >= 0 && remaining - prefix >= group
            cursor = _fill_cursor!(rng, cursor, destination, T, index, prefix, codec)
            index += prefix
            remaining -= prefix
            cursor = _ensure_dense_cursor(rng, cursor)
            block = cursor.block
            # The caller reserved the whole span, including every group.
            while remaining >= group
                _store_transformed_group!(destination, index, rng, block, codec, T, width)
                remaining -= group
                iszero(remaining) && return nothing
                index += group
                block = _advance_stream_block_unchecked(block, span)
            end
            cursor = _dense_cursor(rng, block, UInt16(0))
        end
    end
    _fill_cursor!(rng, cursor, destination, T, index, remaining, codec)
    return nothing
end

@inline _use_grouped_fill(rng, codec, T) =
    rng isa Philox4x32 &&
    T <: _UniformFloat &&
    codec isa Union{_NormalCodec,_ExponentialCodec,_MappedFillCodec}
@inline _use_grouped_fill(
    ::_ScalarUniformGenerators,
    ::_NormalCodec{_CPUBackend},
    ::Type{Float32},
) = true

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
        destination[_destination_index(indices, ordinal)] = value
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
    chunk_elements = _storage_chunk_elements(destination, _fill_chunk_elements(codec, T))
    indices = eachindex(destination)
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), _fill_width(codec, T))
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        chunk = _chunk_indices(indices, first, last)
        _fill_transformed_cpu!(rng, position, destination, T, chunk, codec)
    end
    return destination
end

# The whole span is reserved before any element is written, so a fill
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
    rng,
    destination::AbstractArray{T},
    threaded::Bool,
    codec::_TransformedFillCodec,
) where {T}
    _check_fill_device(rng, destination)
    _check_serviceability(rng, T)
    return _engine_fill!(rng, destination, threaded, codec)
end

@inline function _rand_transformed_next_array(
    rng,
    ::Type{T},
    dims::Tuple,
    codec::_TransformedFillCodec,
    threaded::Bool,
) where {T}
    _check_serviceability(rng, T)
    destination = _allocate_draw_array(_engine_backend(rng), T, dims)
    return _engine_fill!(rng, destination, threaded, codec)
end

# A scalar draw is the one-element case of a fill: the codec's width, reserved
# and chained across the block boundary, then mapped.
@inline function _draw_next(rng::_ScalarUniformGenerators, codec, ::Type{T}) where {T}
    next_rng = _reserve_scalar(rng, _fill_width(codec, T))
    raw = _chain_bits(rng, next_rng, Val(Int(_fill_width(codec, T))))
    return _cooperative_value(codec, T, raw), next_rng
end

# A complex normal spans two draws, wider than one word, so it reserves its whole
# span and reads it at the held position.
@inline function _draw_next(
    rng::_ScalarUniformGenerators,
    codec::_NormalCodec,
    ::Type{T},
) where {T<:_ComplexResult}
    next_rng = _reserve_scalar(rng, _fill_width(codec, T))
    return _transformed_draw_unchecked(codec, rng, rng.position, T), next_rng
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
