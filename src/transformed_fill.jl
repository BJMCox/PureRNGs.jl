# A codec selects the width, mapping, and fill plan through dispatch.
abstract type _MappedFillCodec end
# The exponential codec carries the backend because the log approximation is per backend.
struct _ExponentialCodec{B<:_BackendToken}
    backend::B
end
const _TransformedFillCodec = Union{Val{:normal},_ExponentialCodec,_MappedFillCodec}

# The single seam a backend extension overrides to select a tuned device kernel.
# `nothing` keeps the generic one-work-item-per-element kernel.
@inline _device_fill_plan(backend, rng, codec, ::Type{T}) where {T} = nothing

@inline _transformed_draw_unchecked(::Val{:normal}, rng, position, T) =
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

@inline function _fill_transformed_dense_cpu!(
    rng,
    position,
    destination,
    ::Type{T},
    indices,
    codec::_TransformedFillCodec,
) where {T}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    width = Val(_fill_width(codec, T))
    @inbounds for index in indices
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, width)
        destination[index] = _cooperative_value(codec, T, raw)
    end
    return nothing
end

@inline function _fill_transformed_cursor!(
    rng,
    cursor,
    destination,
    ::Type{T},
    index::Int,
    count::Int,
    codec::_TransformedFillCodec,
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
        word < chunk_words && return :($(Symbol(:group_, word >> 3))[$((word&7)+1)])
        rest = word - chunk_words
        return :($(Symbol(:group_, chunks + (rest >> 1)))[$((rest&1)+1)])
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
    codec::_TransformedFillCodec,
    ::Type{T},
    width::Val,
) where {T}
    raws = _decode_transformed_group(rng, block, width)
    @inbounds for draw in eachindex(raws)
        destination[index+draw-1] = _cooperative_value(codec, T, raws[draw])
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
    codec::_TransformedFillCodec,
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
            cursor =
                _fill_transformed_cursor!(rng, cursor, destination, T, index, prefix, codec)
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
    _fill_transformed_cursor!(
        rng,
        cursor,
        destination,
        T,
        index,
        last_index - index + 1,
        codec,
    )
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
    codec::_TransformedFillCodec,
) where {T}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    _fill_transformed_cursor!(
        rng,
        cursor,
        destination,
        T,
        first(indices),
        length(indices),
        codec,
    )
    return nothing
end

@inline _fill_transformed_dense_cpu!(
    rng::Philox4x32,
    position::_Position64,
    destination::AbstractArray{T},
    ::Type{T},
    indices::AbstractUnitRange,
    codec::_TransformedFillCodec,
) where {T<:_UniformFloat} = _fill_transformed_blocks_cpu!(
    IndexStyle(destination),
    rng,
    position,
    destination,
    T,
    indices,
    codec,
)

@inline function _fill_transformed_grouped_unchecked!(
    rng,
    position,
    destination,
    ::Type{T},
    first::Int,
    group::Val{N},
    codec::_TransformedFillCodec,
) where {T,N}
    return _fill_grouped_cursor!(
        rng,
        position,
        destination,
        T,
        first,
        group,
        Val(_fill_width(codec, T)),
        codec,
    )
end

KernelAbstractions.@kernel function _transformed_fill_kernel!(
    rng,
    destination,
    ::Val{T},
    codec,
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds indices[firstindex(indices)+ordinal-1]
    bits_lo, bits_hi = _bit_span(UInt64(ordinal - 1), _fill_width(codec, T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = _transformed_draw_unchecked(codec, rng, position, T)
end

KernelAbstractions.@kernel function _transformed_fill_grouped_kernel!(
    rng,
    destination,
    ::Val{T},
    group::Val{N},
    codec,
) where {T,N}
    workitem = @index(Global, Linear)
    first = (workitem - 1) * N + 1
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _fill_width(codec, T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_transformed_grouped_unchecked!(rng, position, destination, T, first, group, codec)
end

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::_TransformedFillCodec,
    ::Nothing,
) where {T}
    _transformed_fill_kernel!(backend)(
        rng,
        destination,
        Val(T),
        codec;
        ndrange = length(destination),
    )
    return destination
end

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::_TransformedFillCodec,
    plan::Tuple{Val{:grouped},Val{N}},
) where {T,N}
    group = plan[2]
    workitems = cld(length(destination), _fill_group_size(group))
    _transformed_fill_grouped_kernel!(backend)(
        rng,
        destination,
        Val(T),
        group,
        codec;
        ndrange = workitems,
    )
    return destination
end

const _CPU_TRANSFORMED_FILL_CHUNK_BITS = UInt64(8192 * 32)

@inline _transformed_fill_chunk_elements(codec, ::Type{T}) where {T} =
    Int(_CPU_TRANSFORMED_FILL_CHUNK_BITS ÷ UInt64(_fill_width(codec, T)))

@inline function _launch_transformed!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::_TransformedFillCodec,
) where {T}
    plan = _device_fill_plan(backend, rng, codec, T)
    return _launch_device_fill!(backend, rng, destination, T, codec, plan)
end

function _launch_transformed!(
    ::KernelAbstractions.CPU,
    rng,
    destination::AbstractArray{T},
    ::Type{T},
    codec::_TransformedFillCodec,
) where {T}
    chunk_elements = _transformed_fill_chunk_elements(codec, T)
    indices = eachindex(destination)
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), _fill_width(codec, T))
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        chunk = _chunk_indices(indices, first, last)
        _fill_transformed_dense_cpu!(rng, position, destination, T, chunk, codec)
    end
    return destination
end

function _launch_transformed!(
    ::KernelAbstractions.CPU,
    rng,
    destination::BitArray,
    ::Type{Bool},
    codec::_MappedFillCodec,
)
    _fill_transformed_dense_cpu!(
        rng,
        rng.position,
        destination,
        Bool,
        eachindex(destination),
        codec,
    )
    return destination
end

@inline function _fill_transformed_prevalidated!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded::Bool,
    codec::_TransformedFillCodec,
) where {T}
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _fill_width(codec, T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_transformed_dense_cpu!(
            rng,
            rng.position,
            destination,
            T,
            eachindex(destination),
            codec,
        )
        return destination, next_rng
    end
    backend = _fill_backend(destination)
    _launch_transformed!(backend, rng, destination, T, codec)
    return destination, next_rng
end

@inline function _rand_transformed_next_fill!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded::Bool,
    codec::_TransformedFillCodec,
) where {T}
    _check_fill_device(rng, destination)
    _check_serviceability(rng, T)
    return _fill_transformed_prevalidated!(rng, destination, threaded, codec)
end

@inline function _rand_transformed_next_array(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Tuple,
    codec::_TransformedFillCodec,
) where {T}
    _check_serviceability(rng, T)
    destination = _allocate_draw_array(rng.device, T, dims)
    return _fill_transformed_prevalidated!(rng, destination, true, codec)
end
