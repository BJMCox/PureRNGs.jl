# The engine contract: the hooks through which the codecs read a generator's
# stream. `docs/dev/engine-interface.md` states each hook's contract. The
# built-in generators serve every hook from their dense bit cursor.

# A codec reads `count` takes of `W` bits each per draw. The default is one
# take of the codec's whole width. A codec that reads several overrides it.
@inline _codec_takes(codec, ::Type{T}) where {T} = (1, Val(Int(_fill_width(codec, T))))

# The backend token selects transforms, allocation, residence, and the
# serviceability checks. A built-in generator carries its own.
@inline _engine_backend(rng::AbstractPureRNG) = rng.device

@inline _take_bits(rng, cursor::_DenseBitCursor, width::Val) =
    _take_dense_bits_unchecked(rng, cursor, width)

# The child stream a Gamma draw continues on when every candidate rejects, keyed
# by the draw's stream ordinal: the child generator and a cursor at its start.
# The fallback reads the child through `_take_bits` alone.
@inline function _child_cursor(rng::AbstractPureRNG, purpose::UInt64)
    child = subrng(rng, purpose)
    return child, _dense_cursor(child, _position_block(child.position), child.position.bit)
end

@inline _advance_block_unchecked(block::UInt64, count::UInt64) = block + count
@inline function _advance_block_unchecked(block::NTuple{2,UInt64}, count::UInt64)
    lo = block[1] + count
    return lo, block[2] + UInt64(lo < block[1])
end

# The dense stream packs takes with no gaps, so a skip is a bit offset. A skip
# that stays in the cursor's block keeps its decoded words.
@inline function _skip_takes(
    rng,
    cursor::_DenseBitCursor,
    count::Integer,
    ::Val{W},
) where {W}
    shift = _block_shift(rng)
    offset = (UInt64(cursor.lane) << 6) + UInt64(cursor.bit) + UInt64(count) * UInt64(W)
    blocks = offset >> shift
    bit = UInt16(offset & ((UInt64(1) << shift) - UInt64(1)))
    iszero(blocks) &&
        return _DenseBitCursor(cursor.block, cursor.block_words, bit >> 6, bit & UInt16(63))
    return _dense_cursor(rng, _advance_block_unchecked(cursor.block, blocks), bit)
end

@inline _first_block_word(block::UInt64) = block
@inline _first_block_word(block::NTuple{2,UInt64}) = block[1]

# The stream index modulo 2^64, as `_position_index` gives for a position.
@inline _cursor_ordinal(rng, cursor::_DenseBitCursor) =
    (_first_block_word(cursor.block) << _block_shift(rng)) +
    (UInt64(cursor.lane) << 6) +
    UInt64(cursor.bit)

# The cursor of zero-based draw `ordinal` of a fill of `count`-take draws that
# starts at the held position. The fill has already reserved its span.
@inline function _fill_cursor(
    rng::AbstractPureRNG,
    count::Integer,
    ::Val{W},
    ordinal::UInt64,
) where {W}
    bits_hi, bits_lo = _mulhilo64(ordinal, UInt64(count) * UInt64(W))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    return _dense_cursor(rng, _position_block(position), position.bit)
end

# Reserve `count` takes at the held position: the cursor of the first take and
# the generator past the last one.
@inline function _draw_cursor(rng::AbstractPureRNG, count::Integer, ::Val{W}) where {W}
    bits_hi, bits_lo = _mulhilo64(UInt64(count), UInt64(W))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    return _dense_cursor(rng, _position_block(rng.position), rng.position.bit), next_rng
end

# The generator past `draws` draws of `count` takes. The engine gets both factors,
# so it checks the whole span before a fill writes anything. The default rejects
# a take count past UInt64 before it multiplies.
@inline function _reserve_draws(rng, draws::Integer, count::Integer, width::Val)
    takes = Base.checked_mul(UInt64(draws), UInt64(count))
    return last(_draw_cursor(rng, takes, width))
end

@inline function _reserve_draws(
    rng::AbstractPureRNG,
    draws::Integer,
    count::Integer,
    ::Val{W},
) where {W}
    takes_hi, takes = _mulhilo64(UInt64(draws), UInt64(count))
    iszero(takes_hi) || return _reserve_wide(rng, (BigInt(takes_hi) << 64) + takes, W)
    bits_hi, bits_lo = _mulhilo64(takes, UInt64(W))
    return _reserve(rng, bits_lo, bits_hi)
end

# A span of 2^64 takes or more needs BigInt arithmetic, as a wide address does.
@noinline function _reserve_wide(rng::AbstractPureRNG, takes::BigInt, width::Integer)
    span = takes * width
    _, valid = _try_advance(rng, UInt64(0), UInt64(0))
    valid || _stream_exhausted(typeof(rng), span)
    current = BigInt(rngposition(rng))
    capacity = _stream_capacity(typeof(rng))
    current + span <= capacity || _stream_exhausted(typeof(rng), span)
    return _rebuild(
        rng,
        _position_from_bits(typeof(rng), current + span, capacity),
        rng.device,
    )
end

# The generator at the start of draw `i`, counting from one, of `count`-take draws.
@inline function _addressed_state(
    rng::AbstractPureRNG,
    count::Integer,
    ::Val{W},
    i::Integer,
) where {W}
    stride = UInt64(count) * UInt64(W)
    stride <= typemax(UInt16) && return _addressed_rng(rng, UInt16(stride), i)
    return _addressed_rng(rng, stride, i)
end

# The draws an engine gets from its hooks alone. A built-in generator overrides
# all three with the paths its fills and draws were tuned on.
@inline function _engine_draw_next(rng, codec, ::Type{T}) where {T}
    count, width = _codec_takes(codec, T)
    cursor, next_rng = _draw_cursor(rng, count, width)
    return first(_codec_take(codec, rng, cursor, T)), next_rng
end

@inline function _engine_draw_at(rng, codec, ::Type{T}, i::Integer) where {T}
    count, width = _codec_takes(codec, T)
    addressed = _addressed_state(rng, count, width, i)
    cursor, _ = _draw_cursor(addressed, count, width)
    return first(_codec_take(codec, addressed, cursor, T))
end

@noinline _no_engine_device_fill() =
    throw(ArgumentError("this generator has no device fill; move it to the CPU"))

# A host fill that walks one cursor, or one per chunk when threaded. A device
# fill needs the engine's own method.
function _engine_fill!(rng, destination::AbstractArray{T}, threaded::Bool, codec) where {T}
    _engine_backend(rng) isa _CPUBackend || _no_engine_device_fill()
    count, width = _codec_takes(codec, T)
    next_rng = _reserve_draws(rng, length(destination), count, width)
    isempty(destination) && return destination, next_rng
    cursor = _fill_cursor(rng, count, width, UInt64(0))
    indices = eachindex(destination)
    if threaded
        _run_chunks(length(destination), _fill_chunk_elements(codec, T)) do first, last
            chunk = _fill_cursor(rng, count, width, UInt64(first - 1))
            for ordinal = first:last
                value, chunk = _codec_take(codec, rng, chunk, T)
                destination[_destination_index(indices, ordinal)] = value
            end
        end
    else
        for index in indices
            value, cursor = _codec_take(codec, rng, cursor, T)
            destination[index] = value
        end
    end
    return destination, next_rng
end

# The built-in paths: a scalar reserves its span and reads it at the held
# position, normal and exponential scalars chain across the block boundary, and
# fills run the tuned CPU and device launchers.
@inline function _engine_draw_next(
    rng::_ScalarUniformGenerators,
    codec,
    ::Type{T},
) where {T}
    next_rng = _reserve(rng, UInt64(_fill_width(codec, T)), UInt64(0))
    return _transformed_draw_unchecked(codec, rng, rng.position, T), next_rng
end
@inline _engine_draw_next(
    rng::_ScalarUniformGenerators,
    codec::Union{_NormalCodec,_ExponentialCodec},
    ::Type{T},
) where {T} = _draw_next(rng, codec, T)
@inline _engine_draw_at(
    rng::_ScalarUniformGenerators,
    codec,
    ::Type{T},
    i::Integer,
) where {T} = _draw_at(rng, codec, T, i)
@inline _engine_fill!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray,
    threaded::Bool,
    codec,
) = _fill_prevalidated!(rng, destination, threaded, codec)

# The bodies of `rand_next`, `rand_next!`, and `rand_at` for distributions, with
# the public signatures and any engine. The Distributions extensions own their
# methods. The built-in public methods and an external engine's own methods call
# them.
function _engine_rand_next end
function _engine_rand_next! end
function _engine_rand_at end

# Addressed draws `indices` are the fill that starts at the first of them.
@inline function _engine_addressed_array(
    rng,
    codec,
    ::Type{T},
    indices::AbstractUnitRange{<:Integer},
    threaded::Bool,
) where {T}
    _check_serviceability(rng, T)
    isempty(indices) && return _allocate_draw_array(_engine_backend(rng), T, (0,))
    count, width = _codec_takes(codec, T)
    addressed = _addressed_state(rng, count, width, first(indices))
    destination = _allocate_draw_array(_engine_backend(rng), T, (length(indices),))
    return first(_engine_fill!(addressed, destination, threaded, codec))
end

# A column codec fills a matrix column per draw: `_codec_takes` counts a column's
# takes and `_column_take!` writes one column from a cursor. The default reserves
# every column, then starts each column's cursor with `_fill_cursor`, on any
# backend. An engine overrides it to feed sequential cursors from its bulk stream.
function _engine_fill_columns!(
    rng,
    destination::AbstractMatrix{T},
    threaded::Bool,
    codec,
) where {T}
    count, width = _codec_takes(codec, T)
    next_rng = _reserve_draws(rng, size(destination, 2), count, width)
    backend = _fill_backend(_engine_backend(rng), destination)
    _fill_columns!(backend, rng, destination, threaded, codec)
    return destination, next_rng
end

# The host walks each column with one cursor, which keeps the column in cache.
_fill_columns!(backend::_CPUBackend, rng, destination, threaded::Bool, codec) =
    _foreach_column!(backend, _column_fill!, destination, threaded, rng, codec)

@inline function _column_fill!(destination, column, rng, codec)
    count, width = _codec_takes(codec, eltype(destination))
    cursor = _fill_cursor(rng, count, width, UInt64(column - 1))
    _column_take!(codec, rng, cursor, destination, column)
    return nothing
end

# The workitems that fill the column kernel's resident blocks. A backend without
# a kernel occupancy query fills by element.
_column_workitems(backend, rng, destination, codec) = typemax(Int)

# A device with a column for every resident workitem fills whole columns, one
# cursor each. With fewer columns it fills the log-gamma matrix with a workitem
# per element, so the lanes stay busy however wide the columns are, then
# normalizes each column. The components of a column sit in consecutive spans,
# so element `i` is draw `i - 1` of the component spans.
function _fill_columns!(backend, rng, destination, threaded::Bool, codec)
    isempty(destination) && return destination
    size(destination, 2) >= _column_workitems(backend, rng, destination, codec) &&
        return _foreach_column!(backend, _column_fill!, destination, threaded, rng, codec)
    _foreach_element!(backend, _component_fill!, destination, rng, codec)
    _foreach_column!(backend, _normalize_fill!, destination, threaded, rng, codec)
    return destination
end

@inline function _component_fill!(destination, index, rng, codec)
    T = eltype(destination)
    count, width = _component_takes(codec, T)
    cursor = _fill_cursor(rng, count, width, UInt64(index - 1))
    component = (index - 1) % size(destination, 1) + 1
    destination[index] = _component_log(codec, rng, cursor, component, T)
    return nothing
end

@inline function _normalize_fill!(destination, column)
    _normalize_column!(destination, column)
    return nothing
end

@inline _normalize_fill!(destination, column, rng, codec) =
    _normalize_fill!(destination, column)
