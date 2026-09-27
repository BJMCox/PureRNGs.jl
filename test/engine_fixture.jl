# An engine outside the built-in generator union. It wraps a built-in generator
# and serves the engine contract from the wrapped generator's public uniform
# draws: its cursor reads 64-bit words with `rand_at`, and its fill stages the
# words with one `rand_next` call. Every draw through it must therefore equal the
# wrapped generator's own draw, value and continuation state alike.
using PureRNGs
using Random

struct WrappedEngine{R}
    inner::R
end

# A cursor is a bit offset past the generator that anchors it.
struct WrappedCursor{R}
    anchor::R
    offset::UInt64
end

# A fill's cursor reads the staged words instead of calling `rand_at`.
struct StagedCursor{V}
    words::V
    base::UInt64
    offset::UInt64
end

_wrapped_word(cursor::WrappedCursor, index) = rand_at(cursor.anchor, UInt64, index + 1)
_wrapped_word(cursor::StagedCursor, index) = cursor.words[index+1]
_with_offset(cursor::WrappedCursor, offset) = WrappedCursor(cursor.anchor, offset)
_with_offset(cursor::StagedCursor, offset) = StagedCursor(cursor.words, cursor.base, offset)
_anchor_ordinal(cursor::WrappedCursor) = rngposition(cursor.anchor) % UInt64
_anchor_ordinal(cursor::StagedCursor) = cursor.base

const _EngineCursor = Union{WrappedCursor,StagedCursor}

function PureRNGs._take_bits(::WrappedEngine, cursor::_EngineCursor, ::Val{W}) where {W}
    index, shift = divrem(cursor.offset, UInt64(64))
    word = _wrapped_word(cursor, index) << shift
    iszero(shift) || (word |= _wrapped_word(cursor, index + 1) >> (64 - shift))
    return word >> (64 - W), _with_offset(cursor, cursor.offset + UInt64(W))
end

PureRNGs._skip_takes(
    ::WrappedEngine,
    cursor::_EngineCursor,
    count::Integer,
    ::Val{W},
) where {W} = _with_offset(cursor, cursor.offset + UInt64(count) * UInt64(W))
PureRNGs._cursor_ordinal(::WrappedEngine, cursor::_EngineCursor) =
    _anchor_ordinal(cursor) + cursor.offset

PureRNGs._engine_backend(engine::WrappedEngine) = PureRNGs._engine_backend(engine.inner)

# Public draws advance the wrapped generator bit by bit, so exhaustion throws
# where the wrapped generator's own draw would.
function _advance(rng, bits::Integer)
    for _ = 1:(bits÷64)
        _, rng = rand_next(rng, UInt64)
    end
    for _ = 1:(bits%64)
        _, rng = rand_next(rng, Bool)
    end
    return rng
end

function PureRNGs._draw_cursor(engine::WrappedEngine, count::Integer, ::Val{W}) where {W}
    next = _advance(engine.inner, count * W)
    return WrappedCursor(engine.inner, UInt64(0)), WrappedEngine(next)
end

function PureRNGs._addressed_state(
    engine::WrappedEngine,
    count::Integer,
    ::Val{W},
    i::Integer,
) where {W}
    i >= 1 || throw(ArgumentError("addressed draw index must be positive"))
    return WrappedEngine(_advance(engine.inner, (i - 1) * count * W))
end

PureRNGs._fill_cursor(
    engine::WrappedEngine,
    count::Integer,
    ::Val{W},
    ordinal::UInt64,
) where {W} = WrappedCursor(engine.inner, ordinal * UInt64(count) * UInt64(W))

# The engine's own bulk fill: stage every word the fill reads, then run the
# codec over the staged words. A device destination runs the wrapped
# generator's fill, which keeps the backend's checks and kernels in play.
function PureRNGs._engine_fill!(
    engine::WrappedEngine,
    destination::AbstractArray{T},
    threaded::Bool,
    codec,
) where {T}
    if !(PureRNGs._engine_backend(engine) isa PureRNGs._CPUBackend)
        _, next = PureRNGs._engine_fill!(engine.inner, destination, threaded, codec)
        return destination, WrappedEngine(next)
    end
    count, width = PureRNGs._codec_takes(codec, T)
    span = UInt64(count) * UInt64(PureRNGs._val_count(width))
    bits = span * UInt64(length(destination))
    _, next = PureRNGs._draw_cursor(engine, count * length(destination), width)
    words, _ = rand_next(engine.inner, UInt64, Int(cld(bits, 64)) + 1)
    base = rngposition(engine.inner) % UInt64
    fill_range!(first, last) =
        foldl(first:last; init = StagedCursor(words, base, (first - 1) * span)) do cursor, i
            value, cursor = PureRNGs._codec_take(codec, engine, cursor, T)
            destination[i] = value
            cursor
        end
    if threaded
        chunk = cld(length(destination), Threads.nthreads())
        Threads.@threads for first = 1:chunk:length(destination)
            fill_range!(first, min(first + chunk - 1, length(destination)))
        end
    else
        fill_range!(1, length(destination))
    end
    return destination, next
end

# The forwarding methods an engine package defines for its own type.
for f in (:randn_next, :randn_next!, :randn_at, :randexp_next, :randexp_next!, :randexp_at)
    body = Symbol(:_engine_, f)
    @eval PureRNGs.$f(engine::WrappedEngine, args...; kwargs...) =
        PureRNGs.$body(engine, args...; kwargs...)
end
PureRNGs.rand_next(engine::WrappedEngine, d; kwargs...) =
    PureRNGs._engine_rand_next(engine, d; kwargs...)
PureRNGs.rand_next(engine::WrappedEngine, d, dims::Integer...; kwargs...) =
    PureRNGs._engine_rand_next(engine, d, dims...; kwargs...)
PureRNGs.rand_next!(engine::WrappedEngine, d, destination; kwargs...) =
    PureRNGs._engine_rand_next!(engine, d, destination; kwargs...)
PureRNGs.rand_at(engine::WrappedEngine, d, index::Integer) =
    PureRNGs._engine_rand_at(engine, d, index)

# The public methods the Gamma fallback calls on an engine.
function PureRNGs.rand_next(engine::WrappedEngine, ::Type{UInt64})
    value, next = rand_next(engine.inner, UInt64)
    return value, WrappedEngine(next)
end
PureRNGs.subrng(engine::WrappedEngine, purpose::Integer) =
    WrappedEngine(subrng(engine.inner, purpose))

# The pair a draw returns, with the engine replaced by the generator it wraps.
unwrap((value, engine)::Tuple{Any,WrappedEngine}) = (value, engine.inner)
