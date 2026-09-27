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

# The engine's stream is 2^64 bits long. It checks each span from the factors
# the core passes, and records them, so tests can see that no factor wrapped.
const LAST_RESERVATION = Ref{Any}(nothing)
const LAST_ADDRESS = Ref{Any}(nothing)

_check_span(bits::Integer) =
    bits <= typemax(UInt64) || throw(OverflowError("span past the engine's stream"))

function PureRNGs._reserve_draws(
    engine::WrappedEngine,
    draws::Integer,
    count::Integer,
    ::Val{W},
) where {W}
    LAST_RESERVATION[] = (draws, count, W)
    _check_span(BigInt(draws) * count * W)
    return WrappedEngine(_advance(engine.inner, UInt64(draws) * UInt64(count) * UInt64(W)))
end

function PureRNGs._addressed_state(
    engine::WrappedEngine,
    count::Integer,
    ::Val{W},
    i::Integer,
) where {W}
    LAST_ADDRESS[] = (count, W, i)
    i >= 1 || throw(ArgumentError("addressed draw index must be positive"))
    _check_span(BigInt(i) * count * W)
    return WrappedEngine(_advance(engine.inner, UInt64(i - 1) * UInt64(count) * UInt64(W)))
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
    next = PureRNGs._reserve_draws(engine, length(destination), count, width)
    _staged_draws!(engine, length(destination), count, width, threaded) do cursor, i
        value, cursor = PureRNGs._codec_take(codec, engine, cursor, T)
        destination[i] = value
        cursor
    end
    return destination, next
end

# Runs `take(cursor, i)` over draws 1:n in order, or over one chunk per thread,
# with each draw's cursor the one its predecessor returned.
function _staged_draws!(take, engine, n, count, ::Val{W}, threaded) where {W}
    iszero(n) && return nothing
    span = UInt64(count) * UInt64(W)
    words, _ = rand_next(engine.inner, UInt64, Int(cld(span * UInt64(n), 64)) + 1)
    base = rngposition(engine.inner) % UInt64
    run!(first, last) =
        foldl(take, first:last; init = StagedCursor(words, base, (first - 1) * span))
    if threaded
        chunk = cld(n, Threads.nthreads())
        Threads.@threads for first = 1:chunk:n
            run!(first, min(first + chunk - 1, n))
        end
    else
        run!(1, n)
    end
    return nothing
end

# The column fill stages its words the same way. The counter shows the core
# reached this override.
const COLUMN_FILLS = Threads.Atomic{Int}(0)

function PureRNGs._engine_fill_columns!(
    engine::WrappedEngine,
    destination::AbstractMatrix{T},
    threaded::Bool,
    codec,
) where {T}
    if !(PureRNGs._engine_backend(engine) isa PureRNGs._CPUBackend)
        _, next = PureRNGs._engine_fill_columns!(engine.inner, destination, threaded, codec)
        return destination, WrappedEngine(next)
    end
    count, width = PureRNGs._codec_takes(codec, T)
    next = PureRNGs._reserve_draws(engine, size(destination, 2), count, width)
    Threads.atomic_add!(COLUMN_FILLS, 1)
    _staged_draws!(engine, size(destination, 2), count, width, threaded) do cursor, column
        PureRNGs._column_take!(codec, engine, cursor, destination, column)
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
