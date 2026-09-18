module PureRNGsReactantExt

import PureRNGs
import Random
import Reactant

const IR = PureRNGs
const Ops = Reactant.Ops
const _ReactantRNG = IR._ReactantRNG
const _TracedNumber = Reactant.TracedRNumber
const _TracedArray = Reactant.TracedRArray

# `B` says whether a fill places an optimization barrier at the core's
# checkpoints. XLA's GPU backend fuses every core well and a barrier only adds
# a pass over the round state, two to three times the fill time. The CPU
# backend needs the barriers for two cores: without them it split a
# Threefry4x32 fill into fifty kernels and ran a ChaCha fill fifty times
# slower. The choice follows the default client at trace time.
struct _ReactantWordOps{B} end
struct _ReactantTransformOps end

@inline function _lane_barriers(::Type{R}) where {R}
    R <: Union{IR.Threefry4x32,IR.ChaCha} || return false
    return Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == "cpu"
end

# Every helper emits one stablehlo op on scalars. Broadcasting on traced
# arrays and casts through `T.(x)` trace a private function and a call at each
# site, and those dominated the module size: a single Philox4x32 draw traced
# 4500 lines. Scalars also fuse best: XLA's CPU backend runs each loop fusion
# as a separate kernel, and a two-lane vector form of the same core split a
# draw into several fusions and ran up to six times slower in a chain.
@inline _constant_like(::_TracedNumber{T}, value) where {T} = Ops.constant(T(value))
# A broadcast scalar, not a dense constant: dense constants embed one value
# per element in the module and are capped at 100 MB.
@inline _constant_like(x::_TracedArray{T}, value) where {T} =
    Reactant.broadcast_to_size(Ops.constant(T(value)), size(x))

@inline _convert(::Type{T}, x::_TracedNumber{T}) where {T} = x
@inline _convert(::Type{T}, x::_TracedNumber) where {T} = Ops.convert(_TracedNumber{T}, x)
@inline _convert(::Type{T}, x::_TracedArray{T}) where {T} = x
@inline _convert(::Type{T}, x::_TracedArray{S,N}) where {T,S,N} =
    Ops.convert(_TracedArray{T,N}, x)
@inline _bitcast(::Type{T}, x::_TracedNumber) where {T} = Ops.bitcast_convert(T, x)
@inline _bitcast(::Type{T}, x::_TracedArray{S,N}) where {T,S,N} =
    Ops.bitcast_convert(_TracedArray{T,N}, x)

@inline _shift_amount(x, count::Integer) = _constant_like(x, count)
@inline _shift_amount(x, count::Union{_TracedNumber,_TracedArray}) = count
@inline _shl(x, count) = Ops.shift_left(x, _shift_amount(x, count))
@inline _shr(x, count) = Ops.shift_right_logical(x, _shift_amount(x, count))
@inline _rotate(x, count::Int, ::Val{W}) where {W} =
    Ops.or(_shl(x, count), _shr(x, W - count))

@inline _vector(x::_TracedNumber) = Reactant.broadcast_to_size(x, (1,))
@inline _lanes(x::_TracedNumber, ::Val{N}) where {N} = Reactant.broadcast_to_size(x, (N,))

# A traced vector with one value per draw of a fill. The scalar draw and
# transform code below runs on it unchanged: each operation maps to one
# stablehlo op over the vector, and plain numbers lift to constant vectors.
struct _Lane{A<:_TracedArray}
    data::A
end

@inline _lift(x::_Lane, value) = _Lane(_constant_like(x.data, value))
@inline _lift(x::_Lane, ::Type{T}, value) where {T} =
    _Lane(Reactant.broadcast_to_size(Ops.constant(T(value)), size(x.data)))
@inline _convert(::Type{T}, x::_Lane) where {T} = _Lane(_convert(T, x.data))
@inline _bitcast(::Type{T}, x::_Lane) where {T} = _Lane(_bitcast(T, x.data))
@inline _shift_amount(x::_Lane, count::Integer) = _lift(x, count)
@inline _shift_amount(::_Lane, count::_Lane) = count
@inline _shl(x::_Lane, count) = _Lane(Ops.shift_left(x.data, _shift_amount(x, count).data))
@inline _shr(x::_Lane, count) =
    _Lane(Ops.shift_right_logical(x.data, _shift_amount(x, count).data))

for (op, hlo) in (
    (:+, :add),
    (:-, :subtract),
    (:*, :multiply),
    (:/, :divide),
    (:&, :and),
    (:|, :or),
    (:xor, :xor),
)
    @eval begin
        @inline Base.$op(a::_Lane, b::_Lane) = _Lane(Ops.$hlo(a.data, b.data))
        @inline Base.$op(a::_Lane, b::Number) = $op(a, _lift(a, b))
        @inline Base.$op(a::Number, b::_Lane) = $op(_lift(b, a), b)
    end
end
for f in (:sqrt, :log, :abs)
    @eval @inline Base.$f(a::_Lane) = _Lane(Ops.$f(a.data))
end
# The mixed forms take `Real`, not `Number`: Reactant owns the same comparisons
# for `(Any, TracedRNumber)` and `(TracedRNumber, Any)`, and `TracedRNumber` is
# a `Number` but not a `Real`. A `Number` bound would make all eight pairs
# ambiguous. Only a real scalar ever lifts into a lane.
for (op, direction) in ((:<, "LT"), (:<=, "LE"), (:>, "GT"), (:(==), "EQ"))
    @eval begin
        @inline Base.$op(a::_Lane, b::_Lane) =
            _Lane(Ops.compare(a.data, b.data; comparison_direction = $direction))
        @inline Base.$op(a::_Lane, b::Real) = $op(a, _lift(a, b))
        @inline Base.$op(a::Real, b::_Lane) = $op(_lift(b, a), b)
    end
end
@inline Base.:-(a::_Lane) = _Lane(Ops.negate(a.data))
@inline Base.ifelse(pred::_Lane, a::_Lane, b::_Lane) =
    _Lane(Ops.select(pred.data, a.data, b.data))
@inline Base.ifelse(pred::_Lane, a::T, b::T) where {T<:Number} =
    ifelse(pred, _lift(pred, T, a), _lift(pred, T, b))
@inline Base.ifelse(pred::_Lane, a::_Lane, b::Number) = ifelse(pred, a, _lift(a, b))
@inline Base.ifelse(pred::_Lane, a::Number, b::_Lane) = ifelse(pred, _lift(b, a), b)
@inline Base.muladd(a::_Lane, b, c) = a * b + c
@inline Base.muladd(a::Number, b::_Lane, c) = a * b + c
@inline Base.muladd(a::Number, b::Number, c::_Lane) = a * b + c
# The same form as Reactant's scalar `signbit` and `copysign`: the sign test
# through the integer bits is opaque to XLA, so `_rounded_product` keeps the
# product's rounding out of the next operation in fills as well.
@inline _signed_bits(::Type{Float32}) = Int32
@inline _signed_bits(::Type{Float64}) = Int64
@inline Base.signbit(a::_Lane{<:_TracedArray{T}}) where {T<:AbstractFloat} =
    _bitcast(_signed_bits(T), a) < 0
@inline Base.copysign(a::_Lane, sign::_Lane) =
    ifelse(signbit(sign), -one(a), one(a)) * abs(a)
@inline Base.one(a::_Lane) = _lift(a, 1)

@inline IR._word_constant(::_ReactantWordOps, ::Val{W}, anchor, value) where {W} =
    _constant_like(anchor, value)
@inline IR._word_from_value(::_ReactantWordOps, ::Val{32}, anchor, value::_TracedNumber) =
    _convert(UInt32, value)
@inline IR._word_from_value(::_ReactantWordOps, ::Val{64}, anchor, value::_TracedNumber) =
    _convert(UInt64, value)
@inline IR._word_from_value(::_ReactantWordOps, ::Val{32}, anchor, value::Integer) =
    _constant_like(anchor, value % UInt32)
@inline IR._word_from_value(::_ReactantWordOps, ::Val{64}, anchor, value::Integer) =
    _constant_like(anchor, value % UInt64)
@inline IR._word_add(::_ReactantWordOps, ::Val{W}, a, b) where {W} = Ops.add(a, b)
@inline IR._word_xor(::_ReactantWordOps, ::Val{W}, a, b) where {W} = Ops.xor(a, b)
@inline IR._word_rotate(::_ReactantWordOps, ::Val{W}, value, count) where {W} =
    _rotate(value, count, Val(W))
# In a fill the core runs over one lane per block. The barrier materializes
# the state at the core's checkpoints, see `_lane_barriers`.
@inline function IR._word_checkpoint(
    ::_ReactantWordOps{true},
    words::Tuple{Vararg{IR._CoreWord{W,_ReactantWordOps{true},<:_TracedArray}}},
) where {W}
    return map(words) do word
        value = only(Ops.optimization_barrier(word.value))
        IR._core_word(Val(W), _ReactantWordOps{true}(), value)
    end
end
@inline IR._transform_muladd(::_ReactantTransformOps, a, b, c) = muladd(a, b, c)

@inline function IR._word_mulhilo(::_ReactantWordOps, ::Val{32}, a, b)
    product = Ops.multiply(_convert(UInt64, a), _convert(UInt64, b))
    return _convert(UInt32, _shr(product, 32)), _convert(UInt32, product)
end

# XLA has no 128-bit integers, so the 64-bit product keeps the four-product form.
@inline function IR._word_mulhilo(::_ReactantWordOps, ::Val{64}, a, b)
    mask = _constant_like(a, 0xffffffff)
    alo, ahi = Ops.and(a, mask), _shr(a, 32)
    blo, bhi = Ops.and(b, mask), _shr(b, 32)
    p0 = Ops.multiply(alo, blo)
    p1 = Ops.multiply(ahi, blo)
    p2 = Ops.multiply(alo, bhi)
    p3 = Ops.multiply(ahi, bhi)
    middle = Ops.add(Ops.add(_shr(p0, 32), Ops.and(p1, mask)), Ops.and(p2, mask))
    hi = Ops.add(Ops.add(Ops.add(p3, _shr(p1, 32)), _shr(p2, 32)), _shr(middle, 32))
    lo = Ops.add(Ops.add(p0, _shl(p1, 32)), _shl(p2, 32))
    return hi, lo
end

@inline function _state_value(rng::_ReactantRNG, index::Int)
    return Reactant.@allowscalar rng.state[index]
end

@inline _bitcast_signed(::Type{T}, value) where {T<:Signed} =
    _bitcast(T, _convert(unsigned(T), value))

# The state holds the key words and the position. An exhausted generator
# encodes the address after its last block, where a compiled continuation
# lands. The state does not carry the decoded block: carried words become
# separate values at every draw boundary, and XLA's CPU backend then splits a
# chain of draws into hundreds of small fusions.
@inline function _encode_state(rng::R) where {R<:IR.AbstractPureRNG}
    key = UInt64[rng.key...]
    position = rng.position
    if position isa IR._Position64
        if position.bit == IR._EXHAUSTED_BIT
            maximum = IR._max_block(rng)
            append!(
                key,
                (maximum + UInt64(1), UInt64(iszero(maximum + UInt64(1))), UInt64(0)),
            )
        else
            append!(key, (position.block, UInt64(0), UInt64(position.bit)))
        end
    else
        if position.bit == IR._EXHAUSTED_BIT
            append!(key, (UInt64(0), UInt64(0), UInt64(1), UInt64(0)))
        else
            append!(key, (position.lo, position.hi, UInt64(0), UInt64(position.bit)))
        end
    end
    return key
end

"""
    Reactant.to_rarray(rng::AbstractPureRNG)

Move `rng` into a compiled carrier holding its key and bit position.

The carrier omits the counter-capacity check that eager generators apply, so a
compiled draw past the per-key capacity wraps instead of throwing. Keep every
compiled draw within capacity. The "Differentiation and compilation" page of
the documentation states the full compiled contract.
"""
function Reactant.to_rarray(rng::R) where {R<:IR.AbstractPureRNG}
    state = Reactant.to_rarray(_encode_state(rng))
    return _ReactantRNG{R,typeof(state)}(state)
end

@inline _key_count(::Type{R}) where {R} = fieldcount(fieldtype(R, :key))
@inline _key_type(::Type{R}) where {R} = fieldtype(fieldtype(R, :key), 1)
@inline _word_width(::Type{R}) where {R} = Val(8 * sizeof(_key_type(R)))
@inline _position128(::Type{R}) where {R} = fieldtype(R, :position) === IR._Position128
@inline _block_bits(::Type{R}) where {R} =
    R <: Union{IR.Philox2x32,IR.Threefry2x32} ? UInt64(64) :
    R <: Union{IR.Philox4x64,IR.Threefry4x64} ? UInt64(256) :
    R <: IR.ChaCha ? UInt64(512) : UInt64(128)

# `barriers` carries `_lane_barriers` as a type. Every trace entry point
# resolves it once and passes it down, so the word type is a constant here
# while the choice still follows the default client of that trace.
@inline _core_word(::Type{R}, ::Val{B}, value) where {R,B} =
    IR._core_word(_word_width(R), _ReactantWordOps{B}(), value)

@inline function _key(rng::_ReactantRNG{R}, barriers::Val) where {R}
    T = _key_type(R)
    return ntuple(
        index -> _core_word(R, barriers, _convert(T, _state_value(rng, index))),
        Val(_key_count(R)),
    )
end

@inline function _position(rng::_ReactantRNG{R}) where {R}
    offset = _key_count(R)
    if _position128(R)
        return _state_value(rng, offset + 1),
        _state_value(rng, offset + 2),
        _state_value(rng, offset + 3),
        _state_value(rng, offset + 4)
    end
    return _state_value(rng, offset + 1),
    _state_value(rng, offset + 2),
    _state_value(rng, offset + 3)
end

@inline _unwrap(words) = map(word -> word.value, words)

# Block words of the block at address `(lo, hi)`. The draw counter is the
# block address zero-extended to the core's counter width, matching `_block`
# in src/bits.jl, and 32-bit outputs pair into 64-bit block words as in
# `_block_words` there.
@inline function _core_words(::Type{R}, barriers::Val, key, lo, hi) where {R}
    word(value) = _core_word(R, barriers, value)
    pad = IR._core_constant(key[1], 0)
    rounds = Val(IR._rounds(R))
    if R <: Union{IR.Philox2x32,IR.Threefry2x32}
        low = word(_convert(UInt32, lo))
        high = word(
            Ops.and(_convert(UInt32, _shr(lo, 32)), _constant_like(low.value, 0x00ffffff)),
        )
        words =
            R <: IR.Philox2x32 ? IR._philox2x32((low, high), key, rounds) :
            IR._threefry2x32((low, high), key, rounds)
    elseif R <: Union{IR.Philox4x32,IR.Threefry4x32,IR.ChaCha}
        counter =
            (word(_convert(UInt32, lo)), word(_convert(UInt32, _shr(lo, 32))), pad, pad)
        words =
            R <: IR.Philox4x32 ? IR._philox4x32(counter, key, rounds) :
            R <: IR.Threefry4x32 ? IR._threefry4x32(counter, key, rounds) :
            IR._chacha(counter, key, rounds)
    elseif R <: Union{IR.Philox2x64,IR.Threefry2x64}
        counter = (word(lo), pad)
        words =
            R <: IR.Philox2x64 ? IR._philox2x64(counter, key, rounds) :
            IR._threefry2x64(counter, key, rounds)
    else
        counter = (word(lo), word(hi), pad, pad)
        words =
            R <: IR.Philox4x64 ? IR._philox4x64(counter, key, rounds) :
            IR._threefry4x64(counter, key, rounds)
    end
    return _block_words(R, _unwrap(words))
end

@inline function _block_words(::Type{R}, words) where {R}
    _word_width(R) isa Val{32} || return words
    return ntuple(
        i -> Ops.or(_shl(_convert(UInt64, words[2i-1]), 32), _convert(UInt64, words[2i])),
        Val(length(words) ÷ 2),
    )
end

# Every block in a compiled function calls one shared MLIR function, so the
# module holds one copy of the rounds however many draws it chains. The MLIR
# inliner restores the inline form before XLA sees the module, so the kernels
# are unchanged, while the passes that scale with module size run on the
# shared body.
function _core_body(::Type{R}, barriers::Val, lo, hi, key_values...) where {R}
    key = map(value -> _core_word(R, barriers, value), key_values)
    return _core_words(R, barriers, key, lo, hi)
end

@inline function _shared_words(::Type{R}, barriers::Val, key, lo, hi) where {R}
    return Ops.call(_core_body, R, barriers, lo, hi, _unwrap(key)...)
end

# The block at the position and its successor, in stream order. A draw may
# straddle the two, and the traced program has no branch to skip the second.
@inline function _window_words(rng::_ReactantRNG{R}, position, barriers::Val) where {R}
    key = _key(rng, barriers)
    lo = position[1]
    hi = _position128(R) ? position[2] : lo
    next_lo = lo + UInt64(1)
    next_hi = _position128(R) ? hi + ifelse(iszero(next_lo), UInt64(1), UInt64(0)) : next_lo
    return (
        _shared_words(R, barriers, key, lo, hi)...,
        _shared_words(R, barriers, key, next_lo, next_hi)...,
    )
end

# `values[index + 1]` through a select chain. A dynamic slice would trace
# fewer ops, but XLA's CPU backend runs each one outside the loop fusions.
@inline function _select(values, index)
    result = values[1]
    for i = 2:length(values)
        result = ifelse(index == UInt64(i - 1), values[i], result)
    end
    return result
end

# `W` bits from bit `word_bit` of `first`, continuing into `second` when the
# draw straddles the word boundary. A shift by 64 or more yields zero.
@inline function _extract(first, second, word_bit, ::Val{W}) where {W}
    mask = W == 64 ? typemax(UInt64) : (UInt64(1) << W) - UInt64(1)
    available = UInt64(64) - word_bit
    single = _shr(first, available - UInt64(W)) & mask
    remaining = UInt64(W) - available
    crossed = (_shl(first, remaining) | _shr(second, UInt64(64) - remaining)) & mask
    return ifelse(UInt64(W) <= available, single, crossed)
end

@inline function _raw(rng::_ReactantRNG{R}, ::Val{W}) where {R,W}
    position = _position(rng)
    bit = position[end]
    words = _window_words(rng, position, Val(_lane_barriers(R)))
    lane = _shr(bit, 6)
    first = _select(words, lane)
    second = _select(words[2:end], lane)
    return _extract(first, second, bit & UInt64(63), Val(W))
end

@inline _convert_result(::Type{Bool}, raw) = raw == UInt64(1)
@inline _convert_result(::Type{UInt32}, raw) = _convert(UInt32, raw)
@inline _convert_result(::Type{UInt64}, raw) = raw
@inline _convert_result(::Type{Int32}, raw) = _bitcast_signed(Int32, raw)
@inline _convert_result(::Type{Int64}, raw) = _bitcast_signed(Int64, raw)
@inline _convert_result(::Type{Float32}, raw) = _convert(Float32, raw) * Float32(0x1p-24)
@inline _convert_result(::Type{Float64}, raw) = _convert(Float64, raw) * Float64(0x1p-53)

@inline function _draw(rng::_ReactantRNG, ::Type{T}) where {T}
    return _convert_result(T, _raw(rng, Val(IR._draw_bits(T))))
end

@inline function _advance(
    rng::_ReactantRNG{R},
    bits_lo::UInt64,
    bits_hi::UInt64 = UInt64(0),
    bits_top::UInt64 = UInt64(0),
) where {R}
    position = _position(rng)
    block_bits = _block_bits(R)
    shift = trailing_zeros(block_bits)
    sum_lo = bits_lo + position[end]
    sum_hi = bits_hi + ifelse(sum_lo < bits_lo, UInt64(1), UInt64(0))
    sum_top = bits_top + ifelse(sum_hi < bits_hi, UInt64(1), UInt64(0))
    bit = sum_lo & (block_bits - UInt64(1))
    block_lo = _shr(sum_lo, shift) | _shl(sum_hi, 64 - shift)
    block_hi = _shr(sum_hi, shift) | _shl(sum_top, 64 - shift)
    block_top = _shr(sum_top, shift)

    lo = position[1] + block_lo
    carry = ifelse(lo < position[1], UInt64(1), UInt64(0))
    key = Ops.slice(rng.state, [1], [_key_count(R)])
    if _position128(R)
        partial_hi = position[2] + block_hi
        top = position[3] + block_top
        top += ifelse(partial_hi < position[2], UInt64(1), UInt64(0))
        hi = partial_hi + carry
        top += ifelse(hi < partial_hi, UInt64(1), UInt64(0))
        pieces = [key, _vector(lo), _vector(hi), _vector(top), _vector(bit)]
    else
        hi = position[2] + block_hi + carry
        pieces = [key, _vector(lo), _vector(hi), _vector(bit)]
    end
    state = Ops.concatenate(pieces, 1)
    return _ReactantRNG{R,typeof(state)}(state)
end

@inline function _address_offset(index::Integer, width::UInt64)
    index > 0 || throw(ArgumentError("i must be positive"))
    bits = BigInt(index - 1) * BigInt(width)
    mask = BigInt(typemax(UInt64))
    return (UInt64(bits & mask), UInt64((bits >> 64) & mask), UInt64((bits >> 128) & mask))
end

@inline function IR._addressed_rng(rng::_ReactantRNG, width::UInt16, index::Integer)
    bits_lo, bits_hi, bits_top = _address_offset(index, UInt64(width))
    return _advance(rng, bits_lo, bits_hi, bits_top)
end

for T in (Bool, UInt32, UInt64, Int32, Int64, Float32, Float64)
    @eval begin
        @inline Random.rand(rng::_ReactantRNG, ::Type{$T}) = _draw(rng, $T)
        @inline function IR.rand_next(rng::_ReactantRNG, ::Type{$T})
            return _draw(rng, $T), _advance(rng, UInt64(IR._draw_bits($T)))
        end
        @inline function IR.randat(rng::_ReactantRNG, ::Type{$T}, index::Integer)
            return _draw(IR._addressed_rng(rng, IR._draw_bits($T), index), $T)
        end
    end
end

@inline function _horner(x, coefficients)
    value = coefficients[1]
    for index = 2:length(coefficients)
        value = muladd(value, x, coefficients[index])
    end
    return value
end

@inline function _normal_transform(u, ::Type{T}) where {T}
    A, B, C, D, E, F = IR._as241_coefficients(T)
    q = u - T(0.5)
    central_r = T(0.180625) - IR._rounded_product(q, q, q)
    central = q * (_horner(central_r, A) / _horner(central_r, B))
    tail_r = sqrt(-log(ifelse(q < zero(T), u, one(T) - u)))
    lower_r = tail_r - T(1.6)
    upper_r = tail_r - T(5)
    lower = _horner(lower_r, C) / _horner(lower_r, D)
    upper = _horner(upper_r, E) / _horner(upper_r, F)
    tail = ifelse(tail_r <= T(5), lower, upper)
    tail = ifelse(q < zero(T), -tail, tail)
    return ifelse(abs(q) <= T(0.425), central, tail)
end

# A Float32 midpoint caps the tail radius at r = 4.08, so the far-tail branch of
# the generic method never fires. Omitting it drops two traced polynomials.
@inline function _normal_transform(u, ::Type{Float32})
    A, B, C, D, _, _ = IR._as241_coefficients(Float32)
    q = u - 0.5f0
    central_r = 0.180625f0 - IR._rounded_product(q, q, q)
    central = q * (_horner(central_r, A) / _horner(central_r, B))
    tail_r = sqrt(-log(ifelse(q < 0.0f0, u, 1.0f0 - u))) - 1.6f0
    tail = _horner(tail_r, C) / _horner(tail_r, D)
    tail = ifelse(q < 0.0f0, -tail, tail)
    return ifelse(abs(q) <= 0.425f0, central, tail)
end

@inline function _midpoint_from_raw(raw, ::Type{T}) where {T<:Union{Float32,Float64}}
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    return _convert(T, (raw * UInt64(2)) | UInt64(1)) * scale
end

@inline _normal_from_raw(raw, ::Type{T}) where {T<:Union{Float32,Float64}} =
    _normal_transform(_midpoint_from_raw(raw, T), T)

@inline IR._midpoint_value(rng::_ReactantRNG, ::Type{T}) where {T<:Union{Float32,Float64}} =
    _midpoint_from_raw(_raw(rng, Val(IR._normal_bits(T))), T)

@inline _normal_value(rng, ::Type{T}) where {T} =
    _normal_from_raw(_raw(rng, Val(IR._normal_bits(T))), T)

# Field layout of the binary float: bits type, mantissa width, exponent mask,
# exponent bias, mantissa mask, and the bits of one.
@inline _float_layout(::Type{Float32}) =
    (UInt32, 23, UInt32(0xff), Int32(127), UInt32(0x007fffff), UInt32(0x3f800000))
@inline _float_layout(::Type{Float64}) = (
    UInt64,
    52,
    UInt64(0x7ff),
    Int64(1023),
    UInt64(0x000fffffffffffff),
    UInt64(0x3ff0000000000000),
)

# Mirrors `_exponential_transform` for the CPU backend in src/exponential.jl.
@inline function _exponential_transform(value, ::Type{T}) where {T}
    sqrt2 =
        T === Float32 ? reinterpret(Float32, UInt32(0x3fb504f3)) :
        reinterpret(Float64, UInt64(0x3ff6a09e667f3bcd))
    half =
        T === Float32 ? reinterpret(Float32, UInt32(0x3f000000)) :
        reinterpret(Float64, UInt64(0x3fe0000000000000))
    B, width, exponent_mask, bias, mantissa_mask, one_bits = _float_layout(T)
    bits = _bitcast(B, value)
    exponent = _convert(typeof(bias), _shr(bits, width) & exponent_mask) - bias
    mantissa = _bitcast(T, (bits & mantissa_mask) | one_bits)
    upper = mantissa > sqrt2
    mantissa = ifelse(upper, mantissa * half, mantissa)
    exponent += ifelse(upper, one(bias), zero(bias))
    n = -_convert(T, exponent)
    return IR._exponential_reduced(_ReactantTransformOps(), T, mantissa, n)
end

@inline function _exponential_from_raw(raw, ::Type{T}) where {T}
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = _convert(T, raw) * scale
    return _exponential_transform(one(T) - u, T)
end

@inline _exponential_value(rng, ::Type{T}) where {T} =
    _exponential_from_raw(_raw(rng, Val(IR._exponential_bits(T))), T)

for T in (Float32, Float64)
    @eval begin
        @inline Random.randn(rng::_ReactantRNG, ::Type{$T}) = _normal_value(rng, $T)
        @inline function IR.randn_next(rng::_ReactantRNG, ::Type{$T})
            return _normal_value(rng, $T), _advance(rng, UInt64(IR._normal_bits($T)))
        end
        @inline function IR.randnat(rng::_ReactantRNG, ::Type{$T}, index::Integer)
            return _normal_value(IR._addressed_rng(rng, IR._normal_bits($T), index), $T)
        end
        @inline Random.randexp(rng::_ReactantRNG, ::Type{$T}) = _exponential_value(rng, $T)
        @inline function IR.randexp_next(rng::_ReactantRNG, ::Type{$T})
            return _exponential_value(rng, $T),
            _advance(rng, UInt64(IR._exponential_bits($T)))
        end
        @inline function IR.randexpat(rng::_ReactantRNG, ::Type{$T}, index::Integer)
            return _exponential_value(
                IR._addressed_rng(rng, IR._exponential_bits($T), index),
                $T,
            )
        end
    end
end

# Array fills. The core runs once over every block the fill spans, one lane
# per block, and a gather reads the two words of each draw. The stream layout
# matches the CPU fill: draw `i` occupies the `W` bits after `i - 1` draws.
@inline _fill_blocks(::Type{R}, n::Int, width::Int) where {R} =
    cld(Int(_block_bits(R)) - 1 + n * width, Int(_block_bits(R))) + 1

# The block words of `B` consecutive blocks from the position, as one vector
# in stream order.
function _stream_words(rng::_ReactantRNG{R}, position, ::Val{B}, barriers::Val) where {R,B}
    lo = _lanes(position[1], Val(B))
    los = Ops.add(lo, Ops.iota(UInt64, [B]; iota_dimension = 1))
    his = if _position128(R)
        carry = _convert(UInt64, Ops.compare(los, lo; comparison_direction = "LT"))
        Ops.add(_lanes(position[2], Val(B)), carry)
    else
        los
    end
    key = map(
        word -> _core_word(R, barriers, _lanes(word.value, Val(B))),
        _key(rng, barriers),
    )
    words = _core_words(R, barriers, key, los, his)
    rows = [Ops.broadcast_in_dim(word, [2], [1, B]) for word in words]
    return Ops.reshape(Ops.concatenate(rows, 1), length(words) * B)
end

function _fill_raw(rng::_ReactantRNG{R}, ::Val{W}, n::Int) where {R,W}
    position = _position(rng)
    flat = _stream_words(
        rng,
        position,
        Val(_fill_blocks(R, n, Int(W))),
        Val(_lane_barriers(R)),
    )
    offsets = Ops.multiply(
        Ops.iota(UInt64, [n]; iota_dimension = 1),
        _lanes(Ops.constant(UInt64(W)), Val(n)),
    )
    starts = _Lane(Ops.add(offsets, _lanes(position[end], Val(n))))
    lane = _shr(starts, 6)
    first = _Lane(flat[(lane+UInt64(1)).data])
    second = _Lane(flat[(lane+UInt64(2)).data])
    return _extract(first, second, starts & UInt64(63), Val(W))
end

function _fill_next(rng::_ReactantRNG, ::Val{W}, dims::Dims, finish) where {W}
    n = prod(dims)
    values = finish(_fill_raw(rng, Val(W), n)).data
    array = length(dims) == 1 ? values : Ops.reshape(values, dims...)
    return array, _advance(rng, _address_offset(n + 1, UInt64(W))...)
end

for (draw, draw_next, bits, finish, types) in (
    (
        :(Random.rand),
        :(IR.rand_next),
        :(IR._draw_bits),
        T -> :(raw -> _convert_result($T, raw)),
        (Bool, UInt32, UInt64, Int32, Int64, Float32, Float64),
    ),
    (
        :(Random.randn),
        :(IR.randn_next),
        :(IR._normal_bits),
        T -> :(raw -> _normal_from_raw(raw, $T)),
        (Float32, Float64),
    ),
    (
        :(Random.randexp),
        :(IR.randexp_next),
        :(IR._exponential_bits),
        T -> :(raw -> _exponential_from_raw(raw, $T)),
        (Float32, Float64),
    ),
)
    for T in types
        @eval begin
            @inline function $draw_next(rng::_ReactantRNG, ::Type{$T}, dims::Dims)
                return _fill_next(rng, Val($bits($T)), dims, $(finish(T)))
            end
            @inline function $draw_next(
                rng::_ReactantRNG,
                ::Type{$T},
                dim1::Integer,
                dims::Integer...,
            )
                return $draw_next(rng, $T, Int.((dim1, dims...)))
            end
            @inline $draw(rng::_ReactantRNG, ::Type{$T}, dims::Dims) =
                first($draw_next(rng, $T, dims))
            @inline function $draw(
                rng::_ReactantRNG,
                ::Type{$T},
                dim1::Integer,
                dims::Integer...,
            )
                return first($draw_next(rng, $T, Int.((dim1, dims...))))
            end
        end
    end
    @eval begin
        @inline $draw_next(rng::_ReactantRNG, dims::Dims) = $draw_next(rng, Float64, dims)
        @inline $draw_next(rng::_ReactantRNG, dim1::Integer, dims::Integer...) =
            $draw_next(rng, Float64, Int.((dim1, dims...)))
    end
end

@inline function _mulhi64(word, span::UInt64)
    high = _shr(word, 32)
    low = word & UInt64(0xffffffff)
    return _shr(high * span + _shr(low * span, 32), 32)
end

@inline function _mulhi128(lo, hi, span::UInt64)
    ops = _ReactantWordOps{false}()
    span_word = Ops.constant(span)
    high_hi, high_lo = IR._word_mulhilo(ops, Val(64), hi, span_word)
    low_hi, _ = IR._word_mulhilo(ops, Val(64), lo, span_word)
    sum = high_lo + low_hi
    return high_hi + ifelse(sum < high_lo, UInt64(1), UInt64(0))
end

@inline function _range_offset(rng, span::UInt64)
    if IR._range_bits(span) == UInt16(64)
        _mulhi64(_raw(rng, Val(64)), span)
    else
        hi = _raw(rng, Val(64))
        lo = _raw(_advance(rng, UInt64(64)), Val(64))
        return iszero(span) ? hi : _mulhi128(lo, hi, span)
    end
end

@inline function _range_element(range::OrdinalRange{T}, offset) where {T}
    bits = (first(range) % UInt64) + offset * (step(range) % UInt64)
    return T <: Signed ? _bitcast_signed(T, bits) : _convert(T, bits)
end

@inline function _range_element(range::LinRange{T}, offset) where {T<:IR._RangeInteger}
    denominator = max(length(range) - 1, 1)
    t = _convert(Float64, offset) / Float64(denominator)
    value = (one(Float64) - t) * first(range) + t * last(range)
    return _convert(T, value)
end

@inline function IR._range_value(
    rng::_ReactantRNG,
    range::Union{OrdinalRange{T},LinRange{T}},
) where {T<:IR._RangeInteger}
    span = length(range) % UInt64
    return _range_element(range, _range_offset(rng, span))
end

@inline function Random.rand(
    rng::_ReactantRNG,
    range::Union{OrdinalRange{T},LinRange{T}},
) where {T<:IR._RangeInteger}
    isempty(range) && throw(ArgumentError("range must be non-empty"))
    return IR._range_value(rng, range)
end

@inline function IR.rand_next(
    rng::_ReactantRNG,
    range::Union{OrdinalRange{T},LinRange{T}},
) where {T<:IR._RangeInteger}
    isempty(range) && throw(ArgumentError("range must be non-empty"))
    width = UInt64(IR._range_bits(length(range) % UInt64))
    return IR._range_value(rng, range), _advance(rng, width)
end

# Range arrays and samples. Each element consumes 64 bits, or 128 bits as a
# high word followed by a low word, and reduces them as the eager fill does.
@inline function _mulhi128(lo::_Lane, hi::_Lane, span::UInt64)
    ops = _ReactantWordOps{false}()
    span_lane = _constant_like(hi.data, span)
    high_hi, high_lo = map(_Lane, IR._word_mulhilo(ops, Val(64), hi.data, span_lane))
    low_hi, _ = map(_Lane, IR._word_mulhilo(ops, Val(64), lo.data, span_lane))
    sum = high_lo + low_hi
    return high_hi + ifelse(sum < high_lo, UInt64(1), UInt64(0))
end

function _fill_offsets(rng::_ReactantRNG, span::UInt64, n::Int)
    IR._range_bits(span) == UInt16(64) && return _mulhi64(_fill_raw(rng, Val(64), n), span)
    words = _fill_raw(rng, Val(64), 2n).data
    hi = _Lane(Ops.slice(words, [1], [2n]; strides = [2]))
    lo = _Lane(Ops.slice(words, [2], [2n]; strides = [2]))
    return iszero(span) ? hi : _mulhi128(lo, hi, span)
end

@inline function _shaped(values::_TracedArray, dims::Dims)
    return length(dims) == 1 ? values : Ops.reshape(values, dims...)
end

function _fill_range_next(rng::_ReactantRNG, range, dims::Dims)
    isempty(range) && throw(ArgumentError("range must be non-empty"))
    span = length(range) % UInt64
    n = prod(dims)
    values = _range_element(range, _fill_offsets(rng, span, n)).data
    advanced = _advance(rng, _address_offset(n + 1, UInt64(IR._range_bits(span)))...)
    return _shaped(values, dims), advanced
end

for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
    @eval begin
        @inline function Random.rand(
            rng::_ReactantRNG,
            range::Union{OrdinalRange{$T},LinRange{$T}},
            dim1::Integer,
            dims::Integer...,
        )
            return first(IR.rand_next(rng, range, dim1, dims...))
        end
        @inline function IR.rand_next(
            rng::_ReactantRNG,
            range::Union{OrdinalRange{$T},LinRange{$T}},
            dim1::Integer,
            dims::Integer...,
        )
            return _fill_range_next(rng, range, Int.((dim1, dims...)))
        end
        @inline Random.rand(
            rng::_ReactantRNG,
            range::Union{OrdinalRange{$T},LinRange{$T}},
            dims::Dims,
        ) = first(_fill_range_next(rng, range, dims))
        @inline IR.rand_next(
            rng::_ReactantRNG,
            range::Union{OrdinalRange{$T},LinRange{$T}},
            dims::Dims,
        ) = _fill_range_next(rng, range, dims)
    end
end

# The draws at consecutive addresses are the fill that starts at the first one.
@inline function _addressed_array(
    rng::_ReactantRNG,
    ::Type{T},
    indices::AbstractUnitRange,
    width::UInt16,
    fill_next,
) where {T}
    isempty(indices) && return Ops.constant(T[])
    start = IR._addressed_rng(rng, width, first(indices))
    return first(fill_next(start, T, (length(indices),)))
end

for (at, fill_next, bits, types) in (
    (
        :(IR.randat),
        :(IR.rand_next),
        :(IR._draw_bits),
        (Bool, UInt32, UInt64, Int32, Int64, Float32, Float64),
    ),
    (:(IR.randnat), :(IR.randn_next), :(IR._normal_bits), (Float32, Float64)),
    (:(IR.randexpat), :(IR.randexp_next), :(IR._exponential_bits), (Float32, Float64)),
)
    for T in types
        @eval @inline function $at(
            rng::_ReactantRNG,
            ::Type{$T},
            indices::AbstractUnitRange{<:Integer},
        )
            return _addressed_array(rng, $T, indices, $bits($T), $fill_next)
        end
    end
end

# Destination fills replace the traced destination's value. The `threaded`
# keyword exists so call sites written for the eager fill trace unchanged.
@inline function _store!(destination::_TracedArray, values)
    Reactant.TracedUtils.set_mlir_data!(destination, values.mlir_data)
    return destination
end

for (fill, fill_next, draw_next, T) in (
    (
        :(Random.rand!),
        :(IR.rand_next!),
        :(IR.rand_next),
        :(Union{Bool,UInt32,UInt64,Int32,Int64,Float32,Float64}),
    ),
    (:(Random.randn!), :(IR.randn_next!), :(IR.randn_next), :(Union{Float32,Float64})),
    (
        :(Random.randexp!),
        :(IR.randexp_next!),
        :(IR.randexp_next),
        :(Union{Float32,Float64}),
    ),
)
    @eval begin
        @inline function $fill_next(
            rng::_ReactantRNG,
            destination::_TracedArray{T};
            threaded::Bool = true,
        ) where {T<:$T}
            values, next_rng = $draw_next(rng, T, size(destination)...)
            return _store!(destination, values), next_rng
        end
        @inline function $fill(
            rng::_ReactantRNG,
            destination::_TracedArray{T};
            threaded::Bool = true,
        ) where {T<:$T}
            return first($fill_next(rng, destination))
        end
    end
end

# Unweighted samples with replacement. Each element is a range draw over the
# population cardinality followed by a gather, as in the eager fill.
@inline function _population_values(
    population::Union{OrdinalRange{T},LinRange{T}},
    ordinals,
) where {T<:IR._RangeInteger}
    return _range_element(population, ordinals - UInt64(1)).data
end

@inline function _population_values(population::AbstractArray, ordinals)
    flat = if population isa _TracedArray
        Ops.reshape(population, length(population))
    else
        Ops.constant(vec(collect(population)))
    end
    return flat[ordinals.data]
end

function _sample_next(rng::_ReactantRNG, population, requested_count)
    cardinality = length(population) % UInt64
    count =
        requested_count === nothing ? Int(cardinality) : IR._sampling_count(requested_count)
    count > 0 && iszero(cardinality) && IR._empty_sampling_population()
    iszero(count) && return Ops.constant(Vector{eltype(population)}()), rng
    width = UInt64(IR._range_bits(cardinality))
    advanced = _advance(rng, _address_offset(count + 1, width)...)
    ordinals = _fill_offsets(rng, cardinality, count) + UInt64(1)
    return _population_values(population, ordinals), advanced
end

@inline IR.randsample(rng::_ReactantRNG, population::AbstractArray) =
    first(_sample_next(rng, population, nothing))
@inline IR.randsample(rng::_ReactantRNG, population::AbstractArray, count::Integer) =
    first(_sample_next(rng, population, count))
@inline IR.randsample_next(rng::_ReactantRNG, population::AbstractArray) =
    _sample_next(rng, population, nothing)
@inline IR.randsample_next(rng::_ReactantRNG, population::AbstractArray, count::Integer) =
    _sample_next(rng, population, count)

# A child starts at position zero: its state is the key words followed by zeros.
@inline function _child(rng::_ReactantRNG{R}, key::Tuple) where {R}
    words = map(word -> _vector(_convert(UInt64, word.value)), key)
    rest = Ops.constant(zeros(UInt64, length(rng.state) - _key_count(R)))
    state = Ops.concatenate([words..., rest], 1)
    return _ReactantRNG{R,typeof(state)}(state)
end

@inline function _derive_child(rng::_ReactantRNG{R}, index::UInt64, barriers::Val) where {R}
    return _child(rng, IR._derive_key(R, _key(rng, barriers), index))
end

IR.splitrng(rng::_ReactantRNG) = IR.splitrng(rng, Val(2))

@inline function IR.splitrng(rng::_ReactantRNG{R}, ::Val{N}) where {R,N}
    (N isa Int && N >= 0) || throw(ArgumentError("N must be a non-negative Int"))
    if R <: Union{IR.Philox2x32,IR.Threefry2x32}
        N <= IR._NARROW_SPLIT_COUNT ||
            throw(ArgumentError("two-word generator child index enters the fold namespace"))
    end
    barriers = Val(_lane_barriers(R))
    return ntuple(index -> _derive_child(rng, UInt64(index - 1), barriers), Val(N))
end

@inline IR.subrng(rng::_ReactantRNG{R}, purpose::Integer) where {R} =
    _purpose_child(rng, purpose % UInt64, Val(_lane_barriers(R)))

@inline _purpose_child(rng::_ReactantRNG{R}, purpose::UInt64, barriers::Val) where {R} =
    _child(rng, IR._subrng_key(R, _key(rng, barriers), purpose))

end
