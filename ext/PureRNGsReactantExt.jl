module PureRNGsReactantExt

import PureRNGs
import Random
import Reactant

const IR = PureRNGs
const Ops = Reactant.Ops
const _ReactantRNG = IR._ReactantRNG
const _TracedNumber = Reactant.TracedRNumber

struct _ReactantWordOps end
struct _ReactantTransformOps end

# Every helper emits one stablehlo op on scalars. Broadcasting on traced
# arrays and casts through `T.(x)` trace a private function and a call at each
# site, and those dominated the module size: a single Philox4x32 draw traced
# 4500 lines. Scalars also fuse best: XLA's CPU backend runs each loop fusion
# as a separate kernel, and a two-lane vector form of the same core split a
# draw into several fusions and ran up to six times slower in a chain.
@inline _constant_like(::_TracedNumber{T}, value) where {T} = Ops.constant(T(value))

@inline _convert(::Type{T}, x::_TracedNumber{T}) where {T} = x
@inline _convert(::Type{T}, x::_TracedNumber) where {T} = Ops.convert(_TracedNumber{T}, x)

@inline _shift_amount(x, count::Integer) = _constant_like(x, count)
@inline _shift_amount(x, count::_TracedNumber) = count
@inline _shl(x, count) = Ops.shift_left(x, _shift_amount(x, count))
@inline _shr(x, count) = Ops.shift_right_logical(x, _shift_amount(x, count))
@inline _rotate(x, count::Int, ::Val{W}) where {W} =
    Ops.or(_shl(x, count), _shr(x, W - count))

@inline _vector(x::_TracedNumber) = Reactant.broadcast_to_size(x, (1,))

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
    Ops.bitcast_convert(T, _convert(unsigned(T), value))

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

@inline _core_word(::Type{R}, value) where {R} =
    IR._core_word(_word_width(R), _ReactantWordOps(), value)

@inline function _key(rng::_ReactantRNG{R}) where {R}
    T = _key_type(R)
    return ntuple(
        index -> _core_word(R, _convert(T, _state_value(rng, index))),
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
@inline function _core_words(::Type{R}, key, lo, hi) where {R}
    word(value) = _core_word(R, value)
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
function _core_body(::Type{R}, lo, hi, key_values...) where {R}
    return _core_words(R, map(value -> _core_word(R, value), key_values), lo, hi)
end

@inline function _shared_words(::Type{R}, key, lo, hi) where {R}
    return Ops.call(_core_body, R, lo, hi, _unwrap(key)...)
end

# The block at the position and its successor, in stream order. A draw may
# straddle the two, and the traced program has no branch to skip the second.
@inline function _window_words(rng::_ReactantRNG{R}, position) where {R}
    key = _key(rng)
    lo = position[1]
    hi = _position128(R) ? position[2] : lo
    next_lo = lo + UInt64(1)
    next_hi = _position128(R) ? hi + ifelse(iszero(next_lo), UInt64(1), UInt64(0)) : next_lo
    return (_shared_words(R, key, lo, hi)..., _shared_words(R, key, next_lo, next_hi)...)
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

@inline function _raw(rng::_ReactantRNG, ::Val{W}) where {W}
    position = _position(rng)
    bit = position[end]
    words = _window_words(rng, position)
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

@inline function _normal_value(rng, ::Type{T}) where {T<:Union{Float32,Float64}}
    raw = _raw(rng, Val(IR._normal_bits(T)))
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    midpoint = _convert(T, (raw * UInt64(2)) | UInt64(1)) * scale
    return _normal_transform(midpoint, T)
end

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
    bits = Ops.bitcast_convert(B, value)
    exponent = _convert(typeof(bias), _shr(bits, width) & exponent_mask) - bias
    mantissa = Ops.bitcast_convert(T, (bits & mantissa_mask) | one_bits)
    upper = mantissa > sqrt2
    mantissa = ifelse(upper, mantissa * half, mantissa)
    exponent += ifelse(upper, one(bias), zero(bias))
    n = -_convert(T, exponent)
    return IR._exponential_reduced(_ReactantTransformOps(), T, mantissa, n)
end

@inline function _exponential_value(rng, ::Type{T}) where {T}
    raw = _raw(rng, Val(IR._exponential_bits(T)))
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = _convert(T, raw) * scale
    return _exponential_transform(one(T) - u, T)
end

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

@inline function _mulhi64(word, span::UInt64)
    high = _shr(word, 32)
    low = word & UInt64(0xffffffff)
    return _shr(high * span + _shr(low * span, 32), 32)
end

@inline function _mulhi128(lo, hi, span::UInt64)
    ops = _ReactantWordOps()
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
        lo = _raw(rng, Val(64))
        hi = _raw(_advance(rng, UInt64(64)), Val(64))
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

# A child starts at position zero: its state is the key words followed by zeros.
@inline function _child(rng::_ReactantRNG{R}, key::Tuple) where {R}
    words = map(word -> _vector(_convert(UInt64, word.value)), key)
    rest = Ops.constant(zeros(UInt64, length(rng.state) - _key_count(R)))
    state = Ops.concatenate([words..., rest], 1)
    return _ReactantRNG{R,typeof(state)}(state)
end

@inline function _derive_child(rng::_ReactantRNG{R}, index::UInt64) where {R}
    return _child(rng, IR._derive_key(R, _key(rng), index))
end

IR.splitrng(rng::_ReactantRNG) = IR.splitrng(rng, Val(2))

@inline function IR.splitrng(rng::_ReactantRNG{R}, ::Val{N}) where {R,N}
    (N isa Int && N >= 0) || throw(ArgumentError("N must be a non-negative Int"))
    if R <: Union{IR.Philox2x32,IR.Threefry2x32}
        N <= IR._NARROW_SPLIT_COUNT ||
            throw(ArgumentError("two-word generator child index enters the fold namespace"))
    end
    return ntuple(index -> _derive_child(rng, UInt64(index - 1)), Val(N))
end

@inline IR.subrng(rng::_ReactantRNG{R}, purpose::Integer) where {R} =
    _child(rng, IR._subrng_key(R, _key(rng), purpose % UInt64))

end
