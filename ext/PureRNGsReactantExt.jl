module PureRNGsReactantExt

import PureRNGs
import Random
import Reactant

const IR = PureRNGs
const _ReactantRNG = IR._ReactantRNG

struct _ReactantWordOps end
struct _ReactantTransformOps end

@inline IR._word_constant(::_ReactantWordOps, ::Val{32}, anchor, value) =
    (anchor - anchor) + UInt32(value)
@inline IR._word_constant(::_ReactantWordOps, ::Val{64}, anchor, value) =
    (anchor - anchor) + UInt64(value)
@inline IR._word_from_value(::_ReactantWordOps, ::Val{32}, anchor, value) =
    (anchor - anchor) + _as32(value & UInt64(0xffffffff))
@inline IR._word_from_value(::_ReactantWordOps, ::Val{64}, anchor, value) =
    (anchor - anchor) + value
@inline IR._word_add(::_ReactantWordOps, ::Val{W}, a, b) where {W} = a + b
@inline IR._word_xor(::_ReactantWordOps, ::Val{W}, a, b) where {W} = xor(a, b)
@inline IR._transform_muladd(::_ReactantTransformOps, a, b, c) = muladd(a, b, c)

@inline function IR._word_rotate(::_ReactantWordOps, ::Val{W}, value, count) where {W}
    one_ = W == 32 ? UInt32(1) : UInt64(1)
    left = value * (one_ << count)
    right = div(value, one_ << (W - count))
    return left | right
end

@inline function IR._word_mulhilo(::_ReactantWordOps, ::Val{32}, a, b)
    product = _as64(a) * _as64(b)
    return _as32(div(product, UInt64(1) << 32)), _as32(product & UInt64(0xffffffff))
end

@inline function IR._word_mulhilo(::_ReactantWordOps, ::Val{64}, a, b)
    mask = UInt64(0xffffffff)
    alo, ahi = a & mask, div(a, UInt64(1) << 32)
    blo, bhi = b & mask, div(b, UInt64(1) << 32)
    p0 = alo * blo
    p1 = ahi * blo
    p2 = alo * bhi
    p3 = ahi * bhi
    middle = div(p0, UInt64(1) << 32) + (p1 & mask) + (p2 & mask)
    hi =
        p3 +
        div(p1, UInt64(1) << 32) +
        div(p2, UInt64(1) << 32) +
        div(middle, UInt64(1) << 32)
    return hi, p0 + (p1 * (UInt64(1) << 32)) + (p2 * (UInt64(1) << 32))
end

@inline function _state_value(rng::_ReactantRNG, index::Int)
    return Reactant.@allowscalar rng.state[index]
end

@inline _cast_scalar(::Type{T}, value) where {T} = Reactant.@allowscalar T.([value])[1]

@inline _as32(value) = Reactant.@allowscalar UInt32.([value])[1]
@inline _as64(value) = Reactant.@allowscalar UInt64.([value])[1]

@inline function _bitcast_signed(::Type{T}, value) where {T<:Signed}
    low_mask = UInt64(typemax(T))
    sign_mask = low_mask + UInt64(1)
    low = _cast_scalar(T, value & low_mask)
    return ifelse(iszero(value & sign_mask), low, typemin(T) + low)
end

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

@inline function _key(rng::_ReactantRNG{R}) where {R}
    count = _key_count(R)
    width = _word_width(R)
    words = width isa Val{32} ? UInt32.(rng.state) : rng.state
    return ntuple(
        index ->
            IR._core_word(width, _ReactantWordOps(), Reactant.@allowscalar(words[index])),
        Val(count),
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

@inline function _word32(value)
    return IR._core_word(Val(32), _ReactantWordOps(), _as32(value))
end
@inline function _word64(value)
    return IR._core_word(Val(64), _ReactantWordOps(), value)
end
@inline _unwrap(words) = map(word -> word.value, words)

# The draw counter is the block address zero-extended to the core's counter width,
# matching `_block` in src/bits.jl.
@inline function _block(rng::_ReactantRNG{R}, lo, hi) where {R}
    key = _key(rng)
    pad = IR._core_constant(key[1], 0)
    if R <: Union{IR.Philox2x32,IR.Threefry2x32}
        low = _word32(lo & UInt64(0xffffffff))
        high = _word32(div(lo, UInt64(1) << 32) & UInt64(0x00ffffff))
        words =
            R <: IR.Philox2x32 ? IR._philox2x32((low, high), key, Val(IR._rounds(R))) :
            IR._threefry2x32((low, high), key, Val(IR._rounds(R)))
    elseif R <: Union{IR.Philox4x32,IR.Threefry4x32}
        counter =
            (_word32(lo & UInt64(0xffffffff)), _word32(div(lo, UInt64(1) << 32)), pad, pad)
        words =
            R <: IR.Philox4x32 ? IR._philox4x32(counter, key, Val(IR._rounds(R))) :
            IR._threefry4x32(counter, key, Val(IR._rounds(R)))
    elseif R <: Union{IR.Philox2x64,IR.Threefry2x64}
        counter = (_word64(lo), pad)
        words =
            R <: IR.Philox2x64 ? IR._philox2x64(counter, key, Val(IR._rounds(R))) :
            IR._threefry2x64(counter, key, Val(IR._rounds(R)))
    else
        counter = (_word64(lo), _word64(hi), pad, pad)
        words =
            R <: IR.Philox4x64 ? IR._philox4x64(counter, key, Val(IR._rounds(R))) :
            IR._threefry4x64(counter, key, Val(IR._rounds(R)))
    end
    return _unwrap(words)
end

@inline function _block_words(rng::_ReactantRNG{R}, lo, hi) where {R}
    words = _block(rng, lo, hi)
    _word_width(R) isa Val{32} || return words
    return ntuple(
        i -> (_as64(words[2i-1]) * (UInt64(1) << 32)) | _as64(words[2i]),
        Val(length(words) ÷ 2),
    )
end

@inline function _next_block(rng::_ReactantRNG{R}, lo, hi) where {R}
    next_lo = lo + UInt64(1)
    next_hi = _position128(R) ? hi + ifelse(iszero(next_lo), UInt64(1), UInt64(0)) : hi
    return next_lo, next_hi
end

@inline function _select(values, index)
    result = values[1]
    for i = 2:length(values)
        result = ifelse(index == UInt64(i - 1), values[i], result)
    end
    return result
end

@inline function _shift_right(value, count)
    result = value
    for shift in (1, 2, 4, 8, 16, 32)
        shifted = div(result, UInt64(1) << shift)
        result = ifelse(iszero(count & UInt64(shift)), result, shifted)
    end
    return result
end

@inline _power_of_two(count) = UInt64(2)^count
@inline _shift_left(value, count) = value * _power_of_two(count)
@inline _low_mask(count) = _power_of_two(count) - UInt64(1)

@inline function _raw(rng::_ReactantRNG{R}, ::Val{W}) where {R,W}
    position = _position(rng)
    lo = position[1]
    hi = _position128(R) ? position[2] : UInt64(0)
    bit = position[end]
    current = _block_words(rng, lo, hi)
    next_lo, next_hi = _next_block(rng, lo, hi)
    following = _block_words(rng, next_lo, next_hi)
    block_words = (current..., following...)
    lane = div(bit, UInt64(64))
    word_bit = bit & UInt64(63)
    first = _select(block_words, lane)
    second = _select(block_words, lane + UInt64(1))
    available = UInt64(64) - word_bit
    right_count = (available - UInt64(W)) & UInt64(63)
    single = _shift_right(first, right_count) & _low_mask(UInt64(W))
    remaining = (UInt64(W) - available) & UInt64(63)
    crossed =
        _shift_left(first & _low_mask(available), remaining) |
        _shift_right(second, (UInt64(64) - remaining) & UInt64(63))
    return ifelse(UInt64(W) <= available, single, crossed)
end

# ChaCha traces in the four-lane matrix form: `a`, `b`, `c`, `d` are 4×N words,
# one column per block, and the diagonal round permutes rows. XLA compile time
# grows with the traced op count, and the word-by-word form of a 512-bit block
# costs several seconds per compiled draw. Shifts are unavailable on traced
# words in the pinned Reactant, so rotation uses the multiply and divide form.
const _CHACHA_LANES = (UInt32[2, 3, 4, 1], UInt32[3, 4, 1, 2], UInt32[4, 1, 2, 3])

@inline _chacha_rotate(v, k) = (v .* UInt32(1 << k)) .| div.(v, UInt32(1) << (32 - k))
@inline _chacha_lanes(v, k) = v[_CHACHA_LANES[k], :]

@inline function _chacha_quarter(a, b, c, d)
    a = a .+ b
    d = _chacha_rotate(d .⊻ a, 16)
    c = c .+ d
    b = _chacha_rotate(b .⊻ c, 12)
    a = a .+ b
    d = _chacha_rotate(d .⊻ a, 8)
    c = c .+ d
    b = _chacha_rotate(b .⊻ c, 7)
    return a, b, c, d
end

# `counter` is the 4×N matrix of counter and nonce words, one column per block.
# Returns the 16×N output words in state order.
function _chacha_matrix(key, counter, ::Val{R}) where {R}
    columns = size(counter, 2)
    b0 = repeat(key[1:4], 1, columns)
    c0 = repeat(key[5:8], 1, columns)
    a0 = (b0 .- b0) .+ collect(IR._CHACHA_CONSTANTS)
    a, b, c, d = a0, b0, c0, counter
    for round = 1:R
        if isodd(round)
            a, b, c, d = _chacha_quarter(a, b, c, d)
        else
            b, c, d = _chacha_lanes(b, 1), _chacha_lanes(c, 2), _chacha_lanes(d, 3)
            a, b, c, d = _chacha_quarter(a, b, c, d)
            b, c, d = _chacha_lanes(b, 3), _chacha_lanes(c, 2), _chacha_lanes(d, 1)
        end
    end
    return vcat(a .+ a0, b .+ b0, c .+ c0, d .+ counter)
end

@inline _chacha_key(rng::_ReactantRNG) = UInt32.(rng.state[1:8])

# Pair the 16×N output words into 8N block words, block-major.
@inline function _chacha_block_words(out)
    wide = UInt64.(out)
    return vec((wide[1:2:15, :] .* (UInt64(1) << 32)) .| wide[2:2:16, :])
end

# The block words of blocks `lo` to `lo + N - 1`. The nonce rows are zero for
# draws, matching `_core_block` in src/bits.jl.
function _chacha_words(rng::_ReactantRNG{R}, lo, ::Val{N}) where {R,N}
    key = _chacha_key(rng)
    zeros = repeat(key[1:4] .- key[1:4], 1, N)
    lo32 = _as32(lo & UInt64(0xffffffff))
    hi32 = _as32(div(lo, UInt64(1) << 32))
    row = repeat(UInt32[1, 2, 3, 4], 1, N)
    lo_rows = lo32 .+ repeat(UInt32.(0:N-1)', 4, 1)
    carry = ifelse.(lo_rows .< lo32 .+ zeros, zeros .+ UInt32(1), zeros)
    hi_rows = ifelse.(row .== UInt32(2), hi32 .+ carry, zeros)
    counter = ifelse.(row .== UInt32(1), lo_rows, hi_rows)
    return _chacha_block_words(_chacha_call(key, counter, Val(IR._rounds(R))))
end

# Derivation runs blocks whose counter and nonce words are constants, one
# column per block, as in `_derive_key` and `_subrng_key` for ChaCha in
# src/derive.jl. Each block yields two child keys, its lower and upper halves.
function _chacha_derived_blocks(rng::_ReactantRNG{R}, columns::Vector{UInt32}) where {R}
    key = _chacha_key(rng)
    zeros = repeat(key[1:4] .- key[1:4], 1, length(columns) ÷ 4)
    counter = zeros .+ reshape(columns, 4, :)
    return _chacha_call(key, counter, Val(IR._rounds(R)))
end

# Every block in a compiled function calls one shared MLIR function. Inlining
# the rounds at every draw site made the optimization passes superlinear in
# module size: 48 chained ChaCha20 draws took 86 s and 10 GB to compile.
@inline _chacha_call(key, counter, rounds::Val) =
    Reactant.Ops.call(_chacha_matrix, key, counter, rounds)

@inline _chacha_derived_key(out, column::Int, half::Int) = ntuple(
    index -> IR._core_word(
        Val(32),
        _ReactantWordOps(),
        Reactant.@allowscalar(out[8half+index, column]),
    ),
    Val(8),
)

@inline _chacha_split_column(block_index::UInt64) = (
    block_index % UInt32,
    (block_index >> 32) % UInt32,
    IR._SPLIT_SUBTAG,
    IR._DERIVE_TAG,
)

# Indexing a traced vector with a traced index returns a one-element array in
# the pinned Reactant, so read it back to a scalar.
@inline function _dynamic_word(words, index)
    word = Reactant.@allowscalar words[index]
    return word isa AbstractArray ? (Reactant.@allowscalar word[1]) : word
end

@inline function _raw(rng::_ReactantRNG{R}, ::Val{W}) where {R<:IR.ChaCha,W}
    position = _position(rng)
    lo = position[1]
    bit = position[end]
    words = _chacha_words(rng, lo, Val(2))
    lane = div(bit, UInt64(64))
    word_bit = bit & UInt64(63)
    first = _dynamic_word(words, lane + UInt64(1))
    second = _dynamic_word(words, lane + UInt64(2))
    available = UInt64(64) - word_bit
    right_count = (available - UInt64(W)) & UInt64(63)
    single = _shift_right(first, right_count) & _low_mask(UInt64(W))
    remaining = (UInt64(W) - available) & UInt64(63)
    crossed =
        _shift_left(first & _low_mask(available), remaining) |
        _shift_right(second, (UInt64(64) - remaining) & UInt64(63))
    return ifelse(UInt64(W) <= available, single, crossed)
end

@inline _draw_bits(::Type{T}) where {T} = UInt64(IR._draw_bits(T))

@inline function _convert_result(::Type{Bool}, raw)
    return raw == UInt64(1)
end
@inline _convert_result(::Type{UInt32}, raw) = _cast_scalar(UInt32, raw)
@inline _convert_result(::Type{UInt64}, raw) = raw
@inline _convert_result(::Type{Int32}, raw) = _bitcast_signed(Int32, raw)
@inline _convert_result(::Type{Int64}, raw) = _bitcast_signed(Int64, raw)
@inline function _convert_result(::Type{Float32}, raw)
    return _cast_scalar(Float32, raw) * Float32(0x1p-24)
end
@inline function _convert_result(::Type{Float64}, raw)
    return _cast_scalar(Float64, raw) * Float64(0x1p-53)
end

@inline function _draw(rng::_ReactantRNG, ::Type{T}) where {T}
    width = IR._draw_bits(T)
    return _convert_result(T, _raw(rng, Val(width)))
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
    block_lo = div(sum_lo, block_bits) | (sum_hi * (UInt64(1) << (64 - shift)))
    block_hi = div(sum_hi, block_bits) | (sum_top * (UInt64(1) << (64 - shift)))
    block_top = div(sum_top, block_bits)

    lo = position[1] + block_lo
    carry = ifelse(lo < position[1], UInt64(1), UInt64(0))
    if _position128(R)
        partial_hi = position[2] + block_hi
        top = position[3] + block_top
        top += ifelse(partial_hi < position[2], UInt64(1), UInt64(0))
        hi = partial_hi + carry
        top += ifelse(hi < partial_hi, UInt64(1), UInt64(0))
    else
        hi = position[2] + block_hi + carry
    end
    state = similar(rng.state)
    count = _key_count(R)
    for index = 1:count
        Reactant.@allowscalar state[index] = _state_value(rng, index)
    end
    Reactant.@allowscalar state[count+1] = lo
    if _position128(R)
        Reactant.@allowscalar state[count+2] = hi
        Reactant.@allowscalar state[count+3] = top
        Reactant.@allowscalar state[count+4] = bit
    else
        Reactant.@allowscalar state[count+2] = hi
        Reactant.@allowscalar state[count+3] = bit
    end
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
            return _draw(rng, $T), _advance(rng, _draw_bits($T))
        end
        @inline function IR.randat(rng::_ReactantRNG, ::Type{$T}, index::Integer)
            return _draw(IR._addressed_rng(rng, IR._draw_bits($T), index), $T)
        end
    end
end

@inline _normal_bits(::Type{T}) where {T} = UInt64(IR._normal_bits(T))
@inline _exponential_bits(::Type{T}) where {T} = UInt64(IR._exponential_bits(T))

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
    width = IR._normal_bits(T)
    raw = _raw(rng, Val(width))
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    midpoint = _cast_scalar(T, (raw * UInt64(2)) | UInt64(1)) * scale
    return _normal_transform(midpoint, T)
end

@inline function _normalized_binary(value, ::Type{T}) where {T}
    mantissa = value
    exponent = -zero(T)
    for _ = 1:precision(T)
        lower = mantissa < one(T)
        mantissa = ifelse(lower, mantissa + mantissa, mantissa)
        exponent = ifelse(lower, exponent + one(T), exponent)
    end
    return mantissa, exponent
end

@inline function _exponential_transform(value, ::Type{T}) where {T}
    sqrt2 =
        T === Float32 ? reinterpret(Float32, UInt32(0x3fb504f3)) :
        reinterpret(Float64, UInt64(0x3ff6a09e667f3bcd))
    half =
        T === Float32 ? reinterpret(Float32, UInt32(0x3f000000)) :
        reinterpret(Float64, UInt64(0x3fe0000000000000))
    mantissa, n = _normalized_binary(value, T)
    upper = mantissa > sqrt2
    mantissa = ifelse(upper, mantissa * half, mantissa)
    n = ifelse(upper, n - one(T), n)
    return IR._exponential_reduced(_ReactantTransformOps(), T, mantissa, n)
end

@inline function _exponential_value(rng, ::Type{T}) where {T}
    width = IR._exponential_bits(T)
    raw = _raw(rng, Val(width))
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = _cast_scalar(T, raw) * scale
    return _exponential_transform(one(T) - u, T)
end

for T in (Float32, Float64)
    @eval begin
        @inline Random.randn(rng::_ReactantRNG, ::Type{$T}) = _normal_value(rng, $T)
        @inline function IR.randn_next(rng::_ReactantRNG, ::Type{$T})
            return _normal_value(rng, $T), _advance(rng, _normal_bits($T))
        end
        @inline function IR.randnat(rng::_ReactantRNG, ::Type{$T}, index::Integer)
            return _normal_value(IR._addressed_rng(rng, IR._normal_bits($T), index), $T)
        end
        @inline Random.randexp(rng::_ReactantRNG, ::Type{$T}) = _exponential_value(rng, $T)
        @inline function IR.randexp_next(rng::_ReactantRNG, ::Type{$T})
            return _exponential_value(rng, $T), _advance(rng, _exponential_bits($T))
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
    high = div(word, UInt64(1) << 32)
    low = word & UInt64(0xffffffff)
    return div(high * span + div(low * span, UInt64(1) << 32), UInt64(1) << 32)
end

@inline function _mulhi128(lo, hi, span::UInt64)
    high_hi, high_lo = IR._word_mulhilo(_ReactantWordOps(), Val(64), hi, span)
    low_hi, _ = IR._word_mulhilo(_ReactantWordOps(), Val(64), lo, span)
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
    return T <: Signed ? _bitcast_signed(T, bits) : _cast_scalar(T, bits)
end

@inline function _range_element(range::LinRange{T}, offset) where {T<:IR._RangeInteger}
    denominator = max(length(range) - 1, 1)
    t = _cast_scalar(Float64, offset) / Float64(denominator)
    value = (one(Float64) - t) * first(range) + t * last(range)
    return _cast_scalar(T, value)
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

@inline function _child(rng::_ReactantRNG{R}, key) where {R}
    state = similar(rng.state)
    width = _word_width(R)
    count = _key_count(R)
    for index = (count+1):length(rng.state)
        Reactant.@allowscalar state[index] = UInt64(0)
    end
    for index = 1:count
        value = key[index].value
        Reactant.@allowscalar state[index] = width isa Val{32} ? _as64(value) : value
    end
    return _ReactantRNG{R,typeof(state)}(state)
end

@inline function _derive_child(rng::_ReactantRNG{R}, index::UInt64) where {R}
    return _child(rng, IR._derive_key(R, _key(rng), index))
end

@inline function _derive_child(rng::_ReactantRNG{R}, index::UInt64) where {R<:IR.ChaCha}
    block_index, group = divrem(index, UInt64(2))
    out = _chacha_derived_blocks(rng, collect(_chacha_split_column(block_index)))
    return _child(rng, _chacha_derived_key(out, 1, Int(group)))
end

IR.splitrng(rng::_ReactantRNG) = IR.splitrng(rng, Val(2))

@inline function IR.splitrng(rng::_ReactantRNG{R}, ::Val{N}) where {R<:IR.ChaCha,N}
    (N isa Int && N >= 0) || throw(ArgumentError("N must be a non-negative Int"))
    N == 0 && return ()
    blocks = cld(N, 2)
    columns = collect(Iterators.flatten(_chacha_split_column(UInt64(b)) for b = 0:blocks-1))
    out = _chacha_derived_blocks(rng, columns)
    return ntuple(Val(N)) do index
        block_index, group = divrem(index - 1, 2)
        _child(rng, _chacha_derived_key(out, block_index + 1, group))
    end
end

@inline function IR.splitrng(rng::_ReactantRNG{R}, ::Val{N}) where {R,N}
    (N isa Int && N >= 0) || throw(ArgumentError("N must be a non-negative Int"))
    if R <: Union{IR.Philox2x32,IR.Threefry2x32}
        N <= IR._NARROW_SPLIT_COUNT ||
            throw(ArgumentError("two-word generator child index enters the fold namespace"))
    end
    return ntuple(index -> _derive_child(rng, UInt64(index - 1)), Val(N))
end

@inline function _subrng(rng::_ReactantRNG{R}, purpose) where {R}
    return _child(rng, IR._subrng_key(R, _key(rng), purpose))
end

@inline function _subrng(rng::_ReactantRNG{R}, purpose) where {R<:IR.ChaCha}
    words = (purpose % UInt32, (purpose >> 32) % UInt32, IR._FOLD_SUBTAG, IR._DERIVE_TAG)
    out = _chacha_derived_blocks(rng, collect(words))
    return _child(rng, _chacha_derived_key(out, 1, 0))
end

@inline IR.subrng(rng::_ReactantRNG, purpose::Integer) = _subrng(rng, purpose % UInt64)

end
