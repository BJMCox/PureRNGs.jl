# ChaCha (Bernstein 2008) as a counter-based generator. The state is the four
# constant words, eight key words, and four counter words. The package uses the
# original layout: words 13 and 14 carry the 64-bit block counter, words 15 and
# 16 are the nonce and hold the derivation tag.
const _CHACHA_CONSTANTS =
    (UInt32(0x61707865), UInt32(0x3320646e), UInt32(0x79622d32), UInt32(0x6b206574))

# Twelve rounds is the common generator choice, as in Rust's `StdRng`. Twenty
# is the cipher's round count. Rounds past `R` compile away.
const _CHACHA_DEFAULT_ROUNDS = 12

# The checkpoint after each rotation is the granularity at which a compiled
# ChaCha fill vectorizes. Checkpoints per round leave whole quarter rounds in
# one kernel, and XLA ran those five hundred times slower.
@inline function _chacha_quarter(a, b, c, d)
    a = _core_add(a, b)
    d = _core_checkpoint(_core_rotate(_core_xor(d, a), 16))
    c = _core_add(c, d)
    b = _core_checkpoint(_core_rotate(_core_xor(b, c), 12))
    a = _core_add(a, b)
    d = _core_checkpoint(_core_rotate(_core_xor(d, a), 8))
    c = _core_add(c, d)
    b = _core_checkpoint(_core_rotate(_core_xor(b, c), 7))
    return a, b, c, d
end

@inline function _chacha_column(x)
    x0, x4, x8, x12 = _chacha_quarter(x[1], x[5], x[9], x[13])
    x1, x5, x9, x13 = _chacha_quarter(x[2], x[6], x[10], x[14])
    x2, x6, x10, x14 = _chacha_quarter(x[3], x[7], x[11], x[15])
    x3, x7, x11, x15 = _chacha_quarter(x[4], x[8], x[12], x[16])
    return (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)
end

@inline function _chacha_diagonal(x)
    x0, x5, x10, x15 = _chacha_quarter(x[1], x[6], x[11], x[16])
    x1, x6, x11, x12 = _chacha_quarter(x[2], x[7], x[12], x[13])
    x2, x7, x8, x13 = _chacha_quarter(x[3], x[8], x[9], x[14])
    x3, x4, x9, x14 = _chacha_quarter(x[4], x[5], x[10], x[15])
    return (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)
end

@inline function _chacha(counter::NTuple{4,T}, key::NTuple{8,T}, ::Val{R}) where {T,R}
    constants = map(value -> _core_constant(key[1], value), _CHACHA_CONSTANTS)
    input = (constants..., key..., counter...)
    x = input
    Base.Cartesian.@nexprs 20 r -> if r <= R
        x = isodd(r) ? _chacha_column(x) : _chacha_diagonal(x)
    end
    return map(_core_add, x, input)
end
@inline _chacha(counter::NTuple{4,T}, key::NTuple{8,T}) where {T} =
    _chacha(counter, key, Val(_CHACHA_DEFAULT_ROUNDS))
