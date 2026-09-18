const _PHILOX_M2X32_0 = UInt32(0xd256d193)
const _PHILOX_M4X32_0 = UInt32(0xd2511f53)
const _PHILOX_M4X32_1 = UInt32(0xcd9e8d57)
const _PHILOX_W32_0 = UInt32(0x9e3779b9)
const _PHILOX_W32_1 = UInt32(0xbb67ae85)

const _PHILOX_M2X64_0 = UInt64(0xd2b74407b1ce6e93)
const _PHILOX_M4X64_0 = UInt64(0xd2e7470ee14c6c93)
const _PHILOX_M4X64_1 = UInt64(0xca5a826395121157)
const _PHILOX_W64_0 = UInt64(0x9e3779b97f4a7c15)
const _PHILOX_W64_1 = UInt64(0xbb67ae8584caa73b)

@inline function _mulhilo32(a::UInt32, b::UInt32)
    product = UInt64(a) * UInt64(b)
    return UInt32(product >> 32), UInt32(product & 0xffffffff)
end

@inline function _mulhilo64(a::UInt64, b::UInt64)
    mask = UInt64(0xffffffff)
    alo, ahi = a & mask, a >> 32
    blo, bhi = b & mask, b >> 32
    p0 = alo * blo
    p1 = ahi * blo
    p2 = alo * bhi
    p3 = ahi * bhi
    middle = (p0 >> 32) + (p1 & mask) + (p2 & mask)
    return p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32), p0 + (p1 << 32) + (p2 << 32)
end

@inline function _philox2x32_round(ctr, key)
    hi, lo = _mulhilo32(_core_constant(ctr[1], _PHILOX_M2X32_0), ctr[1])
    return _core_xor(_core_xor(hi, ctr[2]), key[1]), lo
end

@inline function _philox4x32_round(ctr, key)
    hi0, lo0 = _mulhilo32(_core_constant(ctr[1], _PHILOX_M4X32_0), ctr[1])
    hi1, lo1 = _mulhilo32(_core_constant(ctr[3], _PHILOX_M4X32_1), ctr[3])
    return (
        _core_xor(_core_xor(hi1, ctr[2]), key[1]),
        lo1,
        _core_xor(_core_xor(hi0, ctr[4]), key[2]),
        lo0,
    )
end

@inline function _philox2x64_round(ctr, key)
    hi, lo = _mulhilo64(_core_constant(ctr[1], _PHILOX_M2X64_0), ctr[1])
    return _core_xor(_core_xor(hi, ctr[2]), key[1]), lo
end

@inline function _philox4x64_round(ctr, key)
    hi0, lo0 = _mulhilo64(_core_constant(ctr[1], _PHILOX_M4X64_0), ctr[1])
    hi1, lo1 = _mulhilo64(_core_constant(ctr[3], _PHILOX_M4X64_1), ctr[3])
    return (
        _core_xor(_core_xor(hi1, ctr[2]), key[1]),
        lo1,
        _core_xor(_core_xor(hi0, ctr[4]), key[2]),
        lo0,
    )
end

# Round counts follow Random123: ten rounds by default, and the reduced counts
# that still pass BigCrush for the round-reduced generators. Every round except
# the last bumps the key by the Weyl constants.
const _PHILOX_DEFAULT_ROUNDS = 10

@inline _philox2x32_bump(key) = (_core_add(key[1], _core_constant(key[1], _PHILOX_W32_0)),)
@inline _philox4x32_bump(key) = (
    _core_add(key[1], _core_constant(key[1], _PHILOX_W32_0)),
    _core_add(key[2], _core_constant(key[2], _PHILOX_W32_1)),
)
@inline _philox2x64_bump(key) = (_core_add(key[1], _core_constant(key[1], _PHILOX_W64_0)),)
@inline _philox4x64_bump(key) = (
    _core_add(key[1], _core_constant(key[1], _PHILOX_W64_0)),
    _core_add(key[2], _core_constant(key[2], _PHILOX_W64_1)),
)

for (core, round, bump) in (
    (:_philox2x32, :_philox2x32_round, :_philox2x32_bump),
    (:_philox4x32, :_philox4x32_round, :_philox4x32_bump),
    (:_philox2x64, :_philox2x64_round, :_philox2x64_bump),
    (:_philox4x64, :_philox4x64_round, :_philox4x64_bump),
)
    @eval begin
        @inline function $core(
            ctr::Tuple{T,Vararg{T}},
            key::Tuple{T,Vararg{T}},
            ::Val{R},
        ) where {T,R}
            for r = 1:R
                ctr = _core_checkpoint($round(ctr, key))
                r == R || (key = $bump(key))
            end
            return ctr
        end
        @inline $core(ctr::Tuple{T,Vararg{T}}, key::Tuple{T,Vararg{T}}) where {T} =
            $core(ctr, key, Val(_PHILOX_DEFAULT_ROUNDS))
    end
end

@inline function _philox4x32_blocks4(
    a::NTuple{4,UInt32},
    b::NTuple{4,UInt32},
    c::NTuple{4,UInt32},
    d::NTuple{4,UInt32},
    key::NTuple{2,UInt32},
    ::Val{R},
) where {R}
    Base.Cartesian.@nexprs 10 i -> if i <= R
        a = _philox4x32_round(a, key)
        b = _philox4x32_round(b, key)
        c = _philox4x32_round(c, key)
        d = _philox4x32_round(d, key)
        i < R && (key = _philox4x32_bump(key))
    end
    return a, b, c, d
end
