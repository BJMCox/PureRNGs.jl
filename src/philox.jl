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

@inline function _philox2x32_impl(ctr, key)
    for round = 1:10
        ctr = _philox2x32_round(ctr, key)
        round == 10 ||
            (key = (_core_add(key[1], _core_constant(key[1], _PHILOX_W32_0)),))
    end
    return ctr
end

@inline _philox4x32_bump(key) = (
    _core_add(key[1], _core_constant(key[1], _PHILOX_W32_0)),
    _core_add(key[2], _core_constant(key[2], _PHILOX_W32_1)),
)

@inline function _philox4x32_impl(ctr, key)
    for round = 1:10
        ctr = _philox4x32_round(ctr, key)
        round == 10 || (key = _philox4x32_bump(key))
    end
    return ctr
end

@inline function _philox4x32_blocks4(
    a::NTuple{4,UInt32},
    b::NTuple{4,UInt32},
    c::NTuple{4,UInt32},
    d::NTuple{4,UInt32},
    key::NTuple{2,UInt32},
)
    Base.Cartesian.@nexprs 10 i -> begin
        a = _philox4x32_round(a, key)
        b = _philox4x32_round(b, key)
        c = _philox4x32_round(c, key)
        d = _philox4x32_round(d, key)
        i < 10 && (key = _philox4x32_bump(key))
    end
    return a, b, c, d
end

@inline function _philox2x64_impl(ctr, key)
    for round = 1:10
        ctr = _philox2x64_round(ctr, key)
        round == 10 ||
            (key = (_core_add(key[1], _core_constant(key[1], _PHILOX_W64_0)),))
    end
    return ctr
end

@inline function _philox4x64_impl(ctr, key)
    for round = 1:10
        ctr = _philox4x64_round(ctr, key)
        round == 10 ||
            (key = (
                _core_add(key[1], _core_constant(key[1], _PHILOX_W64_0)),
                _core_add(key[2], _core_constant(key[2], _PHILOX_W64_1)),
            ))
    end
    return ctr
end

@inline _philox2x32(ctr::NTuple{2,UInt32}, key::NTuple{1,UInt32}) =
    _philox2x32_impl(ctr, key)
@inline _philox2x32(ctr::NTuple{2,T}, key::NTuple{1,T}) where {T<:_CoreWord{32}} =
    _philox2x32_impl(ctr, key)
@inline _philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32}) =
    _philox4x32_impl(ctr, key)
@inline _philox4x32(ctr::NTuple{4,T}, key::NTuple{2,T}) where {T<:_CoreWord{32}} =
    _philox4x32_impl(ctr, key)
@inline _philox2x64(ctr::NTuple{2,UInt64}, key::NTuple{1,UInt64}) =
    _philox2x64_impl(ctr, key)
@inline _philox2x64(ctr::NTuple{2,T}, key::NTuple{1,T}) where {T<:_CoreWord{64}} =
    _philox2x64_impl(ctr, key)
@inline _philox4x64(ctr::NTuple{4,UInt64}, key::NTuple{2,UInt64}) =
    _philox4x64_impl(ctr, key)
@inline _philox4x64(ctr::NTuple{4,T}, key::NTuple{2,T}) where {T<:_CoreWord{64}} =
    _philox4x64_impl(ctr, key)
