const _THREEFRY2X32_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)
const _THREEFRY2X64_ROTATIONS = (16, 42, 12, 31, 16, 32, 24, 21)
const _THREEFRY4X32_ROTATIONS =
    ((10, 26), (11, 21), (13, 27), (23, 5), (6, 20), (17, 11), (25, 10), (18, 20))
const _THREEFRY4X64_ROTATIONS =
    ((14, 16), (52, 57), (23, 40), (5, 37), (25, 33), (46, 12), (58, 22), (32, 32))

@inline function _threefry2x(
    counter::NTuple{2,T},
    key::NTuple{2,T},
    rotations,
    parity::T,
) where {T}
    k0, k1 = key
    k2 = _core_xor(_core_xor(k0, k1), parity)
    keys = (k0, k1, k2)
    x0 = _core_add(counter[1], k0)
    x1 = _core_add(counter[2], k1)

    Base.Cartesian.@nexprs 20 r -> begin
        round = r - 1
        x0 = _core_add(x0, x1)
        x1 = _core_rotate(x1, rotations[(round&7)+1])
        x1 = _core_xor(x1, x0)
        if round & 3 == 3
            s = (round + 1) >> 2
            x0 = _core_add(x0, keys[s%3+1])
            x1 = _core_add(
                x1,
                _core_add(keys[(s+1)%3+1], _core_constant(x1, s)),
            )
        end
    end
    return (x0, x1)
end

@inline function _threefry4x(
    counter::NTuple{4,T},
    key::NTuple{4,T},
    rotations,
    parity::T,
) where {T}
    k0, k1, k2, k3 = key
    k4 = _core_xor(_core_xor(_core_xor(_core_xor(k0, k1), k2), k3), parity)
    keys = (k0, k1, k2, k3, k4)
    x0 = _core_add(counter[1], k0)
    x1 = _core_add(counter[2], k1)
    x2 = _core_add(counter[3], k2)
    x3 = _core_add(counter[4], k3)

    Base.Cartesian.@nexprs 20 r -> begin
        round = r - 1
        rotations_round = rotations[(round&7)+1]
        if round & 1 == 0
            x0 = _core_add(x0, x1)
            x1 = _core_rotate(x1, rotations_round[1])
            x1 = _core_xor(x1, x0)
            x2 = _core_add(x2, x3)
            x3 = _core_rotate(x3, rotations_round[2])
            x3 = _core_xor(x3, x2)
        else
            x0 = _core_add(x0, x3)
            x3 = _core_rotate(x3, rotations_round[1])
            x3 = _core_xor(x3, x0)
            x2 = _core_add(x2, x1)
            x1 = _core_rotate(x1, rotations_round[2])
            x1 = _core_xor(x1, x2)
        end
        if round & 3 == 3
            s = (round + 1) >> 2
            x0 = _core_add(x0, keys[s%5+1])
            x1 = _core_add(x1, keys[(s+1)%5+1])
            x2 = _core_add(x2, keys[(s+2)%5+1])
            x3 = _core_add(
                x3,
                _core_add(keys[(s+3)%5+1], _core_constant(x3, s)),
            )
        end
    end
    return (x0, x1, x2, x3)
end

@inline _threefry2x32(counter::NTuple{2,UInt32}, key::NTuple{2,UInt32}) =
    _threefry2x32_impl(counter, key)
@inline _threefry2x32(counter::NTuple{2,T}, key::NTuple{2,T}) where {T<:_CoreWord{32}} =
    _threefry2x32_impl(counter, key)
@inline _threefry2x32_impl(counter, key) =
    _threefry2x(
        counter,
        key,
        _THREEFRY2X32_ROTATIONS,
        _core_constant(counter[1], 0x1BD11BDA),
    )

@inline _threefry4x32(counter::NTuple{4,UInt32}, key::NTuple{4,UInt32}) =
    _threefry4x32_impl(counter, key)
@inline _threefry4x32(counter::NTuple{4,T}, key::NTuple{4,T}) where {T<:_CoreWord{32}} =
    _threefry4x32_impl(counter, key)
@inline _threefry4x32_impl(counter, key) =
    _threefry4x(
        counter,
        key,
        _THREEFRY4X32_ROTATIONS,
        _core_constant(counter[1], 0x1BD11BDA),
    )

@inline _threefry2x64(counter::NTuple{2,UInt64}, key::NTuple{2,UInt64}) =
    _threefry2x64_impl(counter, key)
@inline _threefry2x64(counter::NTuple{2,T}, key::NTuple{2,T}) where {T<:_CoreWord{64}} =
    _threefry2x64_impl(counter, key)
@inline _threefry2x64_impl(counter, key) =
    _threefry2x(
        counter,
        key,
        _THREEFRY2X64_ROTATIONS,
        _core_constant(counter[1], 0x1BD11BDAA9FC1A22),
    )

@inline _threefry4x64(counter::NTuple{4,UInt64}, key::NTuple{4,UInt64}) =
    _threefry4x64_impl(counter, key)
@inline _threefry4x64(counter::NTuple{4,T}, key::NTuple{4,T}) where {T<:_CoreWord{64}} =
    _threefry4x64_impl(counter, key)
@inline _threefry4x64_impl(counter, key) =
    _threefry4x(
        counter,
        key,
        _THREEFRY4X64_ROTATIONS,
        _core_constant(counter[1], 0x1BD11BDAA9FC1A22),
    )
