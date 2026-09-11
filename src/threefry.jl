const _THREEFRY2X32_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)
const _THREEFRY2X64_ROTATIONS = (16, 42, 12, 31, 16, 32, 24, 21)
const _THREEFRY4X32_ROTATIONS =
    ((10, 26), (11, 21), (13, 27), (23, 5), (6, 20), (17, 11), (25, 10), (18, 20))
const _THREEFRY4X64_ROTATIONS =
    ((14, 16), (52, 57), (23, 40), (5, 37), (25, 33), (46, 12), (58, 22), (32, 32))
const _THREEFRY_PARITY32 = UInt32(0x1bd11bda)
const _THREEFRY_PARITY64 = UInt64(0x1bd11bdaa9fc1a22)

# Round counts follow Random123: twenty rounds by default, thirteen for the
# round-reduced generators. Rounds past `R` compile away.
const _THREEFRY_DEFAULT_ROUNDS = 20

@inline function _threefry2x(
    counter::NTuple{2,T},
    key::NTuple{2,T},
    rotations,
    parity::T,
    ::Val{R},
) where {T,R}
    k0, k1 = key
    k2 = _core_xor(_core_xor(k0, k1), parity)
    keys = (k0, k1, k2)
    x0 = _core_add(counter[1], k0)
    x1 = _core_add(counter[2], k1)

    Base.Cartesian.@nexprs 20 r -> if r <= R
        round = r - 1
        x0 = _core_add(x0, x1)
        x1 = _core_rotate(x1, rotations[(round&7)+1])
        x1 = _core_xor(x1, x0)
        if round & 3 == 3
            s = (round + 1) >> 2
            x0 = _core_add(x0, keys[s%3+1])
            x1 = _core_add(x1, _core_add(keys[(s+1)%3+1], _core_constant(x1, s)))
        end
    end
    return (x0, x1)
end

@inline function _threefry4x(
    counter::NTuple{4,T},
    key::NTuple{4,T},
    rotations,
    parity::T,
    ::Val{R},
) where {T,R}
    k0, k1, k2, k3 = key
    k4 = _core_xor(_core_xor(_core_xor(_core_xor(k0, k1), k2), k3), parity)
    keys = (k0, k1, k2, k3, k4)
    x0 = _core_add(counter[1], k0)
    x1 = _core_add(counter[2], k1)
    x2 = _core_add(counter[3], k2)
    x3 = _core_add(counter[4], k3)

    Base.Cartesian.@nexprs 20 r -> if r <= R
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
            x3 = _core_add(x3, _core_add(keys[(s+3)%5+1], _core_constant(x3, s)))
        end
    end
    return (x0, x1, x2, x3)
end

for (core, width, rotations, parity) in (
    (:_threefry2x32, 2, :_THREEFRY2X32_ROTATIONS, :_THREEFRY_PARITY32),
    (:_threefry4x32, 4, :_THREEFRY4X32_ROTATIONS, :_THREEFRY_PARITY32),
    (:_threefry2x64, 2, :_THREEFRY2X64_ROTATIONS, :_THREEFRY_PARITY64),
    (:_threefry4x64, 4, :_THREEFRY4X64_ROTATIONS, :_THREEFRY_PARITY64),
)
    mixer = width == 2 ? :_threefry2x : :_threefry4x
    @eval begin
        @inline $core(
            counter::NTuple{$width,T},
            key::NTuple{$width,T},
            rounds::Val,
        ) where {T} =
            $mixer(counter, key, $rotations, _core_constant(counter[1], $parity), rounds)
        @inline $core(counter::NTuple{$width,T}, key::NTuple{$width,T}) where {T} =
            $core(counter, key, Val(_THREEFRY_DEFAULT_ROUNDS))
    end
end
