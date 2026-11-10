# Jarzynski and Olano (2020), section 6.1. Updates within each mixing step are
# sequential, not simultaneous: later words use the already-updated words.
@inline function _pcg4d(input::NTuple{4,UInt32})
    x, y, z, w = map(value -> value * UInt32(1664525) + UInt32(1013904223), input)
    x += y * w
    y += z * x
    z += x * y
    w += y * z
    x ⊻= x >> 16
    y ⊻= y >> 16
    z ⊻= z >> 16
    w ⊻= w >> 16
    x += y * w
    y += z * x
    z += x * y
    w += y * z
    return x, y, z, w
end

@inline _pcg4d_input(key::NTuple{2,UInt32}, index::UInt64, domain::UInt32) =
    (index % UInt32, (index >> 32) % UInt32, key[1], key[2] | (domain << 30))
@inline _pcg4d_key(key, index::UInt64, domain::UInt32) =
    let words = _pcg4d(_pcg4d_input(key, index, domain))
        (words[1], words[2] & UInt32(0x3fffffff))
    end
