using MLDataDevices
using Random: randexp

const EXPONENTIAL_TYPES = (Float32, Float64)

_exponential_width(::Type{Float32}) = 24
_exponential_width(::Type{Float64}) = 53

function _reference_exponential_lattice(::Type{T}, raw::UInt64) where {T}
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = T(raw) * scale
    return u, one(T) - u
end

function _reference_exponential_cpu(v::Float32)
    sqrt2 = reinterpret(Float32, 0x3fb504f3)
    half = reinterpret(Float32, 0x3f000000)
    one_ = reinterpret(Float32, 0x3f800000)
    ln2_hi = reinterpret(Float32, 0x3f317200)
    ln2_lo = reinterpret(Float32, 0x35c00000)
    c3 = reinterpret(Float32, 0x3eaaaaab)
    c5 = reinterpret(Float32, 0x3e4ccccd)
    c7 = reinterpret(Float32, 0x3e124925)
    c9 = reinterpret(Float32, 0x3de38e39)
    c11 = reinterpret(Float32, 0x3dba2e8c)

    bits = reinterpret(UInt32, v)
    exponent = Int32((bits >> 23) & 0xff) - Int32(127)
    m = reinterpret(Float32, (bits & 0x007fffff) | 0x3f800000)
    upper = m > sqrt2
    m = ifelse(upper, m * half, m)
    exponent += ifelse(upper, Int32(1), Int32(0))
    t = (m - one_) / (m + one_)
    z = t * t
    p = fma(z, c11, c9)
    p = fma(z, p, c7)
    p = fma(z, p, c5)
    p = fma(z, p, c3)
    tz = t * z
    log_m = fma(tz, p, t)
    log_m += log_m
    n = -Float32(exponent)
    return fma(n, ln2_hi, fma(n, ln2_lo, -log_m))
end

function _reference_exponential_cpu(v::Float64)
    sqrt2 = reinterpret(Float64, 0x3ff6a09e667f3bcd)
    half = reinterpret(Float64, 0x3fe0000000000000)
    one_ = reinterpret(Float64, 0x3ff0000000000000)
    ln2_hi = reinterpret(Float64, 0x3fe62e42fef00000)
    ln2_lo = reinterpret(Float64, 0x3dd473de00000000)
    c3 = reinterpret(Float64, 0x3fd5555555555555)
    c5 = reinterpret(Float64, 0x3fc999999999999a)
    c7 = reinterpret(Float64, 0x3fc2492492492492)
    c9 = reinterpret(Float64, 0x3fbc71c71c71c71c)
    c11 = reinterpret(Float64, 0x3fb745d1745d1746)
    c13 = reinterpret(Float64, 0x3fb3b13b13b13b14)
    c15 = reinterpret(Float64, 0x3fb1111111111111)
    c17 = reinterpret(Float64, 0x3fae1e1e1e1e1e1e)
    c19 = reinterpret(Float64, 0x3faaf286bca1af28)
    c21 = reinterpret(Float64, 0x3fa8618618618618)
    c23 = reinterpret(Float64, 0x3fa642c8590b2164)

    bits = reinterpret(UInt64, v)
    exponent = Int64((bits >> 52) & 0x7ff) - Int64(1023)
    m = reinterpret(Float64, (bits & 0x000fffffffffffff) | 0x3ff0000000000000)
    upper = m > sqrt2
    m = ifelse(upper, m * half, m)
    exponent += ifelse(upper, Int64(1), Int64(0))
    t = (m - one_) / (m + one_)
    z = t * t
    p = fma(z, c23, c21)
    p = fma(z, p, c19)
    p = fma(z, p, c17)
    p = fma(z, p, c15)
    p = fma(z, p, c13)
    p = fma(z, p, c11)
    p = fma(z, p, c9)
    p = fma(z, p, c7)
    p = fma(z, p, c5)
    p = fma(z, p, c3)
    tz = t * z
    log_m = fma(tz, p, t)
    log_m += log_m
    n = -Float64(exponent)
    return fma(n, ln2_hi, fma(n, ln2_lo, -log_m))
end

function _reference_exponential(rng, ::Type{T}) where {T}
    width = _exponential_width(T)
    raw = _reference_extract(
        rng,
        IR.FAMILY_EXP,
        _reference_position_block(rng.position),
        rng.position.bit,
        width,
    )
    _, v = _reference_exponential_lattice(T, raw)
    return rng.device isa IR._CPUBackend ? _reference_exponential_cpu(v) : -Base.log(v)
end

@testset "R13 and R63 exponential golden vectors" begin
    expected = (
        (
            0xa05803,
            0x140b0076b1f96e,
            0x3f7c02be,
            0x3f7c02bf,
            0x3fef805814968c68,
            0x3fef805814968c69,
        ),
        (
            0xda96ce,
            0x1b52d9c94f7a5f,
            0x3ff62be7,
            0x3ff62be7,
            0x3ffec57d08964390,
            0x3ffec57d08964390,
        ),
        (
            0xabe90b,
            0x157d216f8ed58d,
            0x3f8e8068,
            0x3f8e8068,
            0x3ff1d00d16e33a3a,
            0x3ff1d00d16e33a3b,
        ),
        (
            0x6e4523,
            0x0dc8a4782bb60e,
            0x3f103c72,
            0x3f103c72,
            0x3fe2078e5d780b86,
            0x3fe2078e5d780b86,
        ),
        (
            0x98edd2,
            0x131dba42705949,
            0x3f68e5fb,
            0x3f68e5fb,
            0x3fed1cbf6b7dc906,
            0x3fed1cbf6b7dc907,
        ),
        (
            0xc12127,
            0x182424ea242c91,
            0x3fb3b990,
            0x3fb3b990,
            0x3ff67732129fcac2,
            0x3ff67732129fcac2,
        ),
        (
            0x15379a,
            0x02a6f350819147,
            0x3db12f9c,
            0x3db12f9c,
            0x3fb625f4028b615e,
            0x3fb625f4028b615e,
        ),
        (
            0x0cb66c,
            0x0196cd85660493,
            0x3d50a017,
            0x3d50a017,
            0x3faa14033fc04807,
            0x3faa14033fc04807,
        ),
    )

    for ((F, key), (raw32, raw64, cpu32, device32, cpu64, device64)) in
        zip(PACKED_GOLDEN_FAMILIES, expected)
        rng = _packed_golden_rng(F, key)
        block = _reference_position_block(rng.position)
        got32 =
            IR._extract_bits_unchecked(rng, IR.FAMILY_EXP, block, rng.position.bit, Val(24))
        got64 =
            IR._extract_bits_unchecked(rng, IR.FAMILY_EXP, block, rng.position.bit, Val(53))
        @test got32 === UInt64(raw32)
        @test got64 === UInt64(raw64)
        @test reinterpret(UInt32, randexp(rng, Float32)) === cpu32
        @test reinterpret(UInt64, randexp(rng, Float64)) === cpu64

        device_rng = MLDataDevices.CUDADevice()(rng)
        @test reinterpret(UInt32, randexp(device_rng, Float32)) === device32
        @test reinterpret(UInt64, randexp(device_rng, Float64)) === device64
    end
end

@testset "R63 exponential lattice and transform" begin
    for T in EXPONENTIAL_TYPES
        width = _exponential_width(T)
        maximum = (UInt64(1) << width) - UInt64(1)
        scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
        for raw in (UInt64(0), UInt64(1), maximum >> 1, maximum)
            u, v = _reference_exponential_lattice(T, raw)
            @test IR._exponential_lattice(T, raw) === (u, v)
            @test IR._exponential_transform(IR._CPU_BACKEND, T, v) ===
                  _reference_exponential_cpu(v)
            for token in (IR._CUDA_BACKEND, IR._AMDGPU_BACKEND, IR._METAL_BACKEND)
                @test IR._exponential_transform(token, T, v) === -Base.log(v)
            end
        end
        @test IR._exponential_lattice(T, UInt64(0)) === (zero(T), one(T))
        @test IR._exponential_lattice(T, maximum) === (one(T) - scale, scale)
        @test signbit(IR._exponential_transform(IR._CPU_BACKEND, T, one(T)))
    end
end

@testset "R23, R29, R53, and R63 exponential scalars" begin
    for F in FAMILY_TYPES, T in EXPONENTIAL_TYPES
        for bit in (UInt16(0), UInt16(24), UInt16(52), UInt16(63))
            rng = _positioned(F, 0x863, UInt64(9), bit)
            expected = _reference_exponential(rng, T)
            @test randexp(rng, T) === expected
            @test randexp(rng, T) === expected

            next_rng, value = randexp_next(rng, T)
            @test value === expected
            @test next_rng.position == _reference_position(rng, _exponential_width(T))
            @test randexpat(rng, T, 1) === expected
            @test randexpat(rng, T, 3) === _reference_exponential(
                IR._rebuild(
                    rng,
                    _reference_position(rng, 2 * _exponential_width(T)),
                    rng.device,
                ),
                T,
            )
        end
    end

    rng = Philox4x32(0x864)
    @test randexp_next(rng) === randexp_next(rng, Float64)
    @test_throws ArgumentError randexp(rng)
    @test sprint(showerror, try
        randexp(rng)
    catch error
        error
    end) == "ArgumentError: untyped immutable draws are forbidden; use randexp(rng, T)"
end
