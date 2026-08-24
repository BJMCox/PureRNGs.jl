using InteractiveUtils: code_llvm
using Random: randn

const NORMAL_TYPES = (Float32, Float64)

_normal_width(::Type{Float32}) = 23
_normal_width(::Type{Float64}) = 52

const REFERENCE_AS241_32 = (
    (5.9109374720f1, 1.5929113202f2, 5.0434271938f1, 3.3871327179f0),
    (6.7187563600f1, 7.8757757664f1, 1.7895169469f1, 1.0f0),
    (1.7023821103f-1, 1.3067284816f0, 2.7568153900f0, 1.4234372777f0),
    (1.2021132975f-1, 7.3700164250f-1, 1.0f0),
    (1.7337203997f-2, 4.2868294337f-1, 3.0812263860f0, 6.6579051150f0),
    (1.2258202635f-2, 2.4197894225f-1, 1.0f0),
)

const REFERENCE_AS241_64 = (
    (
        2.5090809287301226727e3,
        3.3430575583588128105e4,
        6.7265770927008700853e4,
        4.5921953931549871457e4,
        1.3731693765509461125e4,
        1.9715909503065514427e3,
        1.3314166789178437745e2,
        3.3871328727963666080,
    ),
    (
        5.2264952788528545610e3,
        2.8729085735721942674e4,
        3.9307895800092710610e4,
        2.1213794301586595867e4,
        5.3941960214247511077e3,
        6.8718700749205790830e2,
        4.2313330701600911252e1,
        1.0,
    ),
    (
        7.74545014278341407640e-4,
        2.27238449892691845833e-2,
        2.41780725177450611770e-1,
        1.27045825245236838258,
        3.64784832476320460504,
        5.76949722146069140550,
        4.63033784615654529590,
        1.42343711074968357734,
    ),
    (
        1.05075007164441684324e-9,
        5.47593808499534494600e-4,
        1.51986665636164571966e-2,
        1.48103976427480074590e-1,
        6.89767334985100004550e-1,
        1.67638483018380384940,
        2.05319162663775882187,
        1.0,
    ),
    (
        2.01033439929228813265e-7,
        2.71155556874348757815e-5,
        1.24266094738807843860e-3,
        2.65321895265761230930e-2,
        2.96560571828504891230e-1,
        1.78482653991729133580,
        5.46378491116411436990,
        6.65790464350110377720,
    ),
    (
        2.04426310338993978564e-15,
        1.42151175831644588870e-7,
        1.84631831751005468180e-5,
        7.86869131145613259100e-4,
        1.48753612908506148525e-2,
        1.36929880922735805310e-1,
        5.99832206555887937690e-1,
        1.0,
    ),
)

function _reference_horner(x::T, coefficients::NTuple{N,T}) where {T,N}
    value = coefficients[1]
    for index = 2:N
        value = fma(value, x, coefficients[index])
    end
    return value
end

function _reference_as241(u::T, coefficients) where {T}
    A, B, C, D, E, F = coefficients
    q = u - T(0.5)
    if abs(q) <= T(0.425)
        r = T(0.180625) - q * q
        return q * (_reference_horner(r, A) / _reference_horner(r, B))
    end
    r = sqrt(-log(q < zero(T) ? u : one(T) - u))
    if r <= T(5)
        r = r - T(1.6)
        z = _reference_horner(r, C) / _reference_horner(r, D)
    else
        r = r - T(5)
        z = _reference_horner(r, E) / _reference_horner(r, F)
    end
    return q < zero(T) ? -z : z
end

_reference_as241(u::Float32) = _reference_as241(u, REFERENCE_AS241_32)
_reference_as241(u::Float64) = _reference_as241(u, REFERENCE_AS241_64)

function _reference_normal(rng, ::Type{T}) where {T}
    width = _normal_width(T)
    raw = _reference_extract(
        rng,
        IR.FAMILY_NORMAL,
        _reference_position_block(rng.position),
        rng.position.bit,
        width,
    )
    u = if T === Float32
        Float32(((raw % UInt32) << UInt32(1)) | UInt32(1)) * Float32(0x1p-24)
    else
        Float64((raw << UInt64(1)) | UInt64(1)) * Float64(0x1p-53)
    end
    return _reference_as241(u)
end

function _terminal_normal_rng(F, ::Type{T}) where {T}
    rng = F(0x741)
    bit = UInt16(IR._block_bits(rng) - _normal_width(T))
    position = if rng.position isa IR._Position64
        IR._Position64(IR._max_block(rng), bit)
    else
        IR._Position128(typemax(UInt64), typemax(UInt64), bit)
    end
    return IR._rebuild(rng, position, rng.device)
end

@testset "R13 and R28 packed normal raw golden vectors" begin
    # The preserved revision-13 C++ oracle was adapted only to select
    # FAMILY_NORMAL and emit 23-bit, 52-bit, and midpoint values. The files
    # PureRNGs-normal-oracle.cpp and PureRNGs-normal-oracle.out live
    # beside the preserved oracle outside Git. Their file SHA-256 hashes are
    # c1917e39610fbc65b6b91872c9eb33fd7103f955da275913516a4b00df90226b and
    # 76d0fd9045667d5df5267a2207a321ad58fbd92b8724feababca92dee8f2997b.
    # Their payload SHA-256 hashes are
    # 7099d627be91f7a7583be82e09b310bcd3db83cf3d739038bd35ee6f47cdaa86 and
    # 77e6f0348d5f391058937a2ad0dc21bb3435d6f809adb6e996d9915ff3c6887b.
    # The adapter includes the preserved source with payload SHA-256
    # 68a432f195091fff370c858b5ef80617ec3bf3399ff9db8bfa4cc06ca753f5e3.
    expected = (
        (0x72d8ec, 0x0e5b1d8e28fcd5, 0x3f65b1d9, 0x3fecb63b1c51f9ab),
        (0x788a42, 0x0f11485c914b5b, 0x3f711485, 0x3fee2290b92296b7),
        (0x33ab13, 0x067562689937a8, 0x3eceac4e, 0x3fd9d589a264dea2),
        (0x0c5af3, 0x018b5e633ebdeb, 0x3dc5af38, 0x3fb8b5e633ebdeb8),
        (0x6ebd7e, 0x0dd7afcc2cf972, 0x3f5d7afd, 0x3febaf5f9859f2e5),
        (0x7c45dd, 0x0f88bbad99b74d, 0x3f788bbb, 0x3fef11775b336e9b),
        (0x6baf7c, 0x0d75ef88d35261, 0x3f575ef9, 0x3feaebdf11a6a4c3),
        (0x53948c, 0x0a72918c5d912e, 0x3f272919, 0x3fe4e52318bb225d),
    )

    for ((F, key), (raw32, raw64, midpoint32, midpoint64)) in
        zip(PACKED_GOLDEN_FAMILIES, expected)
        rng = _packed_golden_rng(F, key)
        block = _reference_position_block(rng.position)
        got32 = IR._extract_bits_unchecked(
            rng,
            IR.FAMILY_NORMAL,
            block,
            rng.position.bit,
            Val(23),
        )
        got64 = IR._extract_bits_unchecked(
            rng,
            IR.FAMILY_NORMAL,
            block,
            rng.position.bit,
            Val(52),
        )
        @test got32 === UInt64(raw32)
        @test got64 === UInt64(raw64)
        @test reinterpret(UInt32, IR._normal_midpoint(Float32, got32)) === midpoint32
        @test reinterpret(UInt64, IR._normal_midpoint(Float64, got64)) === midpoint64
    end
end

@testset "R28 AS241 definition" begin
    @test IR._as241_coefficients(Float32) === REFERENCE_AS241_32
    @test IR._as241_coefficients(Float64) === REFERENCE_AS241_64

    for (T, inputs) in
        ((Float32, (0.5f0, 0.95f0, 1.0f-12)), (Float64, (0.5, 0.95, 1.0e-20)))
        for u in inputs
            @test IR._as241(u) === _reference_as241(u)
        end
    end

    @test IR._normal_midpoint(Float32, UInt64(0)) === Float32(0x1p-24)
    @test IR._normal_midpoint(Float32, UInt64((UInt64(1) << 23) - 1)) ===
          one(Float32) - Float32(0x1p-24)
    @test IR._normal_midpoint(Float64, UInt64(0)) === Float64(0x1p-53)
    @test IR._normal_midpoint(Float64, (UInt64(1) << 52) - 1) ===
          one(Float64) - Float64(0x1p-53)
end

@testset "R28 packed scalar normals" begin
    for F in SCALAR_FAMILIES, T in NORMAL_TYPES
        for bit in (UInt16(0), UInt16(23), UInt16(51), UInt16(63))
            rng = _positioned(F, 0x742, UInt64(9), bit)
            expected = _reference_normal(rng, T)
            @test randn(rng, T) === expected
            @test randn(rng, T) === expected

            next_rng, value = randn_next(rng, T)
            @test value === expected
            @test next_rng.position == _reference_position(rng, _normal_width(T))
            @test rng.position != next_rng.position
            @test randnat(rng, T, 1) === expected
            @test randnat(rng, T, 3) === _reference_normal(
                IR._rebuild(
                    rng,
                    _reference_position(rng, 2 * _normal_width(T)),
                    rng.device,
                ),
                T,
            )
        end
    end
end

@testset "R8 and R53 normal family, capacity, and mixed positions" begin
    for F in SCALAR_FAMILIES, T in NORMAL_TYPES
        rng = _positioned(F, 0x743, UInt64(4), UInt16(61))
        normal_raw = _reference_extract(
            rng,
            IR.FAMILY_NORMAL,
            _reference_position_block(rng.position),
            rng.position.bit,
            _normal_width(T),
        )
        uniform_raw = _reference_extract(
            rng,
            IR.FAMILY_BITS,
            _reference_position_block(rng.position),
            rng.position.bit,
            _normal_width(T),
        )
        @test normal_raw != uniform_raw

        next_rng, _ = randn_next(rng, T)
        @test next_rng.position == _reference_position(rng, _normal_width(T))
        final_rng, final_rng_value = rand_next(next_rng, UInt32)
        @test final_rng_value === rand(next_rng, UInt32)
        @test final_rng.position ==
              _reference_position(rng, _normal_width(T) + _uniform_width(UInt32))

        terminal_rng = _terminal_normal_rng(F, T)
        exhausted, value = randn_next(terminal_rng, T)
        @test value === randn(terminal_rng, T)
        @test exhausted.position.bit === IR._EXHAUSTED_BIT
        @test_throws ArgumentError randn(exhausted, T)
        @test_throws ArgumentError randn_next(exhausted, T)
        @test_throws ArgumentError randnat(terminal_rng, T, 2)
        @test_throws ArgumentError randnat(exhausted, T, 1)
    end
end

@testset "R30 Position128 normal low-limb carry" begin
    for F in (Philox4x64, Threefry4x64), T in NORMAL_TYPES
        rng = F(0x746)
        bit = UInt16(IR._block_bits(rng) - _normal_width(T))
        position = IR._Position128(typemax(UInt64), UInt64(7), bit)
        rng = IR._rebuild(rng, position, rng.device)
        next_rng, _ = randn_next(rng, T)
        @test next_rng.position == IR._Position128(UInt64(0), UInt64(8), UInt16(0))
    end
end

@testset "R23 and R30 scalar normal methods, inference, allocation, and IR" begin
    for F in SCALAR_FAMILIES
        rng = F(0x744)
        default_next, default_value = randn_next(rng)
        typed_next, typed_value = randn_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value

        error = try
            randn(rng)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("randn(rng, T)", error.msg)

        for T in NORMAL_TYPES
            @test which(randn, (typeof(rng), Type{T})).module === IR
            @test which(randn_next, (typeof(rng), Type{T})).module === IR
            @test which(randnat, (typeof(rng), Type{T}, Int)).module === IR

            @test @inferred(randn(rng, T)) isa T
            @test @inferred(randn_next(rng, T)) isa Tuple{typeof(rng),T}
            @test @inferred(randnat(rng, T, 3)) isa T

            randn(rng, T)
            randn_next(rng, T)
            randnat(rng, T, 3)
            @test @allocated(randn(rng, T)) == 0
            @test @allocated(randn_next(rng, T)) == 0
            @test @allocated(randnat(rng, T, 3)) == 0
        end
    end

    rng = Philox4x64(0x745)
    for (function_, signature) in (
        (randn, Tuple{typeof(rng),Type{Float64}}),
        (randn_next, Tuple{typeof(rng),Type{Float32}}),
        (randnat, Tuple{typeof(rng),Type{Float64},Int}),
    )
        typed_ir = sprint(show, code_typed(function_, signature; optimize = true))
        llvm_ir = sprint() do io
            code_llvm(
                io,
                function_,
                signature;
                raw = false,
                dump_module = false,
                optimize = true,
            )
        end
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        @test !occursin(r"\bi128\b", llvm_ir)
    end

    for unsupported in (Union{Float32,Float64}, Float16, Int32, UInt64)
        @test !applicable(randn, rng, unsupported)
        @test !applicable(randn_next, rng, unsupported)
        @test !applicable(randnat, rng, unsupported, 1)
    end

    @test_throws ArgumentError randnat(rng, Float64, 0)
    @test_throws ArgumentError randnat(rng, Float32, -1)
end
