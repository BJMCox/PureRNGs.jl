using InteractiveUtils: code_llvm
using Random: randn, randn!

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

function _reference_normal_chain(rng, ::Type{T}, count::Int) where {T}
    values = Vector{T}(undef, count)
    cursor = rng
    for index in eachindex(values)
        values[index] = _reference_normal(cursor, T)
        cursor = IR._rebuild(
            cursor,
            _reference_position(cursor, _normal_width(T)),
            cursor.device,
        )
    end
    return cursor, values
end

normal_allocations(rng, ::Type{T}) where {T} = (
    @allocated(randn(rng, T)),
    @allocated(randn_next(rng, T)),
    @allocated(randnat(rng, T, 3)),
)

function _serial_normal_fill_allocations(rng, destination)
    randn_next!(rng, destination; threaded = false)
    return @allocated randn_next!(rng, destination; threaded = false)
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
    # Pinned testbed commit for unchanged cores and layouts:
    # 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d.
    # Public normal result bits were captured on aarch64 macOS and verified on
    # x86-64 Linux. R43 permits final AS241 results to differ elsewhere.
    exact_public_results =
        (Sys.ARCH, Sys.KERNEL) in ((:aarch64, :Darwin), (:x86_64, :Linux))
    expected = (
        (
            0x72d8ec,
            0x0e5b1d8e28fcd5,
            0x3f65b1d9,
            0x3fecb63b1c51f9ab,
            0x3fa20c90,
            0x3ff4419224daef1f,
        ),
        (
            0x788a42,
            0x0f11485c914b5b,
            0x3f711485,
            0x3fee2290b92296b7,
            0x3fc8e12b,
            0x3ff91c2631cb2053,
        ),
        (
            0x33ab13,
            0x067562689937a8,
            0x3eceac4e,
            0x3fd9d589a264dea2,
            0xbe79be14,
            0xbfcf37c332c6479a,
        ),
        (
            0x0c5af3,
            0x018b5e633ebdeb,
            0x3dc5af38,
            0x3fb8b5e633ebdeb8,
            0xbfa69b03,
            0xbff4d360c3607e53,
        ),
        (
            0x6ebd7e,
            0x0dd7afcc2cf972,
            0x3f5d7afd,
            0x3febaf5f9859f2e5,
            0x3f8d48ff,
            0x3ff1a91fc02d3427,
        ),
        (
            0x7c45dd,
            0x0f88bbad99b74d,
            0x3f788bbb,
            0x3fef11775b336e9b,
            0x3ff26bf0,
            0x3ffe4d7dd9742d79,
        ),
        (
            0x6baf7c,
            0x0d75ef88d35261,
            0x3f575ef9,
            0x3feaebdf11a6a4c3,
            0x3f7ff1f7,
            0x3feffe3e9f3eaeba,
        ),
        (
            0x53948c,
            0x0a72918c5d912e,
            0x3f272919,
            0x3fe4e52318bb225d,
            0x3ec965a5,
            0x3fd92cb4a0df0afc,
        ),
    )

    for ((F, key), (raw32, raw64, midpoint32, midpoint64, normal32, normal64)) in
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
        if exact_public_results
            @test reinterpret(UInt32, randn(rng, Float32)) === normal32
            @test reinterpret(UInt64, randn(rng, Float64)) === normal64
        end
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
    for F in FAMILY_TYPES, T in NORMAL_TYPES
        for bit in (UInt16(0), UInt16(23), UInt16(51), UInt16(63))
            rng = _positioned(F, 0x742, UInt64(9), bit)
            expected = _reference_normal(rng, T)
            @test randn(rng, T) === expected

            next_rng, value = randn_next(rng, T)
            @test value === expected
            @test next_rng.position == _reference_position(rng, _normal_width(T))
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
    for F in FAMILY_TYPES, T in NORMAL_TYPES
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

@testset "R23 and R30 scalar normal fixed-work and codegen" begin
    for F in FAMILY_TYPES
        rng = F(0x744)
        default_next, default_value = randn_next(rng)
        typed_next, typed_value = randn_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value

        @test_throws ArgumentError randn(rng)

        for T in NORMAL_TYPES
            normal_allocations(rng, T)
            @test normal_allocations(rng, T) == (0, 0, 0)
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

    @test_throws ArgumentError randnat(rng, Float64, 0)
    @test_throws ArgumentError randnat(rng, Float32, -1)
end

@testset "R23, R24, and R26 packed normal fills and allocations" begin
    for F in FAMILY_TYPES, T in NORMAL_TYPES
        for bit in (UInt16(0),)
            rng = _positioned(F, 0x747, UInt64(6), bit)
            next_rng, expected = _reference_normal_chain(rng, T, 17)

            serial = Vector{T}(undef, 17)
            threaded = similar(serial)
            @test randn!(rng, serial; threaded = false) === serial
            @test randn!(rng, threaded; threaded = true) === threaded
            sync_cpu()
            @test serial == threaded == expected
            @test rng.position.bit == bit

            continued_next, continued = randn_next!(rng, similar(serial))
            sync_cpu()
            @test continued == expected
            @test continued_next.position == next_rng.position

            matrix = randn(rng, T, 1, 17)
            sync_cpu()
            @test vec(matrix) == expected
            @test size(matrix) == (1, 17)
            @test rng.position.bit == bit

            allocated_next, allocated = randn_next(rng, T, 17)
            sync_cpu()
            @test allocated == expected
            @test allocated_next.position == next_rng.position
        end

        rng = _positioned(F, 0x748, UInt64(3), UInt16(61))
        next_rng, expected = _reference_normal_chain(rng, T, 12)
        storage = fill(zero(T), 24)
        destination = @view storage[2:2:24]
        view_next, returned = randn_next!(rng, destination; threaded = false)
        @test returned === destination
        @test collect(destination) == expected
        @test all(iszero, @view storage[1:2:23])
        @test view_next.position == next_rng.position

        threaded_storage = fill(zero(T), 24)
        threaded_view = @view threaded_storage[2:2:24]
        randn!(rng, threaded_view; threaded = true)
        sync_cpu()
        @test collect(threaded_view) == expected
        @test all(iszero, @view threaded_storage[1:2:23])
    end

    for T in NORMAL_TYPES
        bit = UInt16(127)
        rng = _positioned(Philox4x32, 0x747, UInt64(6), bit)
        expected_rng, expected = _reference_normal_chain(rng, T, 17)
        next_rng, destination = randn_next!(rng, Vector{T}(undef, 17); threaded = false)
        @test destination == expected
        @test next_rng.position == expected_rng.position
    end

    rng = Philox4x32(0x749)
    default_next, default_values = randn_next(rng, 2, 3)
    typed_next, typed_values = randn_next(rng, Float64, 2, 3)
    sync_cpu()
    @test default_values == typed_values
    @test default_next.position == typed_next.position
    @test size(default_values) == (2, 3)
    @test_throws ArgumentError randn(rng, Float32, -1)
    @test_throws ArgumentError randn_next(rng, -1)

end

@testset "R26 normal fill parallel seams and caller task" begin
    for T in NORMAL_TYPES
        rng = _positioned(Philox4x32, 0x74a, UInt64(4), UInt16(61))
        chunk_elements = IR._transformed_fill_chunk_elements(Val(:normal), T)
        for delta in (-1, 0, 1)
            count = 4chunk_elements + delta
            serial = Vector{T}(undef, count)
            threaded = similar(serial)
            serial_next, _ = randn_next!(rng, serial; threaded = false)
            threaded_next, _ = randn_next!(rng, threaded; threaded = true)
            sync_cpu()
            @test threaded == serial
            @test threaded_next.position == serial_next.position
        end
    end

    rng = _positioned(Philox4x32, 0x74b, UInt64(3), UInt16(29))
    caller = current_task()
    probe = TaskWriteProbe(Vector{Float64}(undef, 37))
    next_rng, returned = randn_next!(rng, probe; threaded = false)
    expected_rng, expected = _reference_normal_chain(rng, Float64, 37)
    @test returned === probe
    @test all(task -> task === caller, probe.writers)
    @test probe.data == expected
    @test next_rng.position == expected_rng.position
end

@testset "R26 small allocating normal boundary" begin
    rng = _positioned(Philox4x32, 0x74b1, UInt64(5), UInt16(61))
    for count in (128, 129)
        expected_next, expected = _reference_normal_chain(rng, Float64, count)
        next_rng, values = randn_next(rng, Float64, count)
        sync_cpu()
        @test values == expected
        @test next_rng.position == expected_next.position
    end
end

@testset "R30, R39, R40, and R54 normal fill validation" begin
    rng = Philox4x32(0x74c)
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    empty = Float32[]
    @test randn!(exhausted, empty; threaded = false) === empty
    empty_next, empty_result = randn_next!(exhausted, empty; threaded = false)
    @test empty_result === empty
    @test empty_next === exhausted
    @test isempty(randn(exhausted, Float32, 0))
    allocated_empty_next, allocated_empty = randn_next(exhausted, Float32, 0)
    @test isempty(allocated_empty)
    @test allocated_empty_next === exhausted
    default_empty_next, default_empty = randn_next(exhausted, 0)
    @test isempty(default_empty)
    @test eltype(default_empty) === Float64
    @test default_empty_next === exhausted

    for F in FAMILY_TYPES, T in NORMAL_TYPES
        last = _terminal_normal_rng(F, T)
        destination = fill(one(T), 2)
        before = copy(destination)
        @test_throws ArgumentError randn!(last, destination; threaded = false)
        @test destination == before
        @test_throws ArgumentError randn_next!(last, destination; threaded = false)
        @test destination == before

        final = Vector{T}(undef, 1)
        final_next, _ = randn_next!(last, final; threaded = false)
        @test final[1] === _reference_normal(last, T)
        expected_terminal =
            last.position isa IR._Position64 ? IR._terminal64(IR._max_block(last)) :
            IR._terminal128()
        @test final_next.position == expected_terminal

        pure_final = randn(last, T, 1)
        allocating_final_next, allocating_final = randn_next(last, T, 1)
        sync_cpu()
        @test pure_final == allocating_final == final
        @test allocating_final_next.position == expected_terminal

        insufficient_position = if last.position isa IR._Position64
            IR._Position64(IR._max_block(last), last.position.bit + UInt16(1))
        else
            IR._Position128(typemax(UInt64), typemax(UInt64), last.position.bit + UInt16(1))
        end
        insufficient = IR._rebuild(last, insufficient_position, last.device)
        @test_throws ArgumentError randn(insufficient, T, 1)
        @test_throws ArgumentError randn_next(insufficient, T, 1)
    end

end

@testset "R23 and R30 normal fill fixed-work and codegen" begin
    for F in FAMILY_TYPES, T in NORMAL_TYPES
        rng = F(0x74d)
        destination = Vector{T}(undef, 7)
        @test _serial_normal_fill_allocations(rng, destination) == 0
    end

    rng = Philox4x64(0x74e)
    destination = Vector{Float64}(undef, 7)
    for (function_, call_signature) in
        ((randn_next!, Tuple{typeof(rng),typeof(destination)}),)
        typed_ir = sprint(show, code_typed(function_, call_signature; optimize = true))
        llvm_ir = sprint() do io
            code_llvm(
                io,
                function_,
                call_signature;
                raw = false,
                dump_module = false,
                optimize = true,
            )
        end
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        @test !occursin(r"\bi128\b", llvm_ir)
    end

end
