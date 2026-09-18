using InteractiveUtils: code_llvm
using Random: randn, randn!

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
    @allocated(randn_at(rng, T, 3)),
)

function _serial_normal_fill_allocations(rng, destination)
    randn_next!(rng, destination; threaded = false)
    return @allocated randn_next!(rng, destination; threaded = false)
end

@testset "R13 and R28 packed normal raw golden vectors" begin
    # Normal draws read the uniform stream, so the 23-bit and 52-bit raws are
    # prefixes of the uniform golden block at this position. The midpoint and
    # result bits pin the AS241 transform.
    # Public normal result bits were captured on aarch64 macOS. R43 permits
    # final AS241 results to differ elsewhere.
    exact_public_results =
        (Sys.ARCH, Sys.KERNEL) in ((:aarch64, :Darwin), (:x86_64, :Linux))
    expected = (
        (
            0x79cb04,
            0x0f39608ad54873,
            0x3f739609,
            0x3fee72c115aa90e7,
            0x3fd46f94,
            0x3ffa8df2a6e3179a,
        ),
        (
            0x1014e4,
            0x02029c87b4a20b,
            0x3e00a724,
            0x3fc014e43da5105c,
            0xbf92d956,
            0xbff25b2aeade0df4,
        ),
        (
            0x1516ac,
            0x02a2d596a681fd,
            0x3e28b564,
            0x3fc516acb5340fec,
            0xbf79a063,
            0xbfef340c238deb88,
        ),
        (
            0x312659,
            0x0624cb22e36a26,
            0x3ec49966,
            0x3fd8932c8b8da89a,
            0xbe970f14,
            0xbfd2e1e3148ee9ef,
        ),
        (
            0x6a587f,
            0x0d4b0fe71e0a71,
            0x3f54b0ff,
            0x3fea961fce3c14e3,
            0x3f751a5c,
            0x3feea34b41c59b6e,
        ),
        (
            0x06bfc7,
            0x00d7f8e4de9196,
            0x3d57f8f0,
            0x3faaff1c9bd232d0,
            0xbfcf3a2a,
            0xbff9e745e1e9dad7,
        ),
        (
            0x2269c9,
            0x044d3930c5cae4,
            0x3e89a726,
            0x3fd134e4c3172b92,
            0xbf1dc4d4,
            0xbfe3b89a5cbc295a,
        ),
        (
            0x36e0cd,
            0x06dc19acb71673,
            0x3edb8336,
            0x3fdb7066b2dc59ce,
            0xbe37e7a4,
            0xbfc6fcf4af71978a,
        ),
    )

    for ((F, key), (raw32, raw64, midpoint32, midpoint64, normal32, normal64)) in
        zip(PACKED_GOLDEN_GENERATORS, expected)
        rng = _packed_golden_rng(F, key)
        block = _reference_position_block(rng.position)
        got32 = IR._extract_bits_unchecked(rng, block, rng.position.bit, Val(23))
        got64 = IR._extract_bits_unchecked(rng, block, rng.position.bit, Val(52))
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
    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        for bit in (UInt16(0), UInt16(23), UInt16(51), UInt16(63))
            rng = _positioned(F, 0x742, UInt64(9), bit)
            expected = _reference_normal(rng, T)
            @test randn(rng, T) === expected

            value, next_rng = randn_next(rng, T)
            @test value === expected
            @test next_rng.position == _reference_position(rng, _normal_width(T))
            @test randn_at(rng, T, 1) === expected
            @test randn_at(rng, T, 3) === _reference_normal(
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

@testset "R8 and R53 normal generator, capacity, and mixed positions" begin
    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        rng = _positioned(F, 0x743, UInt64(4), UInt16(61))
        _, next_rng = randn_next(rng, T)
        @test next_rng.position == _reference_position(rng, _normal_width(T))
        final_rng_value, final_rng = rand_next(next_rng, UInt32)
        @test final_rng_value === rand(next_rng, UInt32)
        @test final_rng.position ==
              _reference_position(rng, _normal_width(T) + _uniform_width(UInt32))

        terminal_rng = _terminal_normal_rng(F, T)
        value, exhausted = randn_next(terminal_rng, T)
        @test value === randn(terminal_rng, T)
        @test exhausted.position.bit === IR._EXHAUSTED_BIT
        @test_throws StreamExhausted randn(exhausted, T)
        @test_throws StreamExhausted randn_next(exhausted, T)
        @test_throws StreamExhausted randn_at(terminal_rng, T, 2)
        @test_throws StreamExhausted randn_at(exhausted, T, 1)
    end
end

@testset "R30 Position128 normal low-word carry" begin
    for F in (Philox4x64, Threefry4x64), T in NORMAL_TYPES
        rng = F(0x746)
        bit = UInt16(IR._block_bits(rng) - _normal_width(T))
        position = IR._Position128(typemax(UInt64), UInt64(7), bit)
        rng = IR._rebuild(rng, position, rng.device)
        _, next_rng = randn_next(rng, T)
        @test next_rng.position == IR._Position128(UInt64(0), UInt64(8), UInt16(0))
    end
end

@testset "R23 and R30 scalar normal fixed-work and codegen" begin
    for F in GENERATOR_TYPES
        rng = F(0x744)
        default_value, default_next = randn_next(rng)
        typed_value, typed_next = randn_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value

        @test_throws ArgumentError randn(rng)

        for T in NORMAL_TYPES
            normal_allocations(rng, T)
            @test normal_allocations(rng, T) == (0, 0, 0)
        end
    end

    rng = IR.MLDataDevices.CUDADevice()(Philox4x64(0x745))
    for (function_, signature) in (
        (randn, Tuple{typeof(rng),Type{Float64}}),
        (randn_next, Tuple{typeof(rng),Type{Float32}}),
        (randn_at, Tuple{typeof(rng),Type{Float64},Int}),
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

    @test_throws ArgumentError randn_at(rng, Float64, 0)
    @test_throws ArgumentError randn_at(rng, Float32, -1)
end

@testset "R23, R24, and R26 packed normal fills and allocations" begin
    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        for bit in (UInt16(0),)
            rng = _positioned(F, 0x747, UInt64(6), bit)
            next_rng, expected = _reference_normal_chain(rng, T, 17)

            serial = Vector{T}(undef, 17)
            threaded = similar(serial)
            @test randn!(rng, serial; threaded = false) === serial
            @test randn!(rng, threaded; threaded = true) === threaded
            @test serial == threaded == expected
            @test rng.position.bit == bit

            continued, continued_next = randn_next!(rng, similar(serial))
            @test continued == expected
            @test continued_next.position == next_rng.position

            matrix = randn(rng, T, 1, 17)
            @test vec(matrix) == expected
            @test size(matrix) == (1, 17)
            @test rng.position.bit == bit

            allocated, allocated_next = randn_next(rng, T, 17)
            @test allocated == expected
            @test allocated_next.position == next_rng.position
        end

        rng = _positioned(F, 0x748, UInt64(3), UInt16(61))
        next_rng, expected = _reference_normal_chain(rng, T, 12)
        storage = fill(zero(T), 24)
        destination = @view storage[2:2:24]
        returned, view_next = randn_next!(rng, destination; threaded = false)
        @test returned === destination
        @test collect(destination) == expected
        @test all(iszero, @view storage[1:2:23])
        @test view_next.position == next_rng.position

        threaded_storage = fill(zero(T), 24)
        threaded_view = @view threaded_storage[2:2:24]
        randn!(rng, threaded_view; threaded = true)
        @test collect(threaded_view) == expected
        @test all(iszero, @view threaded_storage[1:2:23])
    end

    for T in NORMAL_TYPES
        bit = UInt16(127)
        rng = _positioned(Philox4x32, 0x747, UInt64(6), bit)
        expected_rng, expected = _reference_normal_chain(rng, T, 17)
        destination, next_rng = randn_next!(rng, Vector{T}(undef, 17); threaded = false)
        @test destination == expected
        @test next_rng.position == expected_rng.position
    end

    rng = Philox4x32(0x749)
    default_values, default_next = randn_next(rng, 2, 3)
    typed_values, typed_next = randn_next(rng, Float64, 2, 3)
    @test default_values == typed_values
    @test default_next.position == typed_next.position
    @test size(default_values) == (2, 3)
    @test_throws ArgumentError randn(rng, Float32, -1)
    @test_throws ArgumentError randn_next(rng, -1)

end

@testset "R23 and R26 Philox4x32 grouped normal fill stream" begin
    # The Philox4x32 fill decodes whole groups of aligned draws, so the stream
    # has to match the scalar chain on either side of a group boundary and at a
    # start bit no group can align to.
    for T in NORMAL_TYPES, count in (1, 31, 32, 33, 1000, 100_003)
        for bit in (UInt16(0), UInt16(3))
            rng = _positioned(Philox4x32, 0x74b, UInt64(9), bit)
            expected_rng, expected = _reference_normal_chain(rng, T, count)
            serial = Vector{T}(undef, count)
            _, serial_next = randn_next!(rng, serial; threaded = false)
            threaded = similar(serial)
            randn!(rng, threaded; threaded = true)
            @test serial == expected
            @test threaded == expected
            @test serial_next.position == expected_rng.position
        end
    end
end

@testset "R26 normal fill parallel seams and caller task" begin
    for T in NORMAL_TYPES
        rng = _positioned(Philox4x32, 0x74a, UInt64(4), UInt16(61))
        chunk_elements = IR._transformed_fill_chunk_elements(Val(:normal), T)
        for delta in (-1, 0, 1)
            count = 4chunk_elements + delta
            serial = Vector{T}(undef, count)
            threaded = similar(serial)
            _, serial_next = randn_next!(rng, serial; threaded = false)
            _, threaded_next = randn_next!(rng, threaded; threaded = true)
            @test threaded == serial
            @test threaded_next.position == serial_next.position
        end
    end

    rng = _positioned(Philox4x32, 0x74b, UInt64(3), UInt16(29))
    caller = current_task()
    probe = TaskWriteProbe(Vector{Float64}(undef, 37))
    returned, next_rng = randn_next!(rng, probe; threaded = false)
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
        values, next_rng = randn_next(rng, Float64, count)
        @test values == expected
        @test next_rng.position == expected_next.position
    end
end

@testset "R30, R39, R40, and R54 normal fill validation" begin
    rng = Philox4x32(0x74c)
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    empty = Float32[]
    @test randn!(exhausted, empty; threaded = false) === empty
    empty_result, empty_next = randn_next!(exhausted, empty; threaded = false)
    @test empty_result === empty
    @test empty_next === exhausted
    @test isempty(randn(exhausted, Float32, 0))
    allocated_empty, allocated_empty_next = randn_next(exhausted, Float32, 0)
    @test isempty(allocated_empty)
    @test allocated_empty_next === exhausted
    default_empty, default_empty_next = randn_next(exhausted, 0)
    @test isempty(default_empty)
    @test eltype(default_empty) === Float64
    @test default_empty_next === exhausted

    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        last = _terminal_normal_rng(F, T)
        destination = fill(one(T), 2)
        before = copy(destination)
        @test_throws StreamExhausted randn!(last, destination; threaded = false)
        @test destination == before
        @test_throws StreamExhausted randn_next!(last, destination; threaded = false)
        @test destination == before

        final = Vector{T}(undef, 1)
        _, final_next = randn_next!(last, final; threaded = false)
        @test final[1] === _reference_normal(last, T)
        expected_terminal =
            last.position isa IR._Position64 ? IR._terminal64(IR._max_block(last)) :
            IR._terminal128()
        @test final_next.position == expected_terminal

        pure_final = randn(last, T, 1)
        allocating_final, allocating_final_next = randn_next(last, T, 1)
        @test pure_final == allocating_final == final
        @test allocating_final_next.position == expected_terminal

        insufficient_position = if last.position isa IR._Position64
            IR._Position64(IR._max_block(last), last.position.bit + UInt16(1))
        else
            IR._Position128(typemax(UInt64), typemax(UInt64), last.position.bit + UInt16(1))
        end
        insufficient = IR._rebuild(last, insufficient_position, last.device)
        @test_throws StreamExhausted randn(insufficient, T, 1)
        @test_throws StreamExhausted randn_next(insufficient, T, 1)
    end

end

@testset "R23 and R30 normal fill fixed-work and codegen" begin
    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        rng = F(0x74d)
        destination = Vector{T}(undef, 7)
        @test _serial_normal_fill_allocations(rng, destination) == 0
    end

    rng = IR.MLDataDevices.CUDADevice()(Philox4x64(0x74e))
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
