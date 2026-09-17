using InteractiveUtils: code_llvm
using Random: rand!

const SCALAR_UNIFORM_TYPES = (Bool, UInt32, UInt64, Float32, Float64)

function _reference_chain(rng, ::Type{T}, count::Int) where {T}
    values = Vector{T}(undef, count)
    cursor = rng
    for index in eachindex(values)
        values[index] = _reference_uniform(cursor, T)
        cursor = IR._rebuild(
            cursor,
            _reference_position(cursor, _uniform_width(T)),
            cursor.device,
        )
    end
    return cursor, values
end

uniform_allocations(rng, ::Type{T}) where {T} =
    (@allocated(rand(rng, T)), @allocated(rand_next(rng, T)), @allocated(randat(rng, T, 3)))

function _serial_fill_allocations(rng, destination)
    rand_next!(rng, destination; threaded = false)
    return @allocated rand_next!(rng, destination; threaded = false)
end

@testset "R13 packed uniform golden vectors" begin
    # Independent C++17 oracle using DEShawResearch/random123 v1.14.0:
    # commit 726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13;
    # include/Random123/philox.h SHA-256
    # 6c2ef219a855885499a73b338d5f41dafe079618b2dae2f60ea86ee785d771e2;
    # include/Random123/threefry.h SHA-256
    # 4c210b32b5ba605b059c54d5edd6f01bf04190de49a0abeecec76420cd072a72.
    # Revision-13 layouts and MSB-first extraction were transcribed directly;
    # the oracle never imports package code. Oracle source SHA-256
    # 68a432f195091fff370c858b5ef80617ec3bf3399ff9db8bfa4cc06ca753f5e3;
    # output SHA-256
    # 75cb899f9daf671f3b59be5c7602a96eb969f10188ed71c8d011da484c8069fe.
    # Pinned testbed commit for unchanged cores and layouts:
    # 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d.
    expected = (
        (true, 0xf39608ad, 0xf39608ad5487335f, 0x3f739608, 0x3fee72c115aa90e6),
        (false, 0x2029c87b, 0x2029c87b4a20bd10, 0x3e00a720, 0x3fc014e43da5105c),
        (false, 0x2a2d596a, 0x2a2d596a681fd002, 0x3e28b564, 0x3fc516acb5340fe8),
        (false, 0x624cb22e, 0x624cb22e36a263ea, 0x3ec49964, 0x3fd8932c8b8da898),
        (true, 0xd4b0fe71, 0xd4b0fe71e0a7135a, 0x3f54b0fe, 0x3fea961fce3c14e2),
        (false, 0x0d7f8e4d, 0x0d7f8e4de91962b4, 0x3d57f8e0, 0x3faaff1c9bd232c0),
        (false, 0x44d3930c, 0x44d3930c5cae4976, 0x3e89a726, 0x3fd134e4c3172b92),
        (false, 0x6dc19acb, 0x6dc19acb71673eb8, 0x3edb8334, 0x3fdb7066b2dc59ce),
    )

    for ((F, key), (bit, word32, word64, float32_bits, float64_bits)) in
        zip(PACKED_GOLDEN_GENERATORS, expected)
        rng = _packed_golden_rng(F, key)
        @test rand(rng, Bool) === bit
        @test rand(rng, UInt32) === word32
        @test rand(rng, Int32) === reinterpret(Int32, word32)
        @test rand(rng, UInt64) === word64
        @test rand(rng, Int64) === reinterpret(Int64, word64)
        @test reinterpret(UInt32, rand(rng, Float32)) === float32_bits
        @test reinterpret(UInt64, rand(rng, Float64)) === float64_bits
    end
end

@testset "R25 and R53 packed primitive widths" begin
    for (T, width) in zip(PURE_UNIFORM_TYPES, (1, 32, 32, 64, 64, 24, 53))
        @test IR._draw_bits(T) === UInt16(width)
    end

    for F in GENERATOR_TYPES, T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x521, UInt64(9), UInt16(61))
        position = rng.position
        expected = _reference_uniform(rng, T)
        @test rand(rng, T) === expected
        @test rng.position === position

        value, next_rng = rand_next(rng, T)
        @test value === expected
        @test next_rng.position == _reference_position(rng, _uniform_width(T))
    end
end

@testset "R25 signed primitive bitcasts" begin
    for F in GENERATOR_TYPES, (S, U) in ((Int32, UInt32), (Int64, UInt64))
        rng = _positioned(F, 0x5211, UInt64(9), UInt16(61))
        @test reinterpret(U, rand(rng, S)) === rand(rng, U)

        signed_value, signed_next = rand_next(rng, S)
        unsigned_value, unsigned_next = rand_next(rng, U)
        @test reinterpret(U, signed_value) === unsigned_value
        @test signed_next.position == unsigned_next.position

        @test reinterpret(U, randat(rng, S, 7)) === randat(rng, U, 7)

        signed_fill = Vector{S}(undef, 129)
        unsigned_fill = Vector{U}(undef, 129)
        _, signed_fill_next = rand_next!(rng, signed_fill; threaded = false)
        _, unsigned_fill_next = rand_next!(rng, unsigned_fill; threaded = false)
        @test reinterpret(U, signed_fill) == unsigned_fill
        @test signed_fill_next.position == unsigned_fill_next.position
    end
end

@testset "R24, R26, and R53 mixed packed continuation" begin
    trace = (Bool, Float64, Float32, UInt64, UInt32, Bool, Float64)
    for F in GENERATOR_TYPES
        rng = _positioned(F, 0x522, UInt64(11), UInt16(63))
        cursor = rng
        for T in trace
            expected = _reference_uniform(cursor, T)
            expected_position = _reference_position(cursor, _uniform_width(T))
            value, cursor = rand_next(cursor, T)
            @test value === expected
            @test cursor.position == expected_position
        end
    end
end

@testset "R29 packed addressed draws" begin
    for F in GENERATOR_TYPES, T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x523, UInt64(5), UInt16(47))
        cursor = rng
        for i = 1:9
            expected = _reference_uniform(cursor, T)
            @test randat(rng, T, i) === expected
            cursor = IR._rebuild(
                cursor,
                _reference_position(cursor, _uniform_width(T)),
                cursor.device,
            )
        end
        @test rng.position.bit === UInt16(47)
        @test_throws ArgumentError randat(rng, T, 0)
        @test_throws ArgumentError randat(rng, T, -1)
    end

    for F in (Philox2x64, Threefry2x64)
        rng = F(0x523)
        final_index = big(1) << 65
        last = IR._rebuild(
            rng,
            IR._Position64(typemax(UInt64), IR._block_bits(rng) - UInt16(64)),
            rng.device,
        )
        @test randat(rng, UInt64, final_index) === rand(last, UInt64)
        @test_throws ArgumentError randat(rng, UInt64, final_index + 1)
    end

    for F in (Philox4x64, Threefry4x64)
        rng = F(0x523)
        final_index = big(1) << 130
        last = IR._rebuild(
            rng,
            IR._Position128(
                typemax(UInt64),
                typemax(UInt64),
                IR._block_bits(rng) - UInt16(64),
            ),
            rng.device,
        )
        @test randat(rng, UInt64, final_index) === rand(last, UInt64)
        @test_throws ArgumentError randat(rng, UInt64, final_index + 1)

        near_end = IR._rebuild(
            rng,
            IR._Position128(typemax(UInt64), typemax(UInt64), UInt16(0)),
            rng.device,
        )
        @test randat(near_end, UInt64, UInt64(4)) === rand(
            IR._rebuild(
                near_end,
                IR._Position128(typemax(UInt64), typemax(UInt64), UInt16(192)),
                rng.device,
            ),
            UInt64,
        )
        @test_throws ArgumentError randat(near_end, UInt64, UInt64(5))
    end
end

@testset "R29 addressed end-span preflight" begin
    base = Philox4x32(0x5231)
    last = IR._rebuild(
        base,
        IR._Position64(IR._max_block(base), IR._block_bits(base) - UInt16(32)),
        base.device,
    )
    @test randat(last, UInt32, UInt64(1)) === rand(last, UInt32)
    @test_throws ArgumentError randat(last, UInt32, UInt64(2))
end

@testset "R26 packed CPU fills, shapes, views, and BitArray" begin
    for F in GENERATOR_TYPES, T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x524, UInt64(7), UInt16(61))
        expected_rng, expected = _reference_chain(rng, T, 12)

        destination = Vector{T}(undef, 12)
        @test rand!(rng, destination) === destination
        sync_cpu()
        @test destination == expected
        @test rng.position.bit === UInt16(61)

        replay = similar(destination)
        returned, next_rng = rand_next!(rng, replay)
        sync_cpu()
        @test returned === replay
        @test replay == expected
        @test next_rng.position == expected_rng.position

        matrix = Matrix{T}(undef, 3, 4)
        matrix_result, matrix_next = rand_next!(rng, matrix; threaded = false)
        @test matrix_result === matrix
        @test vec(matrix) == expected
        @test matrix_next.position == expected_rng.position

        storage = fill(zero(T), 24)
        view_destination = @view storage[2:2:24]
        @test rand!(rng, view_destination; threaded = false) === view_destination
        @test collect(view_destination) == expected
        @test all(iszero, @view storage[1:2:23])

        threaded_storage = fill(zero(T), 24)
        threaded_view = @view threaded_storage[2:2:24]
        @test rand!(rng, threaded_view; threaded = true) === threaded_view
        sync_cpu()
        @test collect(threaded_view) == expected
        @test all(iszero, @view threaded_storage[1:2:23])
    end

    rng = _positioned(Philox4x32, 0x525, UInt64(4), UInt16(63))
    expected_rng, expected = _reference_chain(rng, Bool, 67)
    ordinary = BitArray(undef, 67)
    explicit = similar(ordinary)
    serial = similar(ordinary)
    ordinary_result, ordinary_next = rand_next!(rng, ordinary)
    explicit_result, explicit_next = rand_next!(rng, explicit; threaded = true)
    serial_result, serial_next = rand_next!(rng, serial; threaded = false)
    sync_cpu()
    @test ordinary_result === ordinary
    @test explicit_result === explicit
    @test serial_result === serial
    @test ordinary == explicit == serial == expected
    @test ordinary_next.position ==
          explicit_next.position ==
          serial_next.position ==
          expected_rng.position

end

@testset "R26 small allocating uniform boundary" begin
    rng = _positioned(Philox4x32, 0x5250, UInt64(5), UInt16(61))
    for count in (128, 129)
        expected_rng, expected = _reference_chain(rng, Float64, count)
        values, next_rng = rand_next(rng, Float64, count)
        sync_cpu()
        @test values == expected
        @test next_rng.position == expected_rng.position
    end
end

@testset "R26 parallel packed fills cross CPU chunks" begin
    for F in (Philox2x32, Philox4x32, Philox4x64), T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x5251, UInt64(4), UInt16(61))
        chunk_elements = IR._dense_fill_chunk_elements(T)
        count = 4chunk_elements + 3
        serial = Vector{T}(undef, count)
        threaded = similar(serial)

        serial_result, serial_next = rand_next!(rng, serial; threaded = false)
        threaded_result, threaded_next = rand_next!(rng, threaded; threaded = true)
        sync_cpu()

        expected_position = _reference_position(rng, count * _uniform_width(T))
        @test serial_result === serial
        @test threaded_result === threaded
        @test threaded == serial
        @test serial_next.position == threaded_next.position == expected_position

        for index in (chunk_elements, chunk_elements + 1, count)
            position = _reference_position(rng, (index - 1) * _uniform_width(T))
            cursor = IR._rebuild(rng, position, rng.device)
            @test threaded[index] === _reference_uniform(cursor, T)
        end
    end
end

@testset "Philox4x32 four-block dense fills" begin
    for T in PURE_UNIFORM_TYPES,
        bit in (UInt16(0), UInt16(127)),
        delta in (-1, 0, 1)

        x4_group =
            T === Bool ? 512 :
            T <: Union{Int32,UInt32} ? 16 :
            T <: Union{Int64,UInt64} ? 8 : T === Float32 ? 64 : 1
        group = max(
            IR._dense_fill_group(T),
            cld(4 * 128 - Int(bit), _uniform_width(T)),
            x4_group,
        )
        count = group + delta
        count < 0 && continue
        rng = _positioned(Philox4x32, 0x5254, UInt64(9), bit)
        expected_rng, expected = _reference_chain(rng, T, count)
        destination = Vector{T}(undef, count)
        result, next_rng = rand_next!(rng, destination; threaded = false)
        @test result === destination
        @test destination == expected
        @test next_rng.position == expected_rng.position
    end

    terminal_base = Philox4x32(0x5255)
    for (T, count, blocks) in (
        (Bool, 512, UInt64(4)),
        (UInt32, 16, UInt64(4)),
        (UInt64, 8, UInt64(4)),
        (Float32, 64, UInt64(12)),
    )
        position =
            IR._Position64(IR._max_block(terminal_base) - blocks + UInt64(1), UInt16(0))
        rng = IR._rebuild(terminal_base, position, terminal_base.device)
        expected = map(1:count) do index
            cursor = IR._rebuild(
                rng,
                _reference_position(rng, (index - 1) * _uniform_width(T)),
                rng.device,
            )
            _reference_uniform(cursor, T)
        end
        destination = Vector{T}(undef, count)
        result, next_rng = rand_next!(rng, destination; threaded = false)
        @test result == expected
        @test next_rng.position == IR._terminal64(IR._max_block(rng))
    end

end

@testset "R49 serial fills stay on the calling task" begin
    rng = _positioned(Philox4x32, 0x526, UInt64(3), UInt16(29))
    caller = current_task()
    probe = TaskWriteProbe(Vector{UInt32}(undef, 37))
    returned, next_rng = rand_next!(rng, probe; threaded = false)
    @test returned === probe
    @test all(task -> task === caller, probe.writers)
    expected_rng, expected = _reference_chain(rng, UInt32, 37)
    @test probe.data == expected
    @test next_rng.position == expected_rng.position
end

@testset "R39, R40, and R54 packed fill validation and preflight" begin
    rng = Philox4x32(0x527)
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    empty = UInt32[]
    @test rand!(exhausted, empty; threaded = false) === empty
    empty_result, empty_next = rand_next!(exhausted, empty; threaded = false)
    @test empty_result === empty
    @test empty_next === exhausted

    for F in GENERATOR_TYPES, T in PURE_UNIFORM_TYPES
        base = F(0x528)
        width = IR._draw_bits(T)
        position =
            base.position isa IR._Position64 ?
            IR._Position64(IR._max_block(base), IR._block_bits(base) - width) :
            IR._Position128(typemax(UInt64), typemax(UInt64), IR._block_bits(base) - width)
        last = IR._rebuild(base, position, base.device)
        destination = fill(convert(T, T === Bool ? true : 1), 2)
        before = copy(destination)
        @test_throws ArgumentError rand!(last, destination; threaded = false)
        @test destination == before
        @test_throws ArgumentError rand_next!(last, destination; threaded = false)
        @test destination == before
        @test last.position == position

        final = Vector{T}(undef, 1)
        final_result, final_next = rand_next!(last, final; threaded = false)
        @test final_result === final
        @test final[1] === _reference_uniform(last, T)
        expected_terminal =
            last.position isa IR._Position64 ? IR._terminal64(IR._max_block(last)) :
            IR._terminal128()
        @test final_next.position == expected_terminal
    end

end

@testset "R23 and R30 packed uniform fixed-work and codegen" begin
    for F in GENERATOR_TYPES
        rng = F(0x529)
        default_value, default_next = rand_next(rng)
        typed_value, typed_next = rand_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value

        for T in PURE_UNIFORM_TYPES
            destination = Vector{T}(undef, 7)
            uniform_allocations(rng, T)
            rand_next!(rng, destination; threaded = false)
            @test uniform_allocations(rng, T) == (0, 0, 0)
            @test _serial_fill_allocations(rng, destination) == 0
        end
    end

    # The codegen check targets the kernel-facing generator. CPU-bound 64-bit
    # Philox uses the host widening multiply, which is 128-bit by design.
    rng = MLD.CUDADevice()(Philox4x64(0x52a))
    @test_throws ArgumentError rand(rng)
    for (function_, signature) in (
        (rand, Tuple{typeof(rng),Type{UInt64}}),
        (rand_next, Tuple{typeof(rng),Type{Float64}}),
        (randat, Tuple{typeof(rng),Type{UInt32},Int}),
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

end

@testset "R26 Philox4x32 Float64 dense fill groups" begin
    # The dense Float64 fill realigns to a block, then emits 128 draws per 53
    # blocks with fixed shifts, and finishes through the bit buffer.
    for bit in (UInt16(0), UInt16(61), UInt16(127)), count in (127, 128, 300)
        rng = _positioned(Philox4x32, 0x52b, UInt64(9), bit)
        expected_rng, expected = _reference_chain(rng, Float64, count)
        destination = Vector{Float64}(undef, count)
        returned, next_rng = rand_next!(rng, destination; threaded = false)
        @test returned === destination
        @test destination == expected
        @test next_rng.position == expected_rng.position
    end

    # Near capacity the aligned group cannot form, and the fill falls back.
    base = Philox4x32(0x52c)
    near_position = IR._Position64(IR._max_block(base) - UInt64(40), UInt16(0))
    near_end = IR._rebuild(base, near_position, base.device)
    expected_rng, expected = _reference_chain(near_end, Float64, 96)
    destination = Vector{Float64}(undef, 96)
    _, next_rng = rand_next!(near_end, destination; threaded = false)
    @test destination == expected
    @test next_rng.position == expected_rng.position
end

@testset "R29 addressed draw ranges" begin
    rng = _positioned(Philox4x64, 0x524, UInt64(5), UInt16(47))
    @test randat(rng, Float64, 3:7) == [randat(rng, Float64, i) for i = 3:7]
    @test randnat(rng, Float32, 2:5) == [randnat(rng, Float32, i) for i = 2:5]
    @test randexpat(rng, Float64, 4:9) == [randexpat(rng, Float64, i) for i = 4:9]
    @test randat(rng, UInt32, 5:4) == UInt32[]
    @test_throws ArgumentError randat(rng, UInt32, 0:3)
end

@testset "Small destination fills allocate nothing" begin
    rng = Philox4x32(0x5f1)
    for T in (Float32, Float64, UInt64)
        buffer = Vector{T}(undef, 64)
        rand_next!(rng, buffer)
        @test @allocated(rand_next!(rng, buffer)) == 0
        normal = Vector{Float64}(undef, 64)
        randn_next!(rng, normal)
        @test @allocated(randn_next!(rng, normal)) == 0
    end
end
