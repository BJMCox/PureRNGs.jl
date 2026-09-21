_uniform_chain(rng, ::Type{T}, count::Integer) where {T} =
    _reference_chain(rng, cursor -> _reference_uniform(cursor, T), _uniform_width(T), count)

uniform_allocations(rng, ::Type{T}) where {T} = (
    @allocated(rand(rng, T)),
    @allocated(rand_next(rng, T)),
    @allocated(rand_at(rng, T, 3))
)

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
const PACKED_GOLDEN_UNIFORM = (
    (true, 0xf39608ad, 0xf39608ad5487335f, 0x3f739608, 0x3fee72c115aa90e6),
    (false, 0x2029c87b, 0x2029c87b4a20bd10, 0x3e00a720, 0x3fc014e43da5105c),
    (false, 0x2a2d596a, 0x2a2d596a681fd002, 0x3e28b564, 0x3fc516acb5340fe8),
    (false, 0x624cb22e, 0x624cb22e36a263ea, 0x3ec49964, 0x3fd8932c8b8da898),
    (true, 0xd4b0fe71, 0xd4b0fe71e0a7135a, 0x3f54b0fe, 0x3fea961fce3c14e2),
    (false, 0x0d7f8e4d, 0x0d7f8e4de91962b4, 0x3d57f8e0, 0x3faaff1c9bd232c0),
    (false, 0x44d3930c, 0x44d3930c5cae4976, 0x3e89a726, 0x3fd134e4c3172b92),
    (false, 0x6dc19acb, 0x6dc19acb71673eb8, 0x3edb8334, 0x3fdb7066b2dc59ce),
)

@testset "R13 packed uniform golden vectors" begin
    for ((F, key), (bit, word32, word64, float32_bits, float64_bits)) in
        zip(PACKED_GOLDEN_GENERATORS, PACKED_GOLDEN_UNIFORM)
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

@testset "R13, R28, R55, and R63 every family reads the uniform stream" begin
    # Normal, exponential and range draws take their bits from the stream the
    # block above pins, so at one position each raw is a prefix of that word.
    # Pinning the prefix leaves every family one thing of its own to check: the
    # transform it applies, against the reference that owns it.
    for ((F, key), golden) in zip(PACKED_GOLDEN_GENERATORS, PACKED_GOLDEN_UNIFORM)
        word64 = golden[3]
        rng = _packed_golden_rng(F, key)
        block = _reference_position_block(rng.position)
        bit = rng.position.bit
        raw32 = IR._extract_bits_unchecked(rng, block, bit, Val(23))
        raw64 = IR._extract_bits_unchecked(rng, block, bit, Val(52))
        @test raw32 === word64 >> 41
        @test raw64 === word64 >> 12
        @test IR._extract_bits_unchecked(rng, block, bit, Val(64)) === word64
        @test last(IR._extract_bits128_unchecked(rng, block, bit)) === word64

        for (T, raw) in ((Float32, raw32), (Float64, raw64))
            _, value = _reference_exponential_lattice(T, raw)
            @test randexp(rng, T) === IR._exponential_transform(IR._CPU_BACKEND, T, value)
            @test randexp(MLD.CUDADevice()(rng), T) === -Base.log(value)
        end
    end
end

@testset "R25 and R53 packed primitive widths" begin
    for (T, width) in zip(PURE_UNIFORM_TYPES, (1, 32, 32, 64, 64, 24, 53))
        @test IR._draw_bits(T) === UInt16(width)
    end

    for F in GENERATOR_TYPES, T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x521, UInt64(9), UInt16(61))
        @test rand(rng, T) === _reference_uniform(rng, T)
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

        @test reinterpret(U, rand_at(rng, S, 7)) === rand_at(rng, U, 7)

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

@testset "R29 packed addressed draws at stream capacity" begin
    for F in (Philox2x64, Threefry2x64)
        rng = F(0x523)
        final_index = big(1) << 65
        last = IR._rebuild(
            rng,
            IR._Position64(typemax(UInt64), IR._block_bits(rng) - UInt16(64)),
            rng.device,
        )
        @test rand_at(rng, UInt64, final_index) === rand(last, UInt64)
        @test_throws StreamExhausted rand_at(rng, UInt64, final_index + 1)
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
        @test rand_at(rng, UInt64, final_index) === rand(last, UInt64)
        @test_throws StreamExhausted rand_at(rng, UInt64, final_index + 1)

        near_end = IR._rebuild(
            rng,
            IR._Position128(typemax(UInt64), typemax(UInt64), UInt16(0)),
            rng.device,
        )
        @test rand_at(near_end, UInt64, UInt64(4)) === rand(
            IR._rebuild(
                near_end,
                IR._Position128(typemax(UInt64), typemax(UInt64), UInt16(192)),
                rng.device,
            ),
            UInt64,
        )
        @test_throws StreamExhausted rand_at(near_end, UInt64, UInt64(5))
    end
end

@testset "R29 addressed end-span preflight" begin
    base = Philox4x32(0x5231)
    last = IR._rebuild(
        base,
        IR._Position64(IR._max_block(base), IR._block_bits(base) - UInt16(32)),
        base.device,
    )
    @test rand_at(last, UInt32, UInt64(1)) === rand(last, UInt32)
    @test_throws StreamExhausted rand_at(last, UInt32, UInt64(2))
end

@testset "R26 packed BitArray fills" begin
    # A BitArray destination packs 64 draws to a word, so it is the one CPU
    # destination whose element write is not a store.
    rng = _positioned(Philox4x32, 0x525, UInt64(4), UInt16(63))
    expected_rng, expected = _uniform_chain(rng, Bool, 67)
    ordinary = BitArray(undef, 67)
    explicit = similar(ordinary)
    serial = similar(ordinary)
    ordinary_result, ordinary_next = rand_next!(rng, ordinary)
    explicit_result, explicit_next = rand_next!(rng, explicit; threaded = true)
    serial_result, serial_next = rand_next!(rng, serial; threaded = false)
    @test ordinary_result === ordinary
    @test explicit_result === explicit
    @test serial_result === serial
    @test ordinary == explicit == serial == expected
    @test ordinary_next.position ==
          explicit_next.position ==
          serial_next.position ==
          expected_rng.position
end

@testset "R26 parallel packed fills cross CPU chunks" begin
    for F in (Philox2x32, Philox4x32, Philox4x64), T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x5251, UInt64(4), UInt16(61))
        chunk_elements = IR._fill_chunk_elements(Val(:uniform), T)
        count = 4chunk_elements + 3
        serial = Vector{T}(undef, count)
        threaded = similar(serial)

        serial_result, serial_next = rand_next!(rng, serial; threaded = false)
        threaded_result, threaded_next = rand_next!(rng, threaded; threaded = true)

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
    for T in PURE_UNIFORM_TYPES, bit in (UInt16(0), UInt16(127)), delta in (-1, 0, 1)

        x4_group =
            T === Bool ? 512 :
            T <: Union{Int32,UInt32} ? 16 :
            T <: Union{Int64,UInt64} ? 8 : T === Float32 ? 64 : 1
        group = max(
            IR._fill_store_elements(T),
            cld(4 * 128 - Int(bit), _uniform_width(T)),
            x4_group,
        )
        count = group + delta
        count < 0 && continue
        rng = _positioned(Philox4x32, 0x5254, UInt64(9), bit)
        expected_rng, expected = _uniform_chain(rng, T, count)
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
    expected_rng, expected = _uniform_chain(rng, UInt32, 37)
    @test probe.data == expected
    @test next_rng.position == expected_rng.position
end

@testset "R23 and R30 packed uniform fixed-work and codegen" begin
    for F in GENERATOR_TYPES
        rng = F(0x529)
        default_value, default_next = rand_next(rng)
        typed_value, typed_next = rand_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value

        for T in PURE_UNIFORM_TYPES
            uniform_allocations(rng, T)
            @test uniform_allocations(rng, T) == (0, 0, 0)
        end
    end

    # The codegen check targets the kernel-facing generator. CPU-bound 64-bit
    # Philox uses the host widening multiply, which is 128-bit by design.
    rng = MLD.CUDADevice()(Philox4x64(0x52a))
    @test_throws ArgumentError rand(rng)
    for (function_, signature) in (
        (rand, Tuple{typeof(rng),Type{UInt64}}),
        (rand_next, Tuple{typeof(rng),Type{Float64}}),
        (rand_at, Tuple{typeof(rng),Type{UInt32},Int}),
    )
        typed_ir, llvm_ir = _codegen_ir(function_, signature)
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        @test !occursin(r"\bi128\b", llvm_ir)
    end
end

@testset "R26 Philox4x32 Float64 dense fill near capacity" begin
    # The dense Float64 fill realigns to a block, then emits 128 draws per 53
    # blocks. Near capacity the aligned group cannot form, and the fill falls
    # back to the bit buffer.
    base = Philox4x32(0x52c)
    near_position = IR._Position64(IR._max_block(base) - UInt64(40), UInt16(0))
    near_end = IR._rebuild(base, near_position, base.device)
    expected_rng, expected = _uniform_chain(near_end, Float64, 96)
    destination = Vector{Float64}(undef, 96)
    _, next_rng = rand_next!(near_end, destination; threaded = false)
    @test destination == expected
    @test next_rng.position == expected_rng.position
end

@testset "R29 addressed draw ranges" begin
    rng = _positioned(Philox4x64, 0x524, UInt64(5), UInt16(47))
    @test rand_at(rng, Float64, 3:7) == [rand_at(rng, Float64, i) for i = 3:7]
    @test randn_at(rng, Float32, 2:5) == [randn_at(rng, Float32, i) for i = 2:5]
    @test randexp_at(rng, Float64, 4:9) == [randexp_at(rng, Float64, i) for i = 4:9]
    @test rand_at(rng, UInt32, 5:4) == UInt32[]
    @test_throws ArgumentError rand_at(rng, UInt32, 0:3)
end

@testset "R8 fills keep one stream across index styles" begin
    linear_destinations(::Type{T}, n) where {T} = (
        Vector{T}(undef, n),
        view(Vector{T}(undef, n + 3), 1:n),
        view(Vector{T}(undef, 2n), 1:2:2n),
        reshape(view(Vector{T}(undef, n + 5), 1:n), n, 1),
    )
    # `Transpose` and a strided second index both index Cartesian, so they take
    # the per-element fallback.
    cartesian_destinations(::Type{T}, n) where {T} =
        (transpose(Matrix{T}(undef, 1, n)), view(Matrix{T}(undef, n, 2), :, 1:2:2))

    function check_stream(run_fill, ::Type{T}, n, threaded) where {T}
        reference, reference_rng = run_fill(Vector{T}(undef, n), false)
        for destination in linear_destinations(T, n)
            @test IndexStyle(destination) === IndexLinear()
        end
        for destination in cartesian_destinations(T, n)
            @test IndexStyle(destination) === IndexCartesian()
        end
        for destination in (linear_destinations(T, n)..., cartesian_destinations(T, n)...)
            filled, next_rng = run_fill(destination, threaded)
            @test vec(collect(filled)) == reference
            @test next_rng.position == reference_rng.position
        end
        return nothing
    end

    for F in (Philox4x32, Threefry4x64, ChaCha),
        T in (Float64, Float32, UInt64, Bool),
        n in (1, 17, 4096, 100_003),
        threaded in (false, true)

        rng = _positioned(F, 0x531, UInt64(3), UInt16(5))
        check_stream(T, n, threaded) do destination, run_threaded
            rand_next!(rng, destination; threaded = run_threaded)
        end
    end

    rng = _positioned(Philox4x32, 0x532, UInt64(3), UInt16(5))
    for threaded in (false, true)
        check_stream(Float64, 4096, threaded) do destination, run_threaded
            randn_next!(rng, destination; threaded = run_threaded)
        end
        check_stream(Float32, 4096, threaded) do destination, run_threaded
            randexp_next!(rng, destination; threaded = run_threaded)
        end
        check_stream(Int32, 4096, threaded) do destination, run_threaded
            rand_next!(rng, destination, Int32(-5):Int32(9); threaded = run_threaded)
        end
    end
end

@testset "Dynamic chunk scheduler covers every ordinal once" begin
    seen = zeros(Int, 100_003)
    IR._run_chunks(100_003, 4096) do first, last
        for i = first:last
            seen[i] += 1
        end
    end
    @test all(==(1), seen)
    rng = Philox4x32(0x5c4)
    a = Vector{Float64}(undef, 2^20)
    b = similar(a)
    rand_next!(rng, a)
    rand_next!(rng, b; threaded = false)
    @test a == b
end

@testset "Threaded CPU fills keep the serial stream" begin
    population = collect(1:10)
    weights = collect(1.0:10.0)
    weighted_chunk = Int(IR._CPU_FILL_CHUNK_BITS ÷ UInt64(IR._WEIGHT_BITS))
    weighted_chunk -= weighted_chunk % IR._WEIGHTED_LOOKUP_LANES
    fills = (
        (
            IR._fill_chunk_elements(Val(:uniform), Float64),
            (rng, n, threaded) ->
                rand_next!(rng, Vector{Float64}(undef, n); threaded = threaded),
        ),
        (
            IR._fill_chunk_elements(IR._NormalCodec(IR._CPUBackend()), Float64),
            (rng, n, threaded) ->
                randn_next!(rng, Vector{Float64}(undef, n); threaded = threaded),
        ),
        (
            Int(IR._CPU_FILL_CHUNK_BITS ÷ UInt64(IR._range_bits(IR._range_span(1:10)))),
            (rng, n, threaded) ->
                rand_next!(rng, Vector{Int}(undef, n), 1:10; threaded = threaded),
        ),
        (
            Int(IR._CPU_FILL_CHUNK_BITS ÷ UInt64(IR._range_bits(UInt64(10)))),
            (rng, n, threaded) -> randsample_next!(
                rng,
                population,
                Vector{Int}(undef, n);
                threaded = threaded,
            ),
        ),
        (
            weighted_chunk,
            (rng, n, threaded) -> randsample_next!(
                rng,
                population,
                weights,
                Vector{Int}(undef, n);
                threaded = threaded,
            ),
        ),
    )
    rng = Philox4x32(0x5c5)
    # The second count leaves a short final chunk.
    for (chunk, fill) in fills, count in (8 * chunk, 4 * chunk + 1)
        threaded, threaded_rng = fill(rng, count, true)
        serial, serial_rng = fill(rng, count, false)
        @test threaded == serial
        @test threaded_rng.position == serial_rng.position
    end
end
