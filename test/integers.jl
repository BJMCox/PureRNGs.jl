integer_allocations(rng, range) =
    (@allocated(rand(rng, range)), @allocated(rand_next(rng, range)))

addressed_range_allocations(rng, range) = @allocated(rand_at(rng, range, 3))

@testset "R55 packed integer ranges" begin
    threshold = UInt64(1) << 32
    @test IR._range_bits(UInt64(1)) === UInt16(64)
    @test IR._range_bits(threshold) === UInt16(64)
    @test IR._range_bits(threshold + UInt64(1)) === UInt16(128)
    @test IR._range_bits(UInt64(0)) === UInt16(128)

    ranges = (
        Int8(-7):Int8(9),
        UInt16(19):Int16(-4):UInt16(3),
        Int32(-100):Int32(7):Int32(103),
        UInt64(9):(UInt64(9)+threshold-UInt64(1)),
        UInt64(9):(UInt64(9)+threshold),
        typemin(Int64):typemax(Int64),
        UInt64(0):typemax(UInt64),
        typemax(Int64):Int64(-1):typemin(Int64),
    )

    for F in GENERATOR_TYPES
        block_bits = Int(IR._block_bits(F(0)))
        bit = UInt16(block_bits - 1)
        for range in ranges
            rng = _positioned(F, 0x551, UInt64(9), bit)
            expected = _range_reference_draw(rng, range)
            @test rand(rng, range) === expected
            # The draw is pure: a second draw from the same generator matches
            # one from a pristine generator at the same position.
            pristine = _positioned(F, 0x551, UInt64(9), bit)
            @test rand(rng, range) === rand(pristine, range)

            value, next_rng = rand_next(rng, range)
            width = _range_reference_width(length(range) % UInt64)
            @test value === expected
            @test next_rng.position == _reference_position(rng, width)
        end
    end
end

@testset "R8 and R53 mixed primitive and range positions" begin
    small = UInt16(3):UInt16(41)
    wide = UInt64(0):(UInt64(1)<<32)
    for F in GENERATOR_TYPES
        rng = _positioned(F, 0x552, UInt64(11), UInt16(63))
        first_expected = _reference_uniform(rng, UInt32)
        first_value, rng = rand_next(rng, UInt32)
        @test first_value === first_expected

        small_expected = _range_reference_draw(rng, small)
        small_position = _reference_position(rng, 64)
        small_value, rng = rand_next(rng, small)
        @test small_value === small_expected
        @test rng.position == small_position

        wide_expected = _range_reference_draw(rng, wide)
        wide_position = _reference_position(rng, 128)
        wide_value, rng = rand_next(rng, wide)
        @test wide_value === wide_expected
        @test rng.position == wide_position

        final_expected = _reference_uniform(rng, Bool)
        final_value, rng = rand_next(rng, Bool)
        @test final_value === final_expected
        @test rng.position ==
              _reference_position(IR._rebuild(rng, wide_position, rng.device), 1)
    end
end

@testset "R53 and R54 range capacity and validation" begin
    small = UInt8(1):UInt8(7)
    wide = UInt64(0):(UInt64(1)<<32)
    for F in GENERATOR_TYPES, (range, width) in ((small, 64), (wide, 128))
        base = F(0x553)
        capacity = _stream_capacity(base)
        last_position = _position_from_absolute(base, capacity - width)
        last = IR._rebuild(base, last_position, base.device)
        expected = _range_reference_draw(last, range)
        value, terminal = rand_next(last, range)
        @test value === expected
        @test terminal.position == (
            base.position isa IR._Position64 ? IR._terminal64(IR._max_block(base)) :
            IR._terminal128()
        )

        insufficient_position = _position_from_absolute(base, capacity - width + 1)
        insufficient = IR._rebuild(base, insufficient_position, base.device)
        @test_throws StreamExhausted rand(insufficient, range)
        @test insufficient.position === insufficient_position
        @test_throws StreamExhausted rand_next(insufficient, range)
        @test insufficient.position === insufficient_position

        exhausted = IR._rebuild(base, terminal.position, base.device)
        @test_throws StreamExhausted rand(exhausted, range)
        @test_throws StreamExhausted rand_next(exhausted, range)
    end

    for F in GENERATOR_TYPES
        rng = F(0x554)
        for range in (Int8(2):Int8(1), UInt64(1):UInt64(0))
            @test_throws ArgumentError rand(rng, range)
            @test_throws ArgumentError rand_next(rng, range)
        end
    end
end

@testset "R23 and R55 range method and mapping surface" begin
    rng = Philox4x32(0x555)
    rounded = LinRange{Int64}(Int64(1)<<53, (Int64(1)<<53)+Int64(4), 5)
    @test rand(rng, rounded) === _range_reference_draw(rng, rounded)
    long = LinRange{Int64}(0, 0, typemax(UInt64))
    @test rand(rng, long) === Int64(0)
end

@testset "R29 and R55 addressed range draws" begin
    # A far index still lands where the chain of same-span draws reaches.
    for F in (Philox4x32, Threefry4x64, ChaCha),
        range in (-3:3, Int32(10):Int32(-2):Int32(-10), UInt64(1):(UInt64(2)^40))

        rng = _positioned(F, 0x558, UInt64(3), UInt16(17))
        cursor = first(_chained_draws(c -> rand_next(c, range), rng, 999))
        @test rand_at(rng, range, 1000) === first(rand_next(cursor, range))
    end

    rng = Philox4x32(0x559)
    for range in (1:6, 1:(2^40))
        # [R29] pins the addressed draw to the fill element at the same index.
        @test [rand_at(rng, range, i) for i = 1:8] == rand(rng, range, 8)
        @inferred rand_at(rng, range, 3)
        addressed_range_allocations(rng, range)
        @test addressed_range_allocations(rng, range) == 0
    end

    @test_throws ArgumentError rand_at(rng, Int8(2):Int8(1), 1)
end

@testset "R30 and R55 fixed-work range codegen" begin
    for F in GENERATOR_TYPES,
        range in (
            Int8(-2):Int8(3),
            UInt16(9):Int16(-2):UInt16(1),
            UInt64(0):(UInt64(1)<<32),
            UInt64(0):typemax(UInt64),
        )

        rng = F(0x556)
        integer_allocations(rng, range)
        @test integer_allocations(rng, range) == (0, 0)
    end

    rng = MLD.CUDADevice()(Philox4x64(0x557))
    for (function_, signature) in (
        (rand, Tuple{typeof(rng),typeof(UInt16(2):UInt16(17))}),
        (rand_next, Tuple{typeof(rng),typeof(UInt64(0):typemax(UInt64))}),
    )
        typed_ir, llvm_ir = _codegen_ir(function_, signature)
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        @test !occursin(r"\bi128\b", llvm_ir)
    end
end
