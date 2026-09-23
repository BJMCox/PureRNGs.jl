_small_allocating_range(::Type{T}) where {T<:Signed} = T(-31):T(3):T(41)
_small_allocating_range(::Type{T}) where {T<:Unsigned} = T(2):T(3):T(74)

_chained_range(rng, range, count) =
    _chained_draws(cursor -> rand_next(cursor, range), rng, count)

@testset "CPU allocating range draws" begin
    for F in GENERATOR_TYPES, T in RANGE_INTS
        range = _small_allocating_range(T)
        rng = _positioned(F, 0x65a, UInt64(7), UInt16(61))
        original_position = rng.position
        expected_next, expected = _chained_range(rng, range, 12)

        pure = rand(rng, range, 12)
        continued, next_rng = rand_next(rng, range, 12)
        @test pure == continued == expected
        @test rng.position === original_position
        @test next_rng === expected_next

        matrix = rand(rng, range, 3, 4)
        continued_matrix, matrix_next = rand_next(rng, range, 3, 4)
        @test size(matrix) == (3, 4)
        @test vec(matrix) == expected
        @test continued_matrix == matrix
        @test matrix_next === next_rng

    end
end

@testset "wide packed range arrays" begin
    ranges = (
        UInt64(0):(UInt64(1)<<32),
        UInt64(0):typemax(UInt64),
        UInt64(7):UInt64(3):UInt64(0xfffffffffffffffd),
    )
    for F in GENERATOR_TYPES, range in ranges
        rng = _positioned(F, 0x65b, UInt64(9), UInt16(63))
        expected_next, expected = _chained_range(rng, range, 9)
        values, next_rng = rand_next(rng, range, 3, 3)
        @test vec(values) == expected
        @test next_rng === expected_next
        @test rand(rng, range, 3, 3) == values
    end
end

@testset "packed range arrays cross CPU chunks" begin
    for F in (Philox2x32, Philox4x32, Philox4x64),
        range in (UInt16(2):UInt16(17), UInt64(0):(UInt64(1)<<32))

        rng = _positioned(F, 0x65b1, UInt64(11), UInt16(61))
        width = IR._range_bits(length(range) % UInt64)
        count = 3 * Int(IR._CPU_FILL_CHUNK_BITS ÷ UInt64(width)) + 3
        expected_next, expected = _chained_range(rng, range, count)
        values, next_rng = rand_next(rng, range, count)
        @test values == expected
        @test next_rng === expected_next
    end


    base = Philox4x64(0x65b2)
    position = IR._Position128(typemax(UInt64), UInt64(7), UInt16(61))
    rng = IR._rebuild(base, position, base.device)
    range = UInt16(2):UInt16(17)
    width = IR._range_bits(length(range) % UInt64)
    chunk_elements = Int(IR._CPU_FILL_CHUNK_BITS ÷ UInt64(width))

    count = 3chunk_elements + 3
    expected_next, expected = _chained_range(rng, range, count)
    values, next_rng = rand_next(rng, range, count)
    @test values == expected
    @test next_rng === expected_next
end

@testset "allocating range validation" begin
    nonempty = UInt16(2):UInt16(3):UInt16(20)
    for F in GENERATOR_TYPES
        base = F(0x65c)
        terminal_position =
            base.position isa IR._Position64 ? IR._terminal64(IR._max_block(base)) :
            IR._terminal128()
        terminal = IR._rebuild(base, terminal_position, base.device)

        empty = rand(terminal, nonempty, 0, 2)
        continued_empty, empty_next = rand_next(terminal, nonempty, 0, 2)
        @test empty isa Matrix{UInt16}
        @test size(empty) == (0, 2)
        @test continued_empty == empty
        @test empty_next === terminal

        for empty_range in (UInt16(2):UInt16(1), UInt64(1):UInt64(0))
            @test_throws ArgumentError rand(terminal, empty_range, 0)
            @test_throws ArgumentError rand_next(terminal, empty_range, 0)
        end

        @test_throws ArgumentError rand(base, nonempty, -1)
        @test_throws ArgumentError rand(base, nonempty, 2, -1)
        @test_throws ArgumentError rand_next(base, nonempty, -1)
        @test_throws ArgumentError rand_next(base, nonempty, 2, -1)
    end

end

@testset "allocating range codegen" begin
    for (F, range) in
        ((Philox2x32, UInt16(2):UInt16(17)), (Philox4x64, UInt64(0):typemax(UInt64)))
        # The kernel-facing generator: the CPU-bound 64-bit Philox core uses a
        # 128-bit widening multiply by design.
        rng = MLD.CUDADevice()(F(0x65f))
        typed_ir, llvm_ir = _codegen_ir(rand_next, Tuple{typeof(rng),typeof(range),Int})
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        # LLVM may pack UInt64 lanes through i128 casts without wide arithmetic.
        wide_instructions = filter(split(llvm_ir, '\n')) do line
            occursin(r"\bi128\b", line) && !occursin(r"= (?:bitcast|trunc)\b", line)
        end
        @test isempty(wide_instructions)
    end
end
