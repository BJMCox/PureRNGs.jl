using InteractiveUtils: code_llvm

const RangeAllocIR = PureRNGs
_small_allocating_range(::Type{T}) where {T<:Signed} = T(-31):T(3):T(41)
_small_allocating_range(::Type{T}) where {T<:Unsigned} = T(2):T(3):T(74)

function _chained_range(rng, range, count)
    values = Vector{eltype(range)}(undef, count)
    cursor = rng
    for index in eachindex(values)
        cursor, values[index] = rand_next(cursor, range)
    end
    return cursor, values
end

@testset "R23-R26 and R55 CPU allocating range draws" begin
    for F in FAMILY_TYPES, T in RANGE_INTS
        range = _small_allocating_range(T)
        rng = _range_positioned(F, 0x65a, UInt64(7), UInt16(61))
        original_position = rng.position
        expected_next, expected = _chained_range(rng, range, 12)

        pure = rand(rng, range, 12)
        next_rng, continued = rand_next(rng, range, 12)
        @test pure == continued == expected
        @test rng.position === original_position
        @test next_rng === expected_next

        matrix = rand(rng, range, 3, 4)
        matrix_next, continued_matrix = rand_next(rng, range, 3, 4)
        @test size(matrix) == (3, 4)
        @test vec(matrix) == expected
        @test continued_matrix == matrix
        @test matrix_next === next_rng

    end
end

@testset "R26 and R55 wide packed range arrays" begin
    ranges = (
        UInt64(0):(UInt64(1)<<32),
        UInt64(0):typemax(UInt64),
        UInt64(7):UInt64(3):UInt64(0xfffffffffffffffd),
    )
    for F in FAMILY_TYPES, range in ranges
        rng = _range_positioned(F, 0x65b, UInt64(9), UInt16(63))
        expected_next, expected = _chained_range(rng, range, 9)
        next_rng, values = rand_next(rng, range, 3, 3)
        @test vec(values) == expected
        @test next_rng === expected_next
        @test rand(rng, range, 3, 3) == values
    end
end

@testset "R26 packed range arrays cross CPU chunks" begin
    for F in (Philox2x32, Philox4x32, Philox4x64),
        range in (UInt16(2):UInt16(17), UInt64(0):(UInt64(1)<<32))

        rng = _range_positioned(F, 0x65b1, UInt64(11), UInt16(61))
        width = RangeAllocIR._range_bits(length(range) % UInt64)
        count = 3 * Int(RangeAllocIR._CPU_FILL_CHUNK_BITS ÷ UInt64(width)) + 3
        expected_next, expected = _chained_range(rng, range, count)
        next_rng, values = rand_next(rng, range, count)
        sync_cpu()
        @test values == expected
        @test next_rng === expected_next
    end


    base = Philox4x64(0x65b2)
    position = RangeAllocIR._Position128(typemax(UInt64), UInt64(7), UInt16(61))
    rng = RangeAllocIR._rebuild(base, position, base.device)
    range = UInt16(2):UInt16(17)
    width = RangeAllocIR._range_bits(length(range) % UInt64)
    chunk_elements = Int(RangeAllocIR._CPU_FILL_CHUNK_BITS ÷ UInt64(width))

    count = 3chunk_elements + 3
    expected_next, expected = _chained_range(rng, range, count)
    next_rng, values = rand_next(rng, range, count)
    sync_cpu()
    @test values == expected
    @test next_rng === expected_next
end

@testset "R26 packed small range arrays" begin
    rng = _range_positioned(Philox4x32, 0x65b3, UInt64(13), UInt16(61))
    range = UInt64(0):(UInt64(1)<<32)
    count = 128
    expected_next, expected = _chained_range(rng, range, count)
    next_rng, values = rand_next(rng, range, count)

    @test values == expected
    @test next_rng === expected_next
end

@testset "R53-R55 allocating range validation and capacity" begin
    nonempty = UInt16(2):UInt16(3):UInt16(20)
    for F in FAMILY_TYPES
        base = F(0x65c)
        terminal_position =
            base.position isa RangeAllocIR._Position64 ?
            RangeAllocIR._terminal64(RangeAllocIR._max_block(base)) :
            RangeAllocIR._terminal128()
        terminal = RangeAllocIR._rebuild(base, terminal_position, base.device)

        empty = rand(terminal, nonempty, 0, 2)
        empty_next, continued_empty = rand_next(terminal, nonempty, 0, 2)
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

    for F in FAMILY_TYPES, range in (UInt8(1):UInt8(7), UInt64(0):(UInt64(1)<<32))
        base = F(0x65d)
        width = _range_reference_width(length(range) % UInt64)
        capacity = _range_capacity(base)
        final_position = _range_position_from_absolute(base, capacity - 2width)
        final = RangeAllocIR._rebuild(base, final_position, base.device)
        expected_next, expected = _chained_range(final, range, 2)
        next_rng, values = rand_next(final, range, 2)
        @test values == expected
        @test next_rng === expected_next
        @test next_rng.position == (
            base.position isa RangeAllocIR._Position64 ?
            RangeAllocIR._terminal64(RangeAllocIR._max_block(base)) :
            RangeAllocIR._terminal128()
        )

        insufficient_position = _range_position_from_absolute(base, capacity - 2width + 1)
        insufficient = RangeAllocIR._rebuild(base, insufficient_position, base.device)
        @test_throws ArgumentError rand(insufficient, range, 2)
        @test_throws ArgumentError rand_next(insufficient, range, 2)
        @test insufficient.position === insufficient_position
    end
end

@testset "R30, R54, and R61 allocating range codegen" begin
    for (F, range) in
        ((Philox2x32, UInt16(2):UInt16(17)), (Philox4x64, UInt64(0):typemax(UInt64)))
        rng = F(0x65f)
        signature = Tuple{typeof(rng),typeof(range),Int}
        typed_ir = sprint(show, code_typed(rand_next, signature; optimize = true))
        llvm_ir = sprint() do io
            code_llvm(
                io,
                rand_next,
                signature;
                raw = false,
                dump_module = false,
                optimize = true,
            )
        end
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        # LLVM may pack UInt64 lanes through i128 casts without wide arithmetic.
        wide_instructions = filter(split(llvm_ir, '\n')) do line
            occursin(r"\bi128\b", line) && !occursin(r"= (?:bitcast|trunc)\b", line)
        end
        @test isempty(wide_instructions)
    end
end
