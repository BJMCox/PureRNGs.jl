using InteractiveUtils: code_llvm

const RangeAllocIR = PureRNGs
const RANGE_ALLOCATING_TYPES = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)

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
    for F in RANGE_FAMILIES, T in RANGE_ALLOCATING_TYPES
        range = _small_allocating_range(T)
        rng = _range_positioned(F, 0x65a, UInt64(7), UInt16(61))
        original_position = rng.position
        expected_next, expected = _chained_range(rng, range, 12)

        pure = rand(rng, range, 12)
        next_rng, continued = rand_next(rng, range, 12)
        @test pure isa Vector{T}
        @test continued isa Vector{T}
        @test pure == continued == expected
        @test rng.position === original_position
        @test next_rng === expected_next

        matrix = rand(rng, range, 3, 4)
        matrix_next, continued_matrix = rand_next(rng, range, 3, 4)
        @test matrix isa Matrix{T}
        @test size(matrix) == (3, 4)
        @test vec(matrix) == expected
        @test continued_matrix == matrix
        @test matrix_next === next_rng

        mixed_dims = rand(rng, range, UInt8(2), Int16(3))
        @test size(mixed_dims) == (2, 3)
        @test vec(mixed_dims) == expected[1:6]

        @test @inferred(rand(rng, range, 2, 3)) isa Matrix{T}
        @test @inferred(rand_next(rng, range, 2, 3)) isa Tuple{typeof(rng),Matrix{T}}
    end
end

@testset "R26 and R55 wide packed range arrays" begin
    ranges = (
        UInt64(0):(UInt64(1)<<32),
        UInt64(0):typemax(UInt64),
        UInt64(7):UInt64(3):UInt64(0xfffffffffffffffd),
    )
    for F in RANGE_FAMILIES, range in ranges
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
        count = Int(RangeAllocIR._CPU_FILL_CHUNK_BITS ÷ UInt64(width)) + 3
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
    chunk_lo, chunk_hi = RangeAllocIR._bit_span(UInt64(chunk_elements), width)
    second_position = RangeAllocIR._advance_position_unchecked(rng, chunk_lo, chunk_hi)
    @test second_position == RangeAllocIR._Position128(UInt64(0x1ff), UInt64(8), UInt16(61))

    count = chunk_elements + 3
    expected_next, expected = _chained_range(rng, range, count)
    next_rng, values = rand_next(rng, range, count)
    sync_cpu()
    @test values == expected
    @test next_rng === expected_next
end

@testset "R53-R55 allocating range validation and capacity" begin
    nonempty = UInt16(2):UInt16(3):UInt16(20)
    for F in RANGE_FAMILIES
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
            for draw in (rand, rand_next)
                exception = try
                    draw(base, empty_range, -1)
                    nothing
                catch error
                    error
                end
                @test exception isa ArgumentError
                @test occursin("range", sprint(showerror, exception))
            end
        end

        @test_throws ArgumentError rand(base, nonempty, -1)
        @test_throws ArgumentError rand(base, nonempty, 2, -1)
        @test_throws ArgumentError rand_next(base, nonempty, -1)
        @test_throws ArgumentError rand_next(base, nonempty, 2, -1)
    end

    for F in RANGE_FAMILIES, range in (UInt8(1):UInt8(7), UInt64(0):(UInt64(1)<<32))
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

@testset "R23, R47, and R49 allocating range method surface" begin
    rng = Philox4x32(0x65e)
    for T in RANGE_ALLOCATING_TYPES
        range = _small_allocating_range(T)
        @test which(rand, (typeof(rng), typeof(range), Int)).module === RangeAllocIR
        @test which(rand_next, (typeof(rng), typeof(range), Int)).module === RangeAllocIR
        @test Base.kwarg_decl(which(rand, (typeof(rng), typeof(range), Int))) == Symbol[]
        @test Base.kwarg_decl(which(rand_next, (typeof(rng), typeof(range), Int))) ==
              Symbol[]
        @test_throws MethodError rand(rng, range, 3; threaded = false)
        @test_throws MethodError rand_next(rng, range, 3; threaded = false)
    end

    for range in (false:true, Int128(1):Int128(3), UInt128(1):UInt128(3), 1.0:3.0)
        @test !applicable(rand, rng, range, 3)
        @test !applicable(rand_next, rng, range, 3)
    end

    unknown =
        RangeAllocIR._rebuild(rng, rng.position, RangeAllocIR.MLDataDevices.UnknownDevice())
    range = UInt16(2):UInt16(17)
    @test !applicable(rand, unknown, range, 3)
    @test !applicable(rand_next, unknown, range, 3)
    @test_throws MethodError rand(unknown, range, 3)
    @test_throws MethodError rand_next(unknown, range, 3)
end

@testset "R30, R54, and R61 allocating range inference and IR" begin
    for (F, range) in
        ((Philox2x32, UInt16(2):UInt16(17)), (Philox4x64, UInt64(0):typemax(UInt64)))
        rng = F(0x65f)
        destination = Vector{eltype(range)}(undef, 17)
        span = RangeAllocIR._range_span(range)
        RangeAllocIR._fill_range_cpu_unchecked!(
            rng,
            rng.position,
            destination,
            range,
            span,
            eachindex(destination),
        )
        @test @allocated(
            RangeAllocIR._fill_range_cpu_unchecked!(
                rng,
                rng.position,
                destination,
                range,
                span,
                eachindex(destination),
            )
        ) == 0

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
        @test !occursin(r"\bi128\b", llvm_ir)
    end
end
