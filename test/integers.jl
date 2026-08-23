using Random

const RangeIR = PureRNGs
const RANGE_INTS = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
const RANGE_FAMILIES = (
    Philox2x32((UInt32(0x13579bdf),)),
    Philox4x32((UInt32(0x01234567), UInt32(0x89abcdef))),
    Threefry2x32((UInt32(0x76543210), UInt32(0xfedcba98))),
    Threefry4x32((
        UInt32(0x01234567),
        UInt32(0x89abcdef),
        UInt32(0xfedcba98),
        UInt32(0x76543210),
    )),
)

struct UnionIntegerRange <: AbstractRange{Union{Int8,UInt8}} end

function reference_range_value(range, offset::UInt64)
    return eltype(range)(BigInt(first(range)) + BigInt(step(range)) * BigInt(offset))
end

function reference_range_words(rng, count::Int)
    width = Int(RangeIR._words_per_block(rng))
    lane = Int(rng.position.lane)
    return ntuple(count) do offset
        absolute_lane = lane + offset - 1
        block = rng.position.block + UInt64(absolute_lane ÷ width)
        block_lane = absolute_lane % width
        RangeIR._block(rng, UInt32(3), block)[block_lane+1]
    end
end

function reference_alignment_padding(rng, words::UInt64)
    width = BigInt(RangeIR._words_per_block(rng))
    position = BigInt(rng.position.block) * width + BigInt(rng.position.lane)
    return UInt64(mod(-position, BigInt(words)))
end

function reference_scalar_range(rng, range)
    span = length(range) % UInt64
    count = span != 0 && span <= UInt64(1) << 32 ? UInt64(2) : UInt64(4)
    padding = reference_alignment_padding(rng, count)
    start = RangeIR._reserve(rng, padding)
    words = reference_range_words(start, Int(count))
    high = (UInt64(words[1]) << 32) | UInt64(words[2])
    offset = if span == 0
        high
    elseif span <= UInt64(1) << 32
        UInt64((BigInt(high) * BigInt(span)) >> 64)
    else
        low = (UInt64(words[3]) << 32) | UInt64(words[4])
        UInt64((((BigInt(high) << 64) + BigInt(low)) * BigInt(span)) >> 128)
    end
    return reference_range_value(range, offset)
end

@testset "scalar integer-range draws" begin
    @test RangeIR.FAMILY_RANGE === UInt32(3)

    @testset "exact method surface" begin
        rng = first(RANGE_FAMILIES)
        for T in RANGE_INTS
            range = T(1):T(3)
            @test applicable(rand, rng, range)
            @test applicable(rand_next, rng, range)
            @test which(rand, (typeof(rng), typeof(range))).module === RangeIR
            @test which(rand_next, (typeof(rng), typeof(range))).module === RangeIR
        end
        for range in (
            false:true,
            Int128(1):Int128(3),
            UInt128(1):UInt128(3),
            1.0:3.0,
            UnionIntegerRange(),
        )
            @test !applicable(rand, rng, range)
            @test !applicable(rand_next, rng, range)
            @test_throws MethodError rand(rng, range)
            @test_throws MethodError rand_next(rng, range)
        end
        @test !applicable(rand, rng, Int8(1):Int8(3), 2)
        @test !applicable(rand_next, rng, Int8(1):Int8(3), 2)
    end

    @testset "pinned oracle vectors" begin
        # Philox4x32 and Threefry2x32 values come from the clean pinned oracle:
        # PureRNGsTestbed.jl 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d
        # Philox2x32 and Threefry4x32 use independent BigInt reductions over
        # frozen core-plus-layout words.
        cases = (
            UInt8(3):UInt8(17),
            Int16(-23):Int16(41),
            UInt32(7):(UInt32(7)+(UInt32(1)<<31)),
            UInt64(9):(UInt64(9)+(UInt64(1)<<32)),
            UInt64(0):typemax(UInt64),
            typemin(Int64):typemax(Int64),
        )
        @test RangeIR._block(first(RANGE_FAMILIES), UInt32(3), UInt64(0)) ==
              (UInt32(0x08a579f1), UInt32(0xc15a7dac))
        @test RangeIR._block(last(RANGE_FAMILIES), UInt32(3), UInt64(0)) == (
            UInt32(0x43ed6cc4),
            UInt32(0x94228d89),
            UInt32(0x83a95813),
            UInt32(0xc6438869),
        )
        expected = (
            (
                UInt8(0x03),
                Int16(-21),
                UInt32(0x0452bcff),
                UInt64(0x08a579fa),
                UInt64(0x08a579f1c15a7dac),
                Int64(-8600333834156081748),
            ),
            (
                UInt8(0x08),
                Int16(-1),
                UInt32(0x2b5edba5),
                UInt64(0x56bdb745),
                UInt64(0x56bdb73bca6cd6b0),
                Int64(-2973018711567575376),
            ),
            (
                UInt8(0x0d),
                Int16(23),
                UInt32(0x5bbf2170),
                UInt64(0xb77e42db),
                UInt64(0xb77e42d15c8f29e7),
                Int64(3998706986120063463),
            ),
            (
                UInt8(0x06),
                Int16(-6),
                UInt32(0x21f6b669),
                UInt64(0x43ed6ccd),
                UInt64(0x43ed6cc494228d89),
                Int64(-4328684075278496375),
            ),
        )
        for (rng, values) in zip(RANGE_FAMILIES, expected)
            @test map(range -> rand(rng, range), cases) == values
        end
    end

    @testset "range forms and fixed work" begin
        threshold = UInt64(1) << 32
        ranges = (
            Int8(-7):Int8(9),
            UInt8(2):UInt8(11),
            Int16(21):Int16(-3):Int16(-18),
            UInt16(19):Int16(-4):UInt16(3),
            Int32(-100):Int32(7):Int32(103),
            UInt32(91):Int32(-9):UInt32(1),
            Int64(-1000):Int64(31):Int64(1300),
            UInt64(1000):Int64(-37):UInt64(1),
            typemin(Int8):typemax(Int8),
            UInt8(0):typemax(UInt8),
            typemin(Int16):typemax(Int16),
            UInt16(0):typemax(UInt16),
            typemin(Int32):typemax(Int32),
            UInt32(0):typemax(UInt32),
            UInt64(9):(UInt64(9)+threshold-UInt64(1)),
            UInt64(9):(UInt64(9)+threshold),
            typemin(Int64):typemax(Int64),
            UInt64(0):typemax(UInt64),
        )
        for base in RANGE_FAMILIES
            width = Int(RangeIR._words_per_block(base))
            for lane = 0:(width-1), range in ranges
                rng = RangeIR._rebuild(
                    base,
                    RangeIR._Position64(UInt64(5), UInt8(lane)),
                    base.device,
                )
                words = RangeIR._range_words(length(range) % UInt64)
                padding = reference_alignment_padding(rng, words)
                expected = reference_scalar_range(rng, range)
                original = rng.position
                @test rand(rng, range) === expected
                @test rng.position === original
                next, value = rand_next(rng, range)
                @test value === expected
                @test next.position === RangeIR._reserve(rng, padding + words).position
            end
        end
    end

    @testset "singleton and exact result type" begin
        for rng in RANGE_FAMILIES, T in RANGE_INTS
            value = T(17)
            range = value:value
            @test rand(rng, range) === value
            next, drawn = rand_next(rng, range)
            @test drawn === value
            @test next.position === RangeIR._reserve(rng, UInt64(2)).position
            @test @inferred(rand(rng, range)) isa T
            @test @inferred(rand_next(rng, range)) isa Tuple{typeof(rng),T}
        end
    end

    @testset "terminal capacity" begin
        small = UInt8(1):UInt8(7)
        wide = UInt64(0):(UInt64(1)<<32)
        for base in RANGE_FAMILIES
            maximum = RangeIR._max_block(base)
            width = Int(RangeIR._words_per_block(base))

            small_start = RangeIR._Position64(maximum, UInt8(width - 2))
            small_rng = RangeIR._rebuild(base, small_start, base.device)
            small_next, small_value = rand_next(small_rng, small)
            @test small_value === reference_scalar_range(small_rng, small)
            @test RangeIR._is_exhausted(small_next.position)

            wide_block = maximum - UInt64(width == 2)
            wide_rng = RangeIR._rebuild(
                base,
                RangeIR._Position64(wide_block, UInt8(0)),
                base.device,
            )
            wide_next, wide_value = rand_next(wide_rng, wide)
            @test wide_value === reference_scalar_range(wide_rng, wide)
            @test RangeIR._is_exhausted(wide_next.position)

            exhausted = RangeIR._rebuild(base, RangeIR._terminal64(maximum), base.device)
            for range in (small, wide)
                @test_throws ArgumentError rand(exhausted, range)
                @test_throws ArgumentError rand_next(exhausted, range)
            end

            insufficient_small = RangeIR._rebuild(
                base,
                RangeIR._Position64(maximum, UInt8(width - 1)),
                base.device,
            )
            @test_throws ArgumentError rand(insufficient_small, small)
            @test_throws ArgumentError rand_next(insufficient_small, small)
        end
    end

    @testset "errors and machine execution" begin
        rng = first(RANGE_FAMILIES)
        for range in (Int8(2):Int8(1), UInt64(1):UInt64(0))
            @test_throws ArgumentError rand(rng, range)
            @test_throws ArgumentError rand_next(rng, range)
        end

        ranges = (
            Int8(-2):Int8(3),
            UInt16(9):Int16(-2):UInt16(1),
            UInt64(0):(UInt64(1)<<32),
            UInt64(0):typemax(UInt64),
        )
        for rng in RANGE_FAMILIES, range in ranges
            rand(rng, range)
            rand_next(rng, range)
            @test @allocated(rand(rng, range)) == 0
            @test @allocated(rand_next(rng, range)) == 0
        end
    end
end

@testset "fixed-work integer-range arithmetic" begin
    @testset "pinned reductions" begin
        random = Xoshiro(0x7e57bed)
        for _ = 1:10_000
            word = rand(random, UInt64)
            span = rand(random, UInt64(1):(UInt64(1)<<32))
            expected = UInt64((BigInt(word) * BigInt(span)) >> 64)
            @test RangeIR._mulhi32limbs(word, span) == expected

            high = rand(random, UInt64)
            low = rand(random, UInt64)
            wide_span = rand(random, UInt64)
            wide_span <= UInt64(1) << 32 && (wide_span += (UInt64(1) << 32) + UInt64(1))
            expected_wide =
                UInt64((((BigInt(high) << 64) + BigInt(low)) * BigInt(wide_span)) >> 128)
            @test RangeIR._mulhi128_by64(high, low, wide_span) == expected_wide
        end

        @test RangeIR._mulhi32limbs(typemax(UInt64), UInt64(1) << 32) == UInt64(0xffffffff)
        @test RangeIR._mulhi128_by64(typemax(UInt64), typemax(UInt64), typemax(UInt64)) ==
              typemax(UInt64) - UInt64(1)
    end

    @testset "width selection" begin
        threshold = UInt64(1) << 32
        @test RangeIR._range_words(UInt64(1)) == UInt64(2)
        @test RangeIR._range_words(threshold) == UInt64(2)
        @test RangeIR._range_words(threshold + UInt64(1)) == UInt64(4)
        @test RangeIR._range_words(UInt64(0)) == UInt64(4)
    end

    @testset "range encoding and mapping" begin
        for T in RANGE_INTS
            examples = (
                T(3):T(3),
                T(1):T(2):T(7),
                T(7):signed(T)(-2):T(1),
                typemin(T):typemax(T),
                typemax(T):signed(T)(-1):typemin(T),
            )
            for range in examples
                @test eltype(range) === T
                base, stride, span = @inferred RangeIR._range_parameters(range)
                @test span == UInt64(length(range))
                @test RangeIR._range_words(span) ==
                      (span != 0 && span <= UInt64(1) << 32 ? UInt64(2) : UInt64(4))

                offsets =
                    span == 0 ? (UInt64(0), UInt64(1), typemax(UInt64)) :
                    (UInt64(0), UInt64(span - UInt64(1)) >> 1, span - UInt64(1))
                for offset in offsets
                    @test RangeIR._range_value(T, base, stride, offset) ==
                          reference_range_value(range, offset)
                end
            end
        end

        threshold = UInt64(1) << 32
        for range in (
            UInt64(9):(UInt64(9)+threshold-UInt64(1)),
            UInt64(9):(UInt64(9)+threshold),
            (Int64(-2)^31):(Int64(2)^31-Int64(1)),
            (Int64(-2)^31):(Int64(2)^31),
        )
            _, _, span = RangeIR._range_parameters(range)
            @test span == UInt64(length(range))
            @test RangeIR._range_words(span) == (span <= threshold ? UInt64(2) : UInt64(4))
        end

        @test_throws ArgumentError RangeIR._range_parameters(Int64(2):Int64(1))
        @test_throws ArgumentError RangeIR._range_parameters(UInt64(1):UInt64(0))
        @test !applicable(RangeIR._range_parameters, false:true)
        @test !applicable(RangeIR._range_parameters, 1.0:2.0)
    end

    @testset "machine execution" begin
        range = Int64(17):Int64(-3):Int64(-19)
        base, stride, span = RangeIR._range_parameters(range)
        @test @inferred(RangeIR._mulhi32limbs(UInt64(7), UInt64(13))) isa UInt64
        @test @inferred(
            RangeIR._mulhi128_by64(UInt64(7), UInt64(11), (UInt64(1) << 40) + UInt64(3)),
        ) isa UInt64
        @test @inferred(RangeIR._range_parameters(range)) isa NTuple{3,UInt64}
        @test @inferred(RangeIR._range_value(Int64, base, stride, span - UInt64(1)),) isa
              Int64

        RangeIR._mulhi32limbs(UInt64(7), UInt64(13))
        RangeIR._mulhi128_by64(UInt64(7), UInt64(11), (UInt64(1) << 40) + UInt64(3))
        RangeIR._range_parameters(range)
        RangeIR._range_value(Int64, base, stride, span - UInt64(1))
        @test @allocated(RangeIR._mulhi32limbs(UInt64(7), UInt64(13))) == 0
        @test @allocated(
            RangeIR._mulhi128_by64(UInt64(7), UInt64(11), (UInt64(1) << 40) + UInt64(3)),
        ) == 0
        @test @allocated(RangeIR._range_parameters(range)) == 0
        @test @allocated(RangeIR._range_value(Int64, base, stride, span - UInt64(1)),) == 0
    end
end
