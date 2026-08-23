using Random

const RangeIR = PureRNGs
const RANGE_INTS = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)

function reference_range_value(range, offset::UInt64)
    return eltype(range)(BigInt(first(range)) + BigInt(step(range)) * BigInt(offset))
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
