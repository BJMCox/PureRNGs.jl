using InteractiveUtils: code_llvm
using Random: rand

integer_allocations(rng, range) =
    (@allocated(rand(rng, range)), @allocated(rand_next(rng, range)))

const RangeIR = PureRNGs
const RANGE_INTS = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)

_range_reference_block(position::RangeIR._Position64) = position.block
_range_reference_block(position::RangeIR._Position128) = (position.lo, position.hi)

_range_reference_width(span::UInt64) =
    span != UInt64(0) && span <= UInt64(1) << 32 ? 64 : 128

function _range_reference_offset(rng, span::UInt64)
    block = _range_reference_block(rng.position)
    bit = rng.position.bit
    if _range_reference_width(span) == 64
        candidate = _reference_extract(rng, block, bit, 64)
        return UInt64((BigInt(candidate) * BigInt(span)) >> 64)
    end
    lo, hi = _reference_extract128(rng, block, bit)
    mathematical_span = iszero(span) ? big(1) << 64 : BigInt(span)
    candidate = (BigInt(hi) << 64) + BigInt(lo)
    return UInt64((candidate * mathematical_span) >> 128)
end

function _range_reference_value(range::OrdinalRange{T}, offset::UInt64) where {T}
    return T(BigInt(first(range)) + BigInt(step(range)) * BigInt(offset))
end

_range_reference_value(range, offset::UInt64) = range[Int(offset)+1]

_range_reference_draw(rng, range) =
    _range_reference_value(range, _range_reference_offset(rng, length(range) % UInt64))

function _range_reference_position(rng, additional_bits::Integer)
    block_bits = BigInt(RangeIR._block_bits(rng))
    position = rng.position
    block =
        position isa RangeIR._Position64 ? BigInt(position.block) :
        (BigInt(position.hi) << 64) + BigInt(position.lo)
    block, bit =
        divrem(block * block_bits + BigInt(position.bit) + additional_bits, block_bits)
    if position isa RangeIR._Position64
        return RangeIR._Position64(UInt64(block), UInt16(bit))
    end
    return RangeIR._Position128(
        UInt64(block & typemax(UInt64)),
        UInt64(block >> 64),
        UInt16(bit),
    )
end

function _range_positioned(F, seed, block::UInt64, bit::UInt16)
    base = F(seed)
    position =
        base.position isa RangeIR._Position64 ? RangeIR._Position64(block, bit) :
        RangeIR._Position128(block, UInt64(7), bit)
    return RangeIR._rebuild(base, position, base.device)
end

function _range_position_from_absolute(rng, absolute::BigInt)
    block, bit = divrem(absolute, BigInt(RangeIR._block_bits(rng)))
    if rng.position isa RangeIR._Position64
        return RangeIR._Position64(UInt64(block), UInt16(bit))
    end
    return RangeIR._Position128(
        UInt64(block & typemax(UInt64)),
        UInt64(block >> 64),
        UInt16(bit),
    )
end

function _range_capacity(rng)
    blocks =
        rng.position isa RangeIR._Position64 ? BigInt(RangeIR._max_block(rng)) + 1 :
        big(1) << 128
    return blocks * BigInt(RangeIR._block_bits(rng))
end

@testset "R13 and R55 packed range golden vectors" begin
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
    range64 = Int32(-1000):Int32(7):Int32(1000)
    range128 = UInt64(7):UInt64(3):UInt64(0xfffffffffffffffd)
    @test length(range128) % UInt64 === UInt64(6148914691236517203)
    # Range draws read the uniform stream, so each 64-bit candidate is the
    # uniform golden block at this position. The offsets and values pin the
    # multiply-high reduction.
    expected = (
        (
            0xf39608ad5487335f,
            272,
            Int32(904),
            (0x53e85b78ec2b79e4, 0xf39608ad5487335f),
            5850742046087865970,
            0xf39608ad5487335d,
        ),
        (
            0x2029c87b4a20bd10,
            35,
            Int32(-755),
            (0x2df82443a4e5ac43, 0x2029c87b4a20bd10),
            772534638369674330,
            0x2029c87b4a20bd15,
        ),
        (
            0x2a2d596a681fd002,
            47,
            Int32(-671),
            (0x8c72a1d5323ca8ba, 0x2a2d596a681fd002),
            1013061212364424533,
            0x2a2d596a681fd006,
        ),
        (
            0x624cb22e36a263ea,
            109,
            Int32(-237),
            (0xc3f54ce12431a0f8, 0x624cb22e36a263ea),
            2361077408500599800,
            0x624cb22e36a263ef,
        ),
        (
            0xd4b0fe71e0a7135a,
            237,
            Int32(659),
            (0xcec17fe430f8841d, 0xd4b0fe71e0a7135a),
            5108676432331867761,
            0xd4b0fe71e0a7135a,
        ),
        (
            0x0d7f8e4de91962b4,
            15,
            Int32(-895),
            (0xf5dd4c02409383c0, 0x0d7f8e4de91962b4),
            324217503269899153,
            0x0d7f8e4de91962ba,
        ),
        (
            0x44d3930c5cae4976,
            76,
            Int32(-468),
            (0x40ff67b3e35799b8, 0x44d3930c5cae4976),
            1653156431989621542,
            0x44d3930c5cae4979,
        ),
        (
            0x6dc19acb71673eb8,
            122,
            Int32(-146),
            (0xdb6a53caba025350, 0x6dc19acb71673eb8),
            2636257539736977297,
            0x6dc19acb71673eba,
        ),
    )

    span64 = length(range64) % UInt64
    span128 = length(range128) % UInt64
    for ((F, key), (candidate64, index64, value64, candidate128, index128, value128)) in
        zip(PACKED_GOLDEN_GENERATORS, expected)
        rng = _packed_golden_rng(F, key)
        block = _range_reference_block(rng.position)
        @test RangeIR._extract_bits_unchecked(
            rng,
            block,
            rng.position.bit,
            Val(64),
        ) === candidate64
        @test RangeIR._range_offset(rng, span64) === UInt64(index64)
        @test rand(rng, range64) === value64
        @test RangeIR._extract_bits128_unchecked(
            rng,
            block,
            rng.position.bit,
        ) === candidate128
        @test RangeIR._range_offset(rng, span128) === UInt64(index128)
        @test rand(rng, range128) === value128
    end
end

@testset "R55 packed integer ranges" begin
    threshold = UInt64(1) << 32
    @test RangeIR._range_bits(UInt64(1)) === UInt16(64)
    @test RangeIR._range_bits(threshold) === UInt16(64)
    @test RangeIR._range_bits(threshold + UInt64(1)) === UInt16(128)
    @test RangeIR._range_bits(UInt64(0)) === UInt16(128)

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
        block_bits = Int(RangeIR._block_bits(F(0)))
        bit = UInt16(block_bits - 1)
        for range in ranges
            rng = _range_positioned(F, 0x551, UInt64(9), bit)
            expected = _range_reference_draw(rng, range)
            position = rng.position
            @test rand(rng, range) === expected
            @test rng.position === position

            value, next_rng = rand_next(rng, range)
            width = _range_reference_width(length(range) % UInt64)
            @test value === expected
            @test next_rng.position == _range_reference_position(rng, width)
        end
    end
end

@testset "R8 and R53 mixed primitive and range positions" begin
    small = UInt16(3):UInt16(41)
    wide = UInt64(0):(UInt64(1)<<32)
    for F in GENERATOR_TYPES
        rng = _range_positioned(F, 0x552, UInt64(11), UInt16(63))
        first_expected = _reference_uniform(rng, UInt32)
        first_value, rng = rand_next(rng, UInt32)
        @test first_value === first_expected

        small_expected = _range_reference_draw(rng, small)
        small_position = _range_reference_position(rng, 64)
        small_value, rng = rand_next(rng, small)
        @test small_value === small_expected
        @test rng.position == small_position

        wide_expected = _range_reference_draw(rng, wide)
        wide_position = _range_reference_position(rng, 128)
        wide_value, rng = rand_next(rng, wide)
        @test wide_value === wide_expected
        @test rng.position == wide_position

        final_expected = _reference_uniform(rng, Bool)
        final_value, rng = rand_next(rng, Bool)
        @test final_value === final_expected
        @test rng.position ==
              _range_reference_position(RangeIR._rebuild(rng, wide_position, rng.device), 1)
    end
end

@testset "R53 and R54 range capacity and validation" begin
    small = UInt8(1):UInt8(7)
    wide = UInt64(0):(UInt64(1)<<32)
    for F in GENERATOR_TYPES, (range, width) in ((small, 64), (wide, 128))
        base = F(0x553)
        capacity = _range_capacity(base)
        last_position = _range_position_from_absolute(base, capacity - width)
        last = RangeIR._rebuild(base, last_position, base.device)
        expected = _range_reference_draw(last, range)
        value, terminal = rand_next(last, range)
        @test value === expected
        @test terminal.position == (
            base.position isa RangeIR._Position64 ?
            RangeIR._terminal64(RangeIR._max_block(base)) : RangeIR._terminal128()
        )

        insufficient_position = _range_position_from_absolute(base, capacity - width + 1)
        insufficient = RangeIR._rebuild(base, insufficient_position, base.device)
        @test_throws ArgumentError rand(insufficient, range)
        @test insufficient.position === insufficient_position
        @test_throws ArgumentError rand_next(insufficient, range)
        @test insufficient.position === insufficient_position

        exhausted = RangeIR._rebuild(base, terminal.position, base.device)
        @test_throws ArgumentError rand(exhausted, range)
        @test_throws ArgumentError rand_next(exhausted, range)
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

    rng = RangeIR.MLDataDevices.CUDADevice()(Philox4x64(0x557))
    for (function_, signature) in (
        (rand, Tuple{typeof(rng),typeof(UInt16(2):UInt16(17))}),
        (rand_next, Tuple{typeof(rng),typeof(UInt64(0):typemax(UInt64))}),
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
