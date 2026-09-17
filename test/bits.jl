const BitsIR = PureRNGs

bits_allocations(rng, block) = (
    @allocated(BitsIR._extract_bits_unchecked(rng, block, UInt16(31), Val(64))),
    @allocated(BitsIR._extract_bits128_unchecked(rng, block, UInt16(63))),
)

const BIT_GENERATORS = (
    Philox2x32(0x1234),
    Philox4x32(0x1234),
    Philox2x64(0x1234),
    Philox4x64(0x1234),
    Threefry2x32(0x1234),
    Threefry4x32(0x1234),
    Threefry2x64(0x1234),
    Threefry4x64(0x1234),
    ChaCha(0x1234),
)

_reference_index(::BitsIR._Position64Generators) = UInt64(9)
_reference_index(::BitsIR._Position128Generators) = (UInt64(9), UInt64(7))

function _boundary_offsets(block_bits)
    candidates = (
        0,
        1,
        23,
        24,
        31,
        32,
        33,
        52,
        53,
        62,
        63,
        64,
        65,
        block_bits - 65,
        block_bits - 64,
        block_bits - 63,
        block_bits - 33,
        block_bits - 32,
        block_bits - 31,
        block_bits - 2,
        block_bits - 1,
    )
    return unique(UInt16(value) for value in candidates if 0 <= value < block_bits)
end

@testset "canonical packed-bit extraction" begin
    relevant_widths = (1, 23, 24, 32, 52, 53, 64)

    overflow_block = (typemax(UInt64), UInt64(7))
    @test BitsIR._next_stream_block_unchecked(overflow_block) == (UInt64(0), UInt64(8))
    for rng in (Philox4x64(0x1234), Threefry4x64(0x1234))
        bit = UInt16(255)
        @test BitsIR._extract_bits_unchecked(rng, overflow_block, bit, Val(64)) ==
              _reference_extract(rng, overflow_block, bit, 64)
        @test BitsIR._extract_bits128_unchecked(rng, overflow_block, bit) ==
              _reference_extract128(rng, overflow_block, bit)
    end

    for rng in BIT_GENERATORS
        block = _reference_index(rng)
        raw = _reference_block(rng, block)
        block_words = @inferred BitsIR._block_words(rng, block)
        expected_block_words = if first(raw) isa UInt32
            ntuple(
                lane -> (UInt64(raw[2lane-1]) << 32) | UInt64(raw[2lane]),
                Val(length(raw) ÷ 2),
            )
        else
            raw
        end
        @test block_words == expected_block_words

        block_bits = 8sizeof(first(raw)) * length(raw)
        offsets = _boundary_offsets(block_bits)
        for width in relevant_widths, bit in offsets
            @test BitsIR._extract_bits_unchecked(rng, block, bit, Val(width)) ==
                  _reference_extract(rng, block, bit, width)
        end

        for bit in offsets
            candidate = BitsIR._extract_bits128_unchecked(rng, block, bit)
            @test candidate == _reference_extract128(rng, block, bit)
        end

        @test @inferred(
            BitsIR._extract_bits_unchecked(rng, block, UInt16(0), Val(1))
        ) isa UInt64
        @test @inferred(
            BitsIR._extract_bits_unchecked(rng, block, UInt16(31), Val(64))
        ) isa UInt64
        @test @inferred(
            BitsIR._extract_bits128_unchecked(rng, block, UInt16(63))
        ) isa Tuple{UInt64,UInt64}

        bits_allocations(rng, block)
        @test bits_allocations(rng, block) == (0, 0)

        index_type = typeof(block)
        kernel_rng = BitsIR.MLDataDevices.CUDADevice()(rng)
        scalar_ir = sprint(
            show,
            code_typed(
                BitsIR._extract_bits_unchecked,
                Tuple{typeof(kernel_rng),index_type,UInt16,Val{64}};
                optimize = true,
            ),
        )
        candidate_ir = sprint(
            show,
            code_typed(
                BitsIR._extract_bits128_unchecked,
                Tuple{typeof(kernel_rng),index_type,UInt16};
                optimize = true,
            ),
        )
        for typed_ir in (scalar_ir, candidate_ir)
            @test !isempty(typed_ir)
            @test !occursin("BigInt", typed_ir)
            @test !occursin("UInt128", typed_ir)
        end
    end
end
