# This fixture contains only R2 raw cores, counter layouts, and key derivation.
# It intentionally contains no draw conversion or packed-stream value. Every draw
# kind reads the same counter region, so one block output per generator suffices.
# Captured from commit 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d.
const PINNED_TESTBED_ORACLE = (
    constants = (
        derive_tag = UInt32(0xc0ffee00),
        split_subtag = UInt32(0),
        fold_subtag = UInt32(1),
        threefry_fold_index = UInt32(0xffffffff),
    ),
    generators = (
        (
            generator = Philox4x32,
            key = (UInt32(0x01234567), UInt32(0x89abcdef)),
            draw_block = UInt64(0x0123456789abcdef),
            draw_output = (0x38e4febf, 0x1c30d87a, 0x3e07256e, 0x43c5e19e),
            maximum_draw_block = typemax(UInt64),
            split_keys = (
                (0x730767c8, 0x34b3bda3),
                (0x36485763, 0x4591f4ae),
                (0x5e5aa077, 0x1242f338),
                (0x1367c2c5, 0xc4916029),
            ),
            far_split = (
                index = UInt64(0x0000000200000003),
                key = (0x1c171a61, 0xefbb143b),
            ),
            subrng_keys = (
                zero = (0xaeb2701f, 0xaf683b45),
                purpose = (0xcf578767, 0xb4b42840),
            ),
        ),
        (
            generator = Threefry2x32,
            key = (UInt32(0x01234567), UInt32(0x89abcdef)),
            draw_block = UInt64(0x00abcdeffedcba98),
            draw_output = (0x195784d4, 0x089df171),
            maximum_draw_block = UInt64(0x00ffffffffffffff),
            split_keys = (
                (0x011ac086, 0x5205f808),
                (0xcb2fd2f2, 0xc0f6498c),
                (0xb8d80ff1, 0xe876ab70),
                (0x62f19539, 0x0e413987),
            ),
            far_split = (index = UInt64(0xfffffffe), key = (0x4a0bb8c6, 0xc9211a0e)),
            subrng_keys = (
                zero = (0x920e1914, 0x6fc51759),
                purpose = (0x96551a73, 0x5c4b7b9a),
            ),
        ),
    ),
)

_testbed_core(::Type{Philox4x32}, counter, key) = PureRNGs._philox4x32(counter, key)
_testbed_core(::Type{Threefry2x32}, counter, key) =
    PureRNGs._threefry2x32(counter, key)

function _testbed_draw_counter(::Type{Philox4x32}, block::UInt64)
    return block % UInt32, (block >> 32) % UInt32, UInt32(0), UInt32(0)
end
function _testbed_draw_counter(::Type{Threefry2x32}, block::UInt64)
    return block % UInt32, ((block >> 32) & 0x00ffffff) % UInt32
end

@testset "R2 pinned testbed agreement" begin
    oracle = PINNED_TESTBED_ORACLE

    constants = oracle.constants
    @test PureRNGs._DERIVE_TAG === constants.derive_tag
    @test PureRNGs._SPLIT_SUBTAG === constants.split_subtag
    @test PureRNGs._FOLD_SUBTAG === constants.fold_subtag
    @test PureRNGs._THREEFRY_FOLD_INDEX === constants.threefry_fold_index

    purpose = 0x0123456789abcdef
    for case in oracle.generators
        rng = case.generator(case.key)
        @test PureRNGs._max_block(rng) === case.maximum_draw_block

        counter = _testbed_draw_counter(case.generator, case.draw_block)
        @test _testbed_core(case.generator, counter, case.key) == case.draw_output
        @test PureRNGs._block(rng, case.draw_block) == case.draw_output

        @test getfield.(splitrng(rng), :key) == case.split_keys[1:2]
        @test getfield.(splitrng(rng, Val(4)), :key) == case.split_keys
        @test getfield.(splitrng(rng, 4), :key) == collect(case.split_keys)
        @test getfield.(
            ntuple(i -> PureRNGs._derive_child(rng, UInt64(i - 1)), Val(4)),
            :key,
        ) == case.split_keys
        @test PureRNGs._derive_child(rng, case.far_split.index).key ==
              case.far_split.key
        @test subrng(rng, 0).key == case.subrng_keys.zero
        @test subrng(rng, purpose).key == case.subrng_keys.purpose
    end
end
