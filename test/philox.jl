@testset "Philox2x32-10 KAT" begin
    @test PureRNGs._philox2x32((0x00000000, 0x00000000), (0x00000000,)) ==
          (0xff1dae59, 0x6cd10df2)
    @test PureRNGs._philox2x32((0xffffffff, 0xffffffff), (0xffffffff,)) ==
          (0x2c3f628b, 0xab4fd7ad)
    @test PureRNGs._philox2x32((0x243f6a88, 0x85a308d3), (0x13198a2e,)) ==
          (0xdd7ce038, 0xf62a4c12)
end

@testset "Philox4x32-10 KAT" begin
    @test PureRNGs._philox4x32(
        (0x00000000, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x00000000),
    ) == (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
    @test PureRNGs._philox4x32(
        (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
        (0xffffffff, 0xffffffff),
    ) == (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)
    @test PureRNGs._philox4x32(
        (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
        (0xa4093822, 0x299f31d0),
    ) == (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)
end

@inline function _consume_blocks4(rng, family, block)
    blocks = PureRNGs._blocks4(rng, family, block)
    return blocks[1][1] ⊻ blocks[2][1] ⊻ blocks[3][1] ⊻ blocks[4][1]
end
@noinline _blocks4_allocations(rng) =
    @allocated _consume_blocks4(rng, PureRNGs.FAMILY_BITS, UInt64(7))

@testset "Philox4x32 four-block core" begin
    for family in (UInt32(0), UInt32(1), UInt32(0x89abcdef)),
        block in (
            UInt64(0),
            UInt64(1),
            UInt64(0xfffffffe),
            UInt64(0xffffffff),
            UInt64(0x1_00000000),
            typemax(UInt64) - UInt64(3),
        )

        rng = Philox4x32((UInt32(0x01234567), UInt32(0x89abcdef)))
        @test PureRNGs._blocks4(rng, family, block) ==
              ntuple(i -> PureRNGs._block(rng, family, block + UInt64(i - 1)), Val(4))
    end

    rng = Philox4x32(0x1234)
    @test @inferred(PureRNGs._blocks4(rng, PureRNGs.FAMILY_BITS, UInt64(7))) isa
          NTuple{4,NTuple{4,UInt32}}
    _consume_blocks4(rng, PureRNGs.FAMILY_BITS, UInt64(7))
    @test _blocks4_allocations(rng) == 0
end

@testset "Philox2x64-10 KAT" begin
    @test PureRNGs._philox2x64(
        (0x0000000000000000, 0x0000000000000000),
        (0x0000000000000000,),
    ) == (0xca00a0459843d731, 0x66c24222c9a845b5)
    @test PureRNGs._philox2x64(
        (0xffffffffffffffff, 0xffffffffffffffff),
        (0xffffffffffffffff,),
    ) == (0x65b021d60cd8310f, 0x4d02f3222f86df20)
    @test PureRNGs._philox2x64(
        (0x243f6a8885a308d3, 0x13198a2e03707344),
        (0xa4093822299f31d0,),
    ) == (0x0a5e742c2997341c, 0xb0f883d38000de5d)
end

@testset "Philox4x64-10 KAT" begin
    @test PureRNGs._philox4x64(
        (0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000),
        (0x0000000000000000, 0x0000000000000000),
    ) == (0x16554d9eca36314c, 0xdb20fe9d672d0fdc, 0xd7e772cee186176b, 0x7e68b68aec7ba23b)
    @test PureRNGs._philox4x64(
        (0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff),
        (0xffffffffffffffff, 0xffffffffffffffff),
    ) == (0x87b092c3013fe90b, 0x438c3c67be8d0224, 0x9cc7d7c69cd777b6, 0xa09caebf594f0ba0)
    @test PureRNGs._philox4x64(
        (0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89),
        (0x452821e638d01377, 0xbe5466cf34e90c6c),
    ) == (0xa528f45403e61d95, 0x38c72dbd566e9788, 0xa5a1610e72fd18b5, 0x57bd43b5e52b7fe6)
end

@testset "Philox core inference and allocation" begin
    let ctr = (0x243f6a88, 0x85a308d3), key = (0x13198a2e,)
        @test @inferred(PureRNGs._philox2x32(ctr, key)) isa NTuple{2,UInt32}
        PureRNGs._philox2x32(ctr, key)
        @test @allocated(PureRNGs._philox2x32(ctr, key)) == 0
    end
    let ctr = (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
        key = (0xa4093822, 0x299f31d0)

        @test @inferred(PureRNGs._philox4x32(ctr, key)) isa NTuple{4,UInt32}
        PureRNGs._philox4x32(ctr, key)
        @test @allocated(PureRNGs._philox4x32(ctr, key)) == 0
    end
    let ctr = (0x243f6a8885a308d3, 0x13198a2e03707344), key = (0xa4093822299f31d0,)
        @test @inferred(PureRNGs._philox2x64(ctr, key)) isa NTuple{2,UInt64}
        PureRNGs._philox2x64(ctr, key)
        @test @allocated(PureRNGs._philox2x64(ctr, key)) == 0
    end
    let ctr = (
            0x243f6a8885a308d3,
            0x13198a2e03707344,
            0xa4093822299f31d0,
            0x082efa98ec4e6c89,
        ),
        key = (0x452821e638d01377, 0xbe5466cf34e90c6c)

        @test @inferred(PureRNGs._philox4x64(ctr, key)) isa NTuple{4,UInt64}
        PureRNGs._philox4x64(ctr, key)
        @test @allocated(PureRNGs._philox4x64(ctr, key)) == 0
    end
end

@testset "Philox core width applicability" begin
    @test !applicable(PureRNGs._philox2x32, (UInt64(0), UInt64(0)), (UInt64(0),))
    @test !applicable(
        PureRNGs._philox4x32,
        ntuple(_ -> UInt64(0), Val(4)),
        (UInt64(0), UInt64(0)),
    )
    @test !applicable(PureRNGs._philox2x64, (UInt32(0), UInt32(0)), (UInt32(0),))
    @test !applicable(
        PureRNGs._philox4x64,
        ntuple(_ -> UInt32(0), Val(4)),
        (UInt32(0), UInt32(0)),
    )
end
