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

@testset "Philox4x32-7 KAT" begin
    @test PureRNGs._philox4x32(
        (0x00000000, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x00000000),
        Val(7),
    ) == (0x5f6fb709, 0x0d893f64, 0x4f121f81, 0x4f730a48)
    @test PureRNGs._philox4x32(
        (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
        (0xffffffff, 0xffffffff),
        Val(7),
    ) == (0x5207ddc2, 0x45165e59, 0x4d8ee751, 0x8c52f662)
    @test PureRNGs._philox4x32(
        (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
        (0xa4093822, 0x299f31d0),
        Val(7),
    ) == (0x4dfccaba, 0x190a87f0, 0xc47362ba, 0xb6b5242a)
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

# The CPU 64-bit Philox generators run the core on host words so that the
# four-word core can use the x86 `mulq` sequence. That multiplication is the
# only thing the host words change, so the blocks must equal the portable ones.
@testset "CPU host-word Philox cores match the portable core" begin
    addresses = (
        (UInt64(0), UInt64(0)),
        (0x243f6a8885a308d3, 0x13198a2e03707344),
        (typemax(UInt64), typemax(UInt64)),
        (0x8000000000000000, UInt64(1)),
    )
    key2 = (0x452821e638d01377,)
    key4 = (0x452821e638d01377, 0xbe5466cf34e90c6c)
    for address in addresses
        @test PureRNGs._core_block(
            PureRNGs.Philox2x64{PureRNGs._CPUBackend,10},
            key2,
            address[1],
        ) == PureRNGs._philox2x64((address[1], UInt64(0)), key2, Val(10))
        @test PureRNGs._core_block(
            PureRNGs.Philox4x64{PureRNGs._CPUBackend,10},
            key4,
            address,
        ) == PureRNGs._philox4x64((address..., UInt64(0), UInt64(0)), key4, Val(10))
    end
end
