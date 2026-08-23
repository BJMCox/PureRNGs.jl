const IR = PureRNGs

@testset "draw block mapping" begin
    @test IR.FAMILY_BITS === UInt32(0)

    family = UInt32(0xa5)
    block = UInt64(0x00123456789abcde)
    block128 = (UInt64(0x0123456789abcdef), UInt64(0xfedcba9876543210))

    p2x32 = Philox2x32((UInt32(0x89abcdef),))
    p4x32 = Philox4x32((UInt32(0x89abcdef), UInt32(0x01234567)))
    p2x64 = Philox2x64((UInt64(0x0123456789abcdef),))
    p4x64 = Philox4x64((UInt64(0x0123456789abcdef), UInt64(0xfedcba9876543210)))
    t2x32 = Threefry2x32((UInt32(0x89abcdef), UInt32(0x01234567)))
    t4x32 = Threefry4x32((
        UInt32(0x89abcdef),
        UInt32(0x01234567),
        UInt32(0xfedcba98),
        UInt32(0x76543210),
    ),)
    t2x64 = Threefry2x64((UInt64(0x0123456789abcdef), UInt64(0xfedcba9876543210)))
    t4x64 = Threefry4x64((
        UInt64(0x0123456789abcdef),
        UInt64(0xfedcba9876543210),
        UInt64(0x13579bdf2468ace0),
        UInt64(0xeca86420fdb97531),
    ),)

    narrow_counter =
        (block % UInt32, (family << 24) | (((block >> 32) & 0x00ffffff) % UInt32))
    wide32_counter = (block % UInt32, (block >> 32) % UInt32, family, UInt32(0))
    wide2x64_counter = (block, UInt64(family))
    wide4x64_counter = (block128..., UInt64(family), UInt64(0))

    @test IR._block(p2x32, family, block) == IR._philox2x32(narrow_counter, p2x32.key)
    @test IR._block(t2x32, family, block) == IR._threefry2x32(narrow_counter, t2x32.key)
    @test IR._block(p4x32, family, block) == IR._philox4x32(wide32_counter, p4x32.key)
    @test IR._block(t4x32, family, block) == IR._threefry4x32(wide32_counter, t4x32.key)
    @test IR._block(p2x64, family, block) == IR._philox2x64(wide2x64_counter, p2x64.key)
    @test IR._block(t2x64, family, block) == IR._threefry2x64(wide2x64_counter, t2x64.key)
    @test IR._block(p4x64, family, block128...) ==
          IR._philox4x64(wide4x64_counter, p4x64.key)
    @test IR._block(t4x64, family, block128...) ==
          IR._threefry4x64(wide4x64_counter, t4x64.key)

    # PureRNGsTestbed.jl 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d
    @test IR._block(p4x32, family, block) ==
          (UInt32(0xea75de19), UInt32(0xfb84293b), UInt32(0xe16f7d84), UInt32(0xd9b42cde))
    @test IR._block(t2x32, family, block) == (UInt32(0x12830fe3), UInt32(0xd895391c))

    @testset "boundaries" begin
        narrow_max = UInt64(0x00ffffffffffffff)
        narrow_max_counter = (typemax(UInt32), (family << 24) | UInt32(0x00ffffff))
        @test IR._block(p2x32, family, narrow_max) ==
              IR._philox2x32(narrow_max_counter, p2x32.key)
        @test IR._block(t2x32, family, narrow_max) ==
              IR._threefry2x32(narrow_max_counter, t2x32.key)

        block_max = typemax(UInt64)
        wide32_max_counter = (typemax(UInt32), typemax(UInt32), family, UInt32(0))
        @test IR._block(p4x32, family, block_max) ==
              IR._philox4x32(wide32_max_counter, p4x32.key)
        @test IR._block(t4x32, family, block_max) ==
              IR._threefry4x32(wide32_max_counter, t4x32.key)
        @test IR._block(p2x64, family, block_max) ==
              IR._philox2x64((block_max, UInt64(family)), p2x64.key)
        @test IR._block(t2x64, family, block_max) ==
              IR._threefry2x64((block_max, UInt64(family)), t2x64.key)

        block128_max = (typemax(UInt64), typemax(UInt64))
        wide4x64_max_counter = (block128_max..., UInt64(family), UInt64(0))
        @test IR._block(p4x64, family, block128_max...) ==
              IR._philox4x64(wide4x64_max_counter, p4x64.key)
        @test IR._block(t4x64, family, block128_max...) ==
              IR._threefry4x64(wide4x64_max_counter, t4x64.key)
    end

    calls = (
        (p2x32, family, block),
        (p4x32, family, block),
        (p2x64, family, block),
        (p4x64, family, block128...),
        (t2x32, family, block),
        (t4x32, family, block),
        (t2x64, family, block),
        (t4x64, family, block128...),
    )
    for args in calls
        @test @inferred(IR._block(args...)) == IR._block(args...)
        IR._block(args...)
        @test @allocated(IR._block(args...)) == 0
    end
end
