# Every vector below is a line of the published Random123 v1.14.0 known-answer
# file, https://github.com/DEShawResearch/random123/blob/v1.14.0/tests/kat_vectors.

@testset "Threefry2x32 Random123 KATs" begin
    core = PureRNGs._threefry2x32
    @test core((UInt32(0), UInt32(0)), (UInt32(0), UInt32(0))) == (0x6b200159, 0x99ba4efe)
    @test core((0xffffffff, 0xffffffff), (0xffffffff, 0xffffffff)) ==
          (0x1cb996fc, 0xbb002be7)
    @test core((0x243f6a88, 0x85a308d3), (0x13198a2e, 0x03707344)) ==
          (0xc4923a9c, 0x483df7a0)
end

@testset "Threefry4x32 Random123 KATs" begin
    core = PureRNGs._threefry4x32
    @test core(ntuple(_ -> UInt32(0), 4), ntuple(_ -> UInt32(0), 4)) ==
          (0x9c6ca96a, 0xe17eae66, 0xfc10ecd4, 0x5256a7d8)
    @test core(ntuple(_ -> 0xffffffff, 4), ntuple(_ -> 0xffffffff, 4)) ==
          (0x2a881696, 0x57012287, 0xf6c7446e, 0xa16a6732)
    @test core(
        (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
        (0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89),
    ) == (0x59cd1dbb, 0xb8879579, 0x86b5d00c, 0xac8b6d84)
end

@testset "Threefry4x64-13 Random123 KATs" begin
    core = PureRNGs._threefry4x64
    @test core(ntuple(_ -> UInt64(0), 4), ntuple(_ -> UInt64(0), 4), Val(13)) ==
          (0x4071fabee1dc8e05, 0x02ed3113695c9c62, 0x397311b5b89f9d49, 0xe21292c3258024bc)
    ones = ntuple(_ -> 0xffffffffffffffff, 4)
    @test core(ones, ones, Val(13)) ==
          (0x7eaed935479722b5, 0x90994358c429f31c, 0x496381083e07a75b, 0x627ed0d746821121)
    @test core(
        (0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89),
        (0x452821e638d01377, 0xbe5466cf34e90c6c, 0xc0ac29b7c97c50dd, 0x3f84d5b5b5470917),
        Val(13),
    ) == (0x4361288ef9c1900c, 0x8717291521782833, 0x0d19db18c20cf47e, 0xa0b41d63ac8581e5)
end

@testset "Threefry2x64 Random123 KATs" begin
    core = PureRNGs._threefry2x64
    @test core((UInt64(0), UInt64(0)), (UInt64(0), UInt64(0))) ==
          (0xc2b6e3a8c2c69865, 0x6f81ed42f350084d)
    @test core(
        (0xffffffffffffffff, 0xffffffffffffffff),
        (0xffffffffffffffff, 0xffffffffffffffff),
    ) == (0xe02cb7c4d95d277a, 0xd06633d0893b8b68)
    @test core(
        (0x243f6a8885a308d3, 0x13198a2e03707344),
        (0xa4093822299f31d0, 0x082efa98ec4e6c89),
    ) == (0x263c7d30bb0f0af1, 0x56be8361d3311526)
end

@testset "Threefry4x64 Random123 KATs" begin
    core = PureRNGs._threefry4x64
    @test core(ntuple(_ -> UInt64(0), 4), ntuple(_ -> UInt64(0), 4)) ==
          (0x09218ebde6c85537, 0x55941f5266d86105, 0x4bd25e16282434dc, 0xee29ec846bd2e40b)
    @test core(ntuple(_ -> 0xffffffffffffffff, 4), ntuple(_ -> 0xffffffffffffffff, 4)) ==
          (0x29c24097942bba1b, 0x0371bbfb0f6f4e11, 0x3c231ffa33f83a1c, 0xcd29113fde32d168)
    # The published 20-round and 72-round pi rows repeat the second key word
    # instead of continuing the pi digits, unlike the 13-round row above. Keep
    # the key as published so the vector stays comparable with Random123.
    @test core(
        (0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89),
        (0x452821e638d01377, 0xbe5466cf34e90c6c, 0xbe5466cf34e90c6c, 0xc0ac29b7c97c50dd),
    ) == (0xa7e8fde591651bd9, 0xbaafd0c30138319b, 0x84a5c1a729e685b9, 0x901d406ccebc1ba4)
end
