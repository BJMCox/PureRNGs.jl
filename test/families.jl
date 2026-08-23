using MLDataDevices

const FAMILY_TYPES = (
    Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
)

@testset "R4 and R14 family representation" begin
    for F in FAMILY_TYPES
        rng = F(0)
        @test supertype(typeof(rng)) === PureRNGs.AbstractPureRNG
        @test isbitstype(typeof(rng))
        @test fieldnames(typeof(rng)) === (:key, :position, :device)
        @test MLDataDevices.get_device(rng) == MLDataDevices.CPUDevice()
        @test rng.position.lane == 0x00
    end
end

@testset "R15 and R16 seed validation and mapping" begin
    @test Philox2x32(0x12345678).key == (0x12345678,)
    @test Philox4x32(0x123456789abcdef0).key == (0x9abcdef0, 0x12345678)
    @test Philox2x64(0x123456789abcdef0).key == (0x123456789abcdef0,)
    @test Philox4x64(big"0x123456789abcdef00fedcba987654321").key ==
          (0x0fedcba987654321, 0x123456789abcdef0)
    @test Threefry2x32(0x123456789abcdef0).key == (0x9abcdef0, 0x12345678)
    @test Threefry4x32(big"0x123456789abcdef00fedcba987654321").key ==
          (0x87654321, 0x0fedcba9, 0x9abcdef0, 0x12345678)
    @test Threefry2x64(big"0x123456789abcdef00fedcba987654321").key ==
          (0x0fedcba987654321, 0x123456789abcdef0)
    @test Threefry4x64(
        big"0x123456789abcdef00fedcba987654321112233445566778899aabbccddeeff00",
    ).key ==
          (0x99aabbccddeeff00, 0x1122334455667788, 0x0fedcba987654321, 0x123456789abcdef0)

    for (F, bits) in zip(FAMILY_TYPES, (32, 64, 64, 128, 64, 128, 128, 256))
        @test_throws ArgumentError F(-1)
        @test_throws ArgumentError F(big(1) << bits)
        @test all(isone, F((big(1) << bits) - 1).key .== typemax.(typeof.(F(0).key)))
    end
end

@testset "R38 device application" begin
    rng = Philox4x32(123)
    rebound = MLDataDevices.CPUDevice()(rng)
    @test rebound.key == rng.key
    @test rebound.position == rng.position
    @test MLDataDevices.get_device(rebound) == MLDataDevices.CPUDevice()
end

@testset "R53 and R54 checked reservation" begin
    narrow = Threefry2x32(1)
    narrow_last = PureRNGs._rebuild(
        narrow,
        PureRNGs._Position64(0x00ffffffffffffff, 0x01),
        narrow.device,
    )
    narrow_end = PureRNGs._reserve(narrow_last, UInt64(1))
    @test PureRNGs._is_exhausted(narrow_end.position)
    @test PureRNGs._reserve(narrow_end, UInt64(0)) === narrow_end
    @test_throws ArgumentError PureRNGs._reserve(narrow_end, UInt64(1))

    wide64 = Philox4x32(1)
    wide64_last = PureRNGs._rebuild(
        wide64,
        PureRNGs._Position64(typemax(UInt64), 0x03),
        wide64.device,
    )
    @test PureRNGs._is_exhausted(
        PureRNGs._reserve(wide64_last, UInt64(1)).position,
    )
    @test_throws ArgumentError PureRNGs._reserve(wide64_last, UInt64(2))
    wide64_penultimate = PureRNGs._rebuild(
        wide64,
        PureRNGs._Position64(typemax(UInt64) - 1, 0x03),
        wide64.device,
    )
    @test PureRNGs._is_exhausted(
        PureRNGs._reserve(wide64_penultimate, UInt64(5)).position,
    )
    @test_throws ArgumentError PureRNGs._reserve(wide64_penultimate, UInt64(6))

    wide128 = Threefry4x64(1)
    wide128_last = PureRNGs._rebuild(
        wide128,
        PureRNGs._Position128(typemax(UInt64), typemax(UInt64), 0x07),
        wide128.device,
    )
    wide128_end = PureRNGs._reserve(wide128_last, UInt64(1))
    @test PureRNGs._is_exhausted(wide128_end.position)
    @test_throws ArgumentError PureRNGs._reserve(wide128_end, UInt64(1))
    wide128_crossing = PureRNGs._rebuild(
        wide128,
        PureRNGs._Position128(typemax(UInt64), typemax(UInt64) - 1, 0x07),
        wide128.device,
    )
    @test PureRNGs._reserve(wide128_crossing, UInt64(2)).position ==
          PureRNGs._Position128(0, typemax(UInt64), 0x01)
    wide128_penultimate = PureRNGs._rebuild(
        wide128,
        PureRNGs._Position128(typemax(UInt64) - 1, typemax(UInt64), 0x07),
        wide128.device,
    )
    @test PureRNGs._is_exhausted(
        PureRNGs._reserve(wide128_penultimate, UInt64(9)).position,
    )
    @test_throws ArgumentError PureRNGs._reserve(wide128_penultimate, UInt64(10))

    crossing = PureRNGs._reserve(Philox4x32(1), UInt64(5))
    @test crossing.position == PureRNGs._Position64(0x01, 0x01)
end

@testset "R4 and R53 inference and allocation" begin
    rng = Philox4x32(123)
    @test @inferred(PureRNGs._reserve(rng, UInt64(2))) isa typeof(rng)
    PureRNGs._reserve(rng, UInt64(2))
    @test @allocated(PureRNGs._reserve(rng, UInt64(2))) == 0
end
