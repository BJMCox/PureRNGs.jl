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
        @test rng.device == MLDataDevices.CPUDevice()
        @test all(
            name -> iszero(getfield(rng.position, name)),
            fieldnames(typeof(rng.position)),
        )
        @test F(rng.key).key === rng.key
        @test !applicable(F, rng.key, rng.position, rng.device)
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
    @test rebound.device == MLDataDevices.CPUDevice()
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
    exact = PureRNGs._reserve(Philox4x32(1), UInt64(4))
    @test exact.position == PureRNGs._Position64(0x01, 0x00)
end

@testset "R4 and R53 inference and allocation" begin
    for rng in (Philox2x32(123), Philox4x32(123), Philox4x64(123))
        @test @inferred(PureRNGs._reserve(rng, UInt64(2))) isa typeof(rng)
        PureRNGs._reserve(rng, UInt64(2))
        @test @allocated(PureRNGs._reserve(rng, UInt64(2))) == 0
    end
end

@testset "R53 aligned reservation" begin
    narrow = Threefry2x32(1)
    narrow_odd =
        PureRNGs._rebuild(narrow, PureRNGs._Position64(1, 0), narrow.device)
    narrow_start, narrow_next =
        PureRNGs._reserve_aligned(narrow_odd, UInt64(4), UInt64(4))
    @test narrow_start.position == PureRNGs._Position64(2, 0)
    @test narrow_next.position == PureRNGs._Position64(4, 0)

    narrow_last = PureRNGs._rebuild(
        narrow,
        PureRNGs._Position64(0x00fffffffffffffd, 0),
        narrow.device,
    )
    last_start, last_next =
        PureRNGs._reserve_aligned(narrow_last, UInt64(4), UInt64(4))
    @test last_start.position == PureRNGs._Position64(0x00fffffffffffffe, 0)
    @test PureRNGs._is_exhausted(last_next.position)
    @test_throws ArgumentError PureRNGs._reserve_aligned(
        PureRNGs._rebuild(
            narrow,
            PureRNGs._Position64(0x00ffffffffffffff, 0),
            narrow.device,
        ),
        UInt64(4),
        UInt64(4),
    )

    wide64 = Philox4x32(1)
    wide64_unaligned =
        PureRNGs._rebuild(wide64, PureRNGs._Position64(3, 1), wide64.device)
    pair_start, pair_next =
        PureRNGs._reserve_aligned(wide64_unaligned, UInt64(2), UInt64(2))
    @test pair_start.position == PureRNGs._Position64(3, 2)
    @test pair_next.position == PureRNGs._Position64(4, 0)
    quad_start, quad_next =
        PureRNGs._reserve_aligned(wide64_unaligned, UInt64(4), UInt64(4))
    @test quad_start.position == PureRNGs._Position64(4, 0)
    @test quad_next.position == PureRNGs._Position64(5, 0)

    wide128 = Threefry4x64(1)
    wide128_crossing = PureRNGs._rebuild(
        wide128,
        PureRNGs._Position128(typemax(UInt64), 0, 7),
        wide128.device,
    )
    crossing_start, crossing_next =
        PureRNGs._reserve_aligned(wide128_crossing, UInt64(4), UInt64(4))
    @test crossing_start.position == PureRNGs._Position128(0, 1, 0)
    @test crossing_next.position == PureRNGs._Position128(0, 1, 4)

    exhausted = PureRNGs._reserve(
        PureRNGs._rebuild(
            wide128,
            PureRNGs._Position128(typemax(UInt64), typemax(UInt64), 7),
            wide128.device,
        ),
        UInt64(1),
    )
    zero_start, zero_next = PureRNGs._reserve_aligned(exhausted, UInt64(0), UInt64(4))
    @test zero_start === exhausted
    @test zero_next === exhausted
    @test PureRNGs._reserve_aligned(exhausted, UInt64(0), UInt64(0)) ===
          (exhausted, exhausted)

    for rng in (narrow_odd, wide64_unaligned, wide128_crossing)
        result = @inferred PureRNGs._reserve_aligned(rng, UInt64(4), UInt64(4))
        @test result isa Tuple{typeof(rng),typeof(rng)}
        PureRNGs._reserve_aligned(rng, UInt64(4), UInt64(4))
        @test @allocated(PureRNGs._reserve_aligned(rng, UInt64(4), UInt64(4))) == 0
    end

    @test_throws ArgumentError PureRNGs._reserve_aligned(narrow, UInt64(1), UInt64(0))
    @test_throws ArgumentError PureRNGs._reserve_aligned(narrow, UInt64(1), UInt64(3))
end
