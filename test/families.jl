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
const BACKEND_TOKENS = (
    PureRNGs._CPU_BACKEND,
    PureRNGs._CUDA_BACKEND,
    PureRNGs._AMDGPU_BACKEND,
    PureRNGs._METAL_BACKEND,
)

@testset "R4 and R14 family representation" begin
    for F in FAMILY_TYPES
        rng = F(0)
        @test supertype(typeof(rng)) === PureRNGs.AbstractPureRNG
        @test isbitstype(typeof(rng))
        @test fieldnames(typeof(rng)) === (:key, :position, :device)
        @test rng.device === PureRNGs._CPU_BACKEND
        @test sizeof(rng.device) == 0
        @test MLDataDevices.get_device_type(rng.device) === MLDataDevices.CPUDevice
        @test all(
            name -> iszero(getfield(rng.position, name)),
            fieldnames(typeof(rng.position)),
        )
        @test F(rng.key).key === rng.key
        @test !applicable(F, rng.key, rng.position, rng.device)
        @test_throws TypeError Core.apply_type(F, MLDataDevices.UnknownDevice)
    end

    @test length(unique(typeof.(BACKEND_TOKENS))) == 4
    @test all(token -> token isa PureRNGs._BackendToken, BACKEND_TOKENS)
    @test all(token -> isbits(token) && sizeof(token) == 0, BACKEND_TOKENS)
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
    devices = (
        (MLDataDevices.CPUDevice(), PureRNGs._CPU_BACKEND, MLDataDevices.CPUDevice),
        (
            MLDataDevices.CUDADevice(:discarded),
            PureRNGs._CUDA_BACKEND,
            MLDataDevices.CUDADevice,
        ),
        (
            MLDataDevices.AMDGPUDevice(:discarded),
            PureRNGs._AMDGPU_BACKEND,
            MLDataDevices.AMDGPUDevice,
        ),
        (
            MLDataDevices.MetalDevice(),
            PureRNGs._METAL_BACKEND,
            MLDataDevices.MetalDevice,
        ),
    )

    @test length(unique(last.(devices))) == 4
    for (device, token, device_type) in devices
        rebound = device(rng)
        @test rebound.key == rng.key
        @test rebound.position == rng.position
        @test rebound.device === token
        @test isbits(rebound.device)
        @test sizeof(rebound.device) == 0
        @test MLDataDevices.get_device_type(rebound.device) === device_type
        @test MLDataDevices.CPUDevice()(rebound).device === PureRNGs._CPU_BACKEND
    end

    @test MLDataDevices.CUDADevice()(rng).device === PureRNGs._CUDA_BACKEND
    @test MLDataDevices.CUDADevice(:first)(rng).device ===
          MLDataDevices.CUDADevice(:second)(rng).device
    @test MLDataDevices.with_eltype(MLDataDevices.CUDADevice(:discarded), Float32)(rng).device ===
          PureRNGs._CUDA_BACKEND
    @test MLDataDevices.with_eltype(MLDataDevices.CPUDevice(), Float32)(rng).device ===
          PureRNGs._CPU_BACKEND
    @test_throws ArgumentError MLDataDevices.oneAPIDevice()(rng)
    @test_throws ArgumentError MLDataDevices.ReactantDevice()(rng)
end

@testset "R38 active-device documentation" begin
    documentation = read(
        joinpath(pkgdir(PureRNGs), "docs", "src", "guides", "devices.md"),
        String,
    )
    @test occursin("discards any physical device", documentation)
    @test occursin("active device", documentation)
    @test occursin("CUDA.device!", documentation)
    @test occursin("AMDGPU", documentation)
    @test occursin("Metal", documentation)
end

@testset "R53 exact packed-bit position representation" begin
    @test fieldtypes(PureRNGs._Position64) === (UInt64, UInt16)
    @test fieldtypes(PureRNGs._Position128) === (UInt64, UInt64, UInt16)

    for (F, block_bits) in zip(FAMILY_TYPES, (64, 128, 128, 256, 64, 128, 128, 256))
        rng = F(0)
        @test PureRNGs._block_bits(rng) === UInt16(block_bits)
        @test rng.position.bit === UInt16(0)
    end
end

@testset "R53 and R54 checked packed-bit reservation" begin
    cases = (
        (Threefry2x32(1), UInt16(64), UInt64(0x00ffffffffffffff)),
        (Philox4x32(1), UInt16(128), typemax(UInt64)),
        (Threefry4x64(1), UInt16(256), nothing),
    )

    for (rng, block_bits, maximum) in cases
        last_position =
            maximum isa UInt64 ?
            PureRNGs._Position64(maximum, block_bits - UInt16(1)) :
            PureRNGs._Position128(
                typemax(UInt64),
                typemax(UInt64),
                block_bits - UInt16(1),
            )
        terminal =
            maximum isa UInt64 ? PureRNGs._Position64(maximum, typemax(UInt16)) :
            PureRNGs._Position128(typemax(UInt64), typemax(UInt64), typemax(UInt16))
        last = PureRNGs._rebuild(rng, last_position, rng.device)
        exhausted = PureRNGs._reserve(last, UInt64(1), UInt64(0))
        @test exhausted.position == terminal
        @test PureRNGs._reserve(exhausted, UInt64(0), UInt64(0)) === exhausted
        @test_throws ArgumentError PureRNGs._reserve(exhausted, UInt64(1), UInt64(0))
        @test_throws ArgumentError PureRNGs._reserve(last, UInt64(2), UInt64(0))
        @test last.position == last_position
    end

    crossing = PureRNGs._rebuild(
        Threefry4x64(1),
        PureRNGs._Position128(typemax(UInt64), UInt64(6), UInt16(255)),
        Threefry4x64(1).device,
    )
    @test PureRNGs._reserve(crossing, UInt64(2), UInt64(0)).position ==
          PureRNGs._Position128(UInt64(0), UInt64(7), UInt16(1))
end

@testset "R53 invalid packed-bit positions are rejected" begin
    narrow = Threefry2x32(1)
    invalid_positions = (
        PureRNGs._Position64(UInt64(0), UInt16(64)),
        PureRNGs._Position64(UInt64(0), typemax(UInt16) - UInt16(1)),
        PureRNGs._Position64(UInt64(0), typemax(UInt16)),
        PureRNGs._Position64(UInt64(0x0100000000000000), UInt16(0)),
    )
    for position in invalid_positions
        invalid = PureRNGs._rebuild(narrow, position, narrow.device)
        @test_throws ArgumentError PureRNGs._reserve(invalid, UInt64(0), UInt64(0))
        @test_throws ArgumentError PureRNGs._reserve(invalid, UInt64(1), UInt64(0))
    end

    wide = Threefry4x64(1)
    for bit in (UInt16(256), typemax(UInt16) - UInt16(1))
        invalid = PureRNGs._rebuild(
            wide,
            PureRNGs._Position128(UInt64(0), UInt64(0), bit),
            wide.device,
        )
        @test_throws ArgumentError PureRNGs._reserve(invalid, UInt64(0), UInt64(0))
    end
end

@testset "R30 and R54 checked bit spans" begin
    @test PureRNGs._bit_span(UInt64(0), UInt16(128)) == (UInt64(0), UInt64(0))
    @test PureRNGs._bit_span(UInt64(3), UInt16(53)) == (UInt64(159), UInt64(0))
    @test PureRNGs._bit_span(UInt64(typemax(Int)), UInt16(128)) ==
          (UInt64(0xffffffffffffff80), UInt64(0x3f))

    for rng in (Philox2x32(123), Philox4x32(123), Philox4x64(123))
        result = @inferred PureRNGs._reserve(rng, UInt64(159), UInt64(0))
        @test result isa typeof(rng)
        PureRNGs._reserve(rng, UInt64(159), UInt64(0))
        @test @allocated(PureRNGs._reserve(rng, UInt64(159), UInt64(0))) == 0
    end
end
