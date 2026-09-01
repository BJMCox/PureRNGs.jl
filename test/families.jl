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

@testset "R53 position layout" begin
    @test fieldtypes(PureRNGs._Position64) === (UInt64, UInt16)
    @test fieldtypes(PureRNGs._Position128) === (UInt64, UInt64, UInt16)
end

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
