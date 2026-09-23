using MLDataDevices

const BACKEND_TOKENS = (
    PureRNGs._CPU_BACKEND,
    PureRNGs._CUDA_BACKEND,
    PureRNGs._AMDGPU_BACKEND,
    PureRNGs._METAL_BACKEND,
)

@testset "position layout" begin
    @test fieldtypes(PureRNGs._Position64) === (UInt64, UInt16)
    @test fieldtypes(PureRNGs._Position128) === (UInt64, UInt64, UInt16)
end

@testset "generator representation" begin
    for F in GENERATOR_TYPES
        rng = F(0)
        @test supertype(typeof(rng)) === PureRNGs.AbstractPureRNG
        @test isbitstype(typeof(rng))
        @test fieldnames(typeof(rng)) === (:key, :position, :device, :block_words)
        @test rng.block_words ===
              PureRNGs._block_words(rng, PureRNGs._position_block(rng.position))
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

@testset "seed validation and mapping" begin
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
    @test ChaCha(big"0x123456789abcdef00fedcba987654321112233445566778899aabbccddeeff00").key ==
          (
        0xddeeff00,
        0x99aabbcc,
        0x55667788,
        0x11223344,
        0x87654321,
        0x0fedcba9,
        0x9abcdef0,
        0x12345678,
    )

    for F in GENERATOR_TYPES
        bits = 8 * sizeof(fieldtype(F, :key))
        @test_throws ArgumentError F(-1)
        @test_throws ArgumentError F(big(1) << bits)
        @test all(isone, F((big(1) << bits) - 1).key .== typemax.(typeof.(F(0).key)))
    end
end

@testset "device application" begin
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
        (MLDataDevices.MetalDevice(), PureRNGs._METAL_BACKEND, MLDataDevices.MetalDevice),
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

@testset "round-reduced generators" begin
    cases = (
        (Philox4x32R7, Philox4x32, 7),
        (Philox2x64R6, Philox2x64, 6),
        (Philox4x64R7, Philox4x64, 7),
        (Threefry2x64R13, Threefry2x64, 13),
        (Threefry4x32R12, Threefry4x32, 12),
        (Threefry4x64R13, Threefry4x64, 13),
        (ChaCha8, ChaCha, 8),
        (ChaCha20, ChaCha, 20),
    )
    for (alias, base, rounds) in cases
        reduced = alias(0x1234)
        full = base(0x1234)
        @test reduced isa base
        @test PureRNGs._rounds(typeof(reduced)) == rounds
        @test PureRNGs._rounds(typeof(full)) == PureRNGs._default_rounds(base)
        @test rngkey(reduced) == rngkey(full)
        @test rand(reduced, UInt64) != rand(full, UInt64)

        # The reference extracts from the core at the alias's round count, so
        # this pins the rounds the alias selects, not only its type parameter.
        cursor = reduced
        expected = Vector{UInt32}(undef, 4)
        for index in eachindex(expected)
            expected[index] = _reference_uniform(cursor, UInt32)
            cursor = PureRNGs._rebuild(
                cursor,
                _reference_position(cursor, _uniform_width(UInt32)),
                cursor.device,
            )
        end
        @test rand(reduced, UInt32, 4) == expected
        @test reduced.block_words ==
              PureRNGs._block_words(reduced, PureRNGs._position_block(reduced.position))
        _, moved = rand_next(reduced, Float64, 5)
        @test alias(rngkey(moved), rngposition(moved)) === moved
        @test alias(0x1234, rngposition(moved)) === moved
        @test typeof(splitrng(reduced)[1]) === typeof(reduced)
        @test typeof(MLDataDevices.CUDADevice()(reduced)) === alias{PureRNGs._CUDABackend}
    end
end

@testset "key and position access and reconstruction" begin
    for F in GENERATOR_TYPES
        rng = F(7)
        @test rngkey(rng) === rng.key
        @test iszero(rngposition(rng))
        @test F(rngkey(rng), rngposition(rng)) === rng

        _, moved = rand_next(rng, Bool)
        _, moved = rand_next(moved, Float64, 3)
        _, moved = rand_next(moved, UInt64)
        @test rngposition(moved) == 1 + 3 * 53 + 64
        @test F(rngkey(moved), rngposition(moved)) === moved
        @test F(7, rngposition(moved)) === moved
        @test rngposition(MLDataDevices.CUDADevice()(moved)) == rngposition(moved)

        terminal_position =
            moved.position isa PureRNGs._Position64 ?
            PureRNGs._terminal64(PureRNGs._max_block(rng)) : PureRNGs._terminal128()
        terminal = PureRNGs._rebuild(rng, terminal_position, rng.device)
        @test F(rngkey(terminal), rngposition(terminal)) === terminal
        @test_throws ArgumentError F(rngkey(rng), -1)
        @test_throws ArgumentError F(rngkey(rng), rngposition(terminal) + 1)
    end
end

@testset "Seeded constructors equal key constructors" begin
    for F in GENERATOR_TYPES
        seeded = F(0x5eed)
        @test F(rngkey(seeded)) === seeded
        @test F(0x5eed, 77) === F(rngkey(seeded), 77)
        @test PureRNGs._family_key(F, 0x5eed) === rngkey(seeded)
    end
    for (A, F) in (
        (Philox4x32R7, Philox4x32),
        (Threefry4x64R13, Threefry4x64),
        (ChaCha8, ChaCha),
        (ChaCha20, ChaCha),
    )
        @test A(0x5eed) === A(PureRNGs._family_key(F, 0x5eed))
        @test A(0x5eed, 9) === A(PureRNGs._family_key(F, 0x5eed), 9)
    end
end

@testset "generators print as the constructor call that rebuilds them" begin
    for F in (
            GENERATOR_TYPES...,
            Philox4x32R7,
            Philox2x64R6,
            Philox4x64R7,
            Threefry2x64R13,
            Threefry4x32R12,
            Threefry4x64R13,
            ChaCha8,
            ChaCha20,
        ),
        position in (0, 77)

        rng = F(0x5eed, position)
        @test eval(Meta.parse(repr(rng))) === rng
        @test eval(Meta.parse(repr(StatefulRNG(rng)))).rng === rng
    end
    @test repr(Philox4x32R7(1)) == "Philox4x32R7((0x00000001, 0x00000000), 0)"
    @test repr(ChaCha12(1)) == repr(ChaCha(1))
    @test repr(MLD.CUDADevice()(Philox2x32(1, 64))) ==
          "CUDADevice()(Philox2x32((0x00000001,), 64))"
end
