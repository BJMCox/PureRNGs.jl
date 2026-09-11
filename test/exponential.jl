using MLDataDevices
using InteractiveUtils: code_llvm
using Random: randexp, randexp!

const EXPONENTIAL_TYPES = (Float32, Float64)

_exponential_width(::Type{Float32}) = 24
_exponential_width(::Type{Float64}) = 53

function _reference_exponential_lattice(::Type{T}, raw::UInt64) where {T}
    scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
    u = T(raw) * scale
    return u, one(T) - u
end

function _scalar_exponential_chain(rng, ::Type{T}, count::Int) where {T}
    values = Vector{T}(undef, count)
    cursor = rng
    for index in eachindex(values)
        values[index] = randexp(cursor, T)
        cursor = IR._rebuild(
            cursor,
            _reference_position(cursor, _exponential_width(T)),
            cursor.device,
        )
    end
    return cursor, values
end

function _terminal_exponential_rng(F, ::Type{T}) where {T}
    rng = F(0x861)
    bit = UInt16(IR._block_bits(rng) - _exponential_width(T))
    position = if rng.position isa IR._Position64
        IR._Position64(IR._max_block(rng), bit)
    else
        IR._Position128(typemax(UInt64), typemax(UInt64), bit)
    end
    return IR._rebuild(rng, position, rng.device)
end

function _serial_exponential_fill_allocations(rng, destination)
    randexp_next!(rng, destination; threaded = false)
    return @allocated randexp_next!(rng, destination; threaded = false)
end

@testset "R13 and R63 exponential golden vectors" begin
    # Exponential draws read the uniform stream, so the 24-bit and 53-bit raws
    # are prefixes of the uniform golden block at this position. The result
    # bits pin the lattice and log transform on the CPU.
    expected = (
        (0xf39608, 0x1e72c115aa90e6, 0x4041afd7, 0x400835fb4ab5275f),
        (0x2029c8, 0x0405390f694417, 0x3e097b86, 0x3fc12f7102f376e4),
        (0x2a2d59, 0x0545ab2d4d03fa, 0x3e3859a9, 0x3fc70b355eb44b9b),
        (0x624cb2, 0x0c499645c6d44c, 0x3ef80dcf, 0x3fdf01b9f96445f5),
        (0xd4b0fe, 0x1a961fce3c14e2, 0x3fe36f06, 0x3ffc6de0e6e0cd33),
        (0x0d7f8e, 0x01aff1c9bd232c, 0x3d5ddfda, 0x3fabbbfbefc5db56),
        (0x44d393, 0x089a72618b95c9, 0x3ea0540d, 0x3fd40a81990acb40),
        (0x6dc19a, 0x0db833596e2ce7, 0x3f0f55c9, 0x3fe1eab9510e18f7),
    )

    for ((F, key), (raw32, raw64, cpu32, cpu64)) in zip(PACKED_GOLDEN_GENERATORS, expected)
        rng = _packed_golden_rng(F, key)
        block = _reference_position_block(rng.position)
        got32 =
            IR._extract_bits_unchecked(rng, block, rng.position.bit, Val(24))
        got64 =
            IR._extract_bits_unchecked(rng, block, rng.position.bit, Val(53))
        @test got32 === UInt64(raw32)
        @test got64 === UInt64(raw64)
        @test reinterpret(UInt32, randexp(rng, Float32)) === cpu32
        @test reinterpret(UInt64, randexp(rng, Float64)) === cpu64

        device_rng = MLDataDevices.CUDADevice()(rng)
        _, v32 = _reference_exponential_lattice(Float32, got32)
        _, v64 = _reference_exponential_lattice(Float64, got64)
        @test randexp(device_rng, Float32) === -Base.log(v32)
        @test randexp(device_rng, Float64) === -Base.log(v64)
    end
end

@testset "R63 exponential lattice and transform" begin
    cases = (
        (
            Float32,
            UInt32,
            (0x000000, 0x000001, 0x7fffff, 0xffffff),
            (0x80000000, 0x33800000, 0x3f317216, 0x41851592),
        ),
        (
            Float64,
            UInt64,
            (0x00000000000000, 0x00000000000001, 0x0fffffffffffff, 0x1fffffffffffff),
            (
                0x8000000000000000,
                0x3ca0000000000000,
                0x3fe62e42fefa39ed,
                0x40425e4f7b2737fa,
            ),
        ),
    )
    for (T, U, raw_values, expected_bits) in cases
        maximum = UInt64(last(raw_values))
        scale = T === Float32 ? Float32(0x1p-24) : Float64(0x1p-53)
        for (raw_value, expected) in zip(raw_values, expected_bits)
            raw = UInt64(raw_value)
            u, v = _reference_exponential_lattice(T, raw)
            @test IR._exponential_lattice(T, raw) === (u, v)
            @test reinterpret(U, IR._exponential_transform(IR._CPU_BACKEND, T, v)) ===
                  expected
            for token in (IR._CUDA_BACKEND, IR._AMDGPU_BACKEND, IR._METAL_BACKEND)
                @test IR._exponential_transform(token, T, v) === -Base.log(v)
            end
        end
        @test IR._exponential_lattice(T, UInt64(0)) === (zero(T), one(T))
        @test IR._exponential_lattice(T, maximum) === (one(T) - scale, scale)
    end
end

@testset "R29 exponential addressed boundaries" begin
    for F in GENERATOR_TYPES, T in EXPONENTIAL_TYPES
        last = _terminal_exponential_rng(F, T)
        position = last.position
        @test randexpat(last, T, 1) === randexp(last, T)
        @test last.position == position
        @test_throws ArgumentError randexpat(last, T, 0)
        @test_throws ArgumentError randexpat(last, T, -1)
        @test_throws ArgumentError randexpat(last, T, 2)
        @test last.position == position

        exhausted_position =
            position isa IR._Position64 ? IR._terminal64(IR._max_block(last)) :
            IR._terminal128()
        exhausted = IR._rebuild(last, exhausted_position, last.device)
        @test_throws ArgumentError randexpat(exhausted, T, 1)
        @test exhausted.position == exhausted_position
    end
end

@testset "R23, R29, R53, and R63 exponential scalars" begin
    for F in GENERATOR_TYPES, T in EXPONENTIAL_TYPES
        for bit in (UInt16(0), UInt16(24), UInt16(52), UInt16(63))
            rng = _positioned(F, 0x863, UInt64(9), bit)
            pure = randexp(rng, T)

            value, next_rng = randexp_next(rng, T)
            @test value === pure
            @test next_rng.position == _reference_position(rng, _exponential_width(T))
            @test randexpat(rng, T, 1) === pure
            third_rng = IR._rebuild(
                rng,
                _reference_position(rng, 2 * _exponential_width(T)),
                rng.device,
            )
            @test randexpat(rng, T, 3) === randexp(third_rng, T)
        end
    end

    rng = Philox4x32(0x864)
    @test randexp_next(rng) === randexp_next(rng, Float64)
    @test_throws ArgumentError randexp(rng)
end

@testset "R8, R26, R38, and R63 exponential stream" begin
    for F in GENERATOR_TYPES, T in EXPONENTIAL_TYPES
        rng = _positioned(F, 0x865, UInt64(4), UInt16(61))
        _, next_rng = randexp_next(rng, T)
        final_value, final_rng = rand_next(next_rng, UInt32)
        @test final_value === rand(next_rng, UInt32)
        @test final_rng.position ==
              _reference_position(rng, _exponential_width(T) + _uniform_width(UInt32))
    end

    rng = _packed_golden_rng(PACKED_GOLDEN_GENERATORS[1]...)
    device_rng = MLDataDevices.CUDADevice()(rng)
    @test rand(rng, Float32) === rand(device_rng, Float32)
    @test randn(rng, Float32) === randn(device_rng, Float32)
    @test _reference_exponential_lattice(
        Float32,
        _reference_extract(
            rng,
            _reference_position_block(rng.position),
            rng.position.bit,
            24,
        ),
    ) === _reference_exponential_lattice(
        Float32,
        _reference_extract(
            device_rng,
            _reference_position_block(device_rng.position),
            device_rng.position.bit,
            24,
        ),
    )
end

@testset "R23, R24, and R26 exponential arrays and fills" begin
    for F in GENERATOR_TYPES, T in EXPONENTIAL_TYPES
        for bit in (UInt16(0),)
            rng = _positioned(F, 0x866, UInt64(6), bit)
            next_rng, expected = _scalar_exponential_chain(rng, T, 17)

            serial = Vector{T}(undef, 17)
            threaded = similar(serial)
            @test randexp!(rng, serial; threaded = false) === serial
            @test randexp!(rng, threaded; threaded = true) === threaded
            sync_cpu()
            @test serial == threaded == expected
            @test rng.position.bit == bit

            continued, continued_next = randexp_next!(rng, similar(serial))
            sync_cpu()
            @test continued == expected
            @test continued_next.position == next_rng.position

            matrix = randexp(rng, T, 1, 17)
            sync_cpu()
            @test vec(matrix) == expected
            @test size(matrix) == (1, 17)

            allocated, allocated_next = randexp_next(rng, T, 17)
            sync_cpu()
            @test allocated == expected
            @test allocated_next.position == next_rng.position
        end

        rng = _positioned(F, 0x867, UInt64(3), UInt16(61))
        next_rng, expected = _scalar_exponential_chain(rng, T, 12)
        storage = fill(zero(T), 24)
        destination = @view storage[2:2:24]
        returned, view_next = randexp_next!(rng, destination; threaded = false)
        @test returned === destination
        @test collect(destination) == expected
        @test all(iszero, @view storage[1:2:23])
        @test view_next.position == next_rng.position
    end

    for T in EXPONENTIAL_TYPES
        bit = UInt16(127)
        rng = _positioned(Philox4x32, 0x866, UInt64(6), bit)
        expected_rng, expected = _scalar_exponential_chain(rng, T, 17)
        destination, next_rng = randexp_next!(rng, Vector{T}(undef, 17); threaded = false)
        @test destination == expected
        @test next_rng.position == expected_rng.position
    end

    rng = Philox4x32(0x868)
    default_values, default_next = randexp_next(rng, 2, 3)
    typed_values, typed_next = randexp_next(rng, Float64, 2, 3)
    sync_cpu()
    @test default_values == typed_values
    @test default_next.position == typed_next.position
    @test size(default_values) == (2, 3)
    @test_throws ArgumentError randexp(rng, Float32, -1)
    @test_throws ArgumentError randexp_next(rng, -1)

    caller = current_task()
    probe = TaskWriteProbe(Vector{Float64}(undef, 37))
    returned, next_rng = randexp_next!(rng, probe; threaded = false)
    expected_rng, expected = _scalar_exponential_chain(rng, Float64, 37)
    @test returned === probe
    @test all(task -> task === caller, probe.writers)
    @test probe.data == expected
    @test next_rng.position == expected_rng.position
end

@testset "R26 small allocating exponential boundary" begin
    rng = _positioned(Philox4x32, 0x86a1, UInt64(5), UInt16(61))
    for count in (128, 129)
        expected_next, expected = _scalar_exponential_chain(rng, Float64, count)
        values, next_rng = randexp_next(rng, Float64, count)
        sync_cpu()
        @test values == expected
        @test next_rng.position == expected_next.position
    end
end

@testset "R30, R39, R40, and R54 exponential fill validation" begin
    rng = Philox4x32(0x869)
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    empty = Float32[]
    @test randexp!(exhausted, empty; threaded = false) === empty
    empty_result, empty_next = randexp_next!(exhausted, empty; threaded = false)
    @test empty_result === empty
    @test empty_next === exhausted
    @test isempty(randexp(exhausted, Float32, 0))
    allocated_empty, allocated_empty_next = randexp_next(exhausted, Float32, 0)
    @test isempty(allocated_empty)
    @test allocated_empty_next === exhausted
    default_empty, default_empty_next = randexp_next(exhausted, 0)
    @test isempty(default_empty)
    @test eltype(default_empty) === Float64
    @test default_empty_next === exhausted

    for F in GENERATOR_TYPES, T in EXPONENTIAL_TYPES
        last = _terminal_exponential_rng(F, T)
        destination = fill(one(T), 2)
        before = copy(destination)
        @test_throws ArgumentError randexp!(last, destination; threaded = false)
        @test destination == before
        @test_throws ArgumentError randexp_next!(last, destination; threaded = false)
        @test destination == before

        final = Vector{T}(undef, 1)
        _, final_next = randexp_next!(last, final; threaded = false)
        @test final[1] === randexp(last, T)
        expected_terminal =
            last.position isa IR._Position64 ? IR._terminal64(IR._max_block(last)) :
            IR._terminal128()
        @test final_next.position == expected_terminal

        insufficient_position = if last.position isa IR._Position64
            IR._Position64(IR._max_block(last), last.position.bit + UInt16(1))
        else
            IR._Position128(typemax(UInt64), typemax(UInt64), last.position.bit + UInt16(1))
        end
        insufficient = IR._rebuild(last, insufficient_position, last.device)
        @test_throws ArgumentError randexp(insufficient, T, 1)
        @test_throws ArgumentError randexp_next(insufficient, T, 1)
    end

end

@testset "R30 exponential fixed-work and codegen" begin
    for F in GENERATOR_TYPES, T in EXPONENTIAL_TYPES
        rng = F(0x86a)
        destination = Vector{T}(undef, 7)
        @test _serial_exponential_fill_allocations(rng, destination) == 0
    end

    rng = IR.MLDataDevices.CUDADevice()(Philox4x64(0x86b))
    destination = Vector{Float64}(undef, 7)
    for (function_, call_signature) in
        ((randexp_next!, Tuple{typeof(rng),typeof(destination)}),)
        typed_ir = sprint(show, code_typed(function_, call_signature; optimize = true))
        llvm_ir = sprint() do io
            code_llvm(
                io,
                function_,
                call_signature;
                raw = false,
                dump_module = false,
                optimize = true,
            )
        end
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        @test !occursin(r"\bi128\b", llvm_ir)
    end

end
