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
    expected = (
        (0xa05803, 0x140b0076b1f96e, 0x3f7c02be, 0x3fef805814968c68),
        (0xda96ce, 0x1b52d9c94f7a5f, 0x3ff62be7, 0x3ffec57d08964390),
        (0xabe90b, 0x157d216f8ed58d, 0x3f8e8068, 0x3ff1d00d16e33a3a),
        (0x6e4523, 0x0dc8a4782bb60e, 0x3f103c72, 0x3fe2078e5d780b86),
        (0x98edd2, 0x131dba42705949, 0x3f68e5fb, 0x3fed1cbf6b7dc906),
        (0xc12127, 0x182424ea242c91, 0x3fb3b990, 0x3ff67732129fcac2),
        (0x15379a, 0x02a6f350819147, 0x3db12f9c, 0x3fb625f4028b615e),
        (0x0cb66c, 0x0196cd85660493, 0x3d50a017, 0x3faa14033fc04807),
    )

    for ((F, key), (raw32, raw64, cpu32, cpu64)) in zip(PACKED_GOLDEN_FAMILIES, expected)
        rng = _packed_golden_rng(F, key)
        block = _reference_position_block(rng.position)
        got32 =
            IR._extract_bits_unchecked(rng, IR.FAMILY_EXP, block, rng.position.bit, Val(24))
        got64 =
            IR._extract_bits_unchecked(rng, IR.FAMILY_EXP, block, rng.position.bit, Val(53))
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
    for F in FAMILY_TYPES, T in EXPONENTIAL_TYPES
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
    for F in FAMILY_TYPES, T in EXPONENTIAL_TYPES
        for bit in (UInt16(0), UInt16(24), UInt16(52), UInt16(63))
            rng = _positioned(F, 0x863, UInt64(9), bit)
            pure = randexp(rng, T)

            next_rng, value = randexp_next(rng, T)
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
    @test sprint(showerror, try
        randexp(rng)
    catch error
        error
    end) == "ArgumentError: untyped immutable draws are forbidden; use randexp(rng, T)"
end

@testset "R8, R26, R38, and R63 exponential stream" begin
    for F in FAMILY_TYPES, T in EXPONENTIAL_TYPES
        rng = _positioned(F, 0x865, UInt64(4), UInt16(61))
        exponential_raw = _reference_extract(
            rng,
            IR.FAMILY_EXP,
            _reference_position_block(rng.position),
            rng.position.bit,
            _exponential_width(T),
        )
        uniform_raw = _reference_extract(
            rng,
            IR.FAMILY_BITS,
            _reference_position_block(rng.position),
            rng.position.bit,
            _exponential_width(T),
        )
        @test exponential_raw != uniform_raw

        next_rng, _ = randexp_next(rng, T)
        final_rng, final_value = rand_next(next_rng, UInt32)
        @test final_value === rand(next_rng, UInt32)
        @test final_rng.position ==
              _reference_position(rng, _exponential_width(T) + _uniform_width(UInt32))
    end

    rng = _packed_golden_rng(PACKED_GOLDEN_FAMILIES[1]...)
    device_rng = MLDataDevices.CUDADevice()(rng)
    @test rand(rng, Float32) === rand(device_rng, Float32)
    @test randn(rng, Float32) === randn(device_rng, Float32)
    @test _reference_exponential_lattice(
        Float32,
        _reference_extract(
            rng,
            IR.FAMILY_EXP,
            _reference_position_block(rng.position),
            rng.position.bit,
            24,
        ),
    ) === _reference_exponential_lattice(
        Float32,
        _reference_extract(
            device_rng,
            IR.FAMILY_EXP,
            _reference_position_block(device_rng.position),
            device_rng.position.bit,
            24,
        ),
    )
    @test randexp(rng, Float32) !== randexp(device_rng, Float32)
end

@testset "R23, R24, and R26 exponential arrays and fills" begin
    for F in FAMILY_TYPES, T in EXPONENTIAL_TYPES
        block_bits = IR._block_bits(F(0x866))
        for bit in (UInt16(0), UInt16(24), UInt16(53), UInt16(block_bits - 1))
            rng = _positioned(F, 0x866, UInt64(6), bit)
            next_rng, expected = _scalar_exponential_chain(rng, T, 17)

            serial = Vector{T}(undef, 17)
            threaded = similar(serial)
            @test randexp!(rng, serial; threaded = false) === serial
            @test randexp!(rng, threaded; threaded = true) === threaded
            sync_cpu()
            @test serial == threaded == expected
            @test rng.position.bit == bit

            continued_next, continued = randexp_next!(rng, similar(serial))
            sync_cpu()
            @test continued == expected
            @test continued_next.position == next_rng.position

            matrix = randexp(rng, T, 1, 17)
            sync_cpu()
            @test vec(matrix) == expected
            @test size(matrix) == (1, 17)

            allocated_next, allocated = randexp_next(rng, T, 17)
            sync_cpu()
            @test allocated == expected
            @test allocated_next.position == next_rng.position
        end

        rng = _positioned(F, 0x867, UInt64(3), UInt16(61))
        next_rng, expected = _scalar_exponential_chain(rng, T, 12)
        storage = fill(zero(T), 24)
        destination = @view storage[2:2:24]
        view_next, returned = randexp_next!(rng, destination; threaded = false)
        @test returned === destination
        @test collect(destination) == expected
        @test all(iszero, @view storage[1:2:23])
        @test view_next.position == next_rng.position
    end

    rng = Philox4x32(0x868)
    default_next, default_values = randexp_next(rng, 2, 3)
    typed_next, typed_values = randexp_next(rng, Float64, 2, 3)
    sync_cpu()
    @test default_values == typed_values
    @test default_next.position == typed_next.position
    @test size(default_values) == (2, 3)
    @test_throws ArgumentError randexp(rng, Float32, -1)
    @test_throws ArgumentError randexp_next(rng, -1)

    caller = current_task()
    probe = TaskWriteProbe(Vector{Float64}(undef, 37))
    next_rng, returned = randexp_next!(rng, probe; threaded = false)
    expected_rng, expected = _scalar_exponential_chain(rng, Float64, 37)
    @test returned === probe
    @test all(task -> task === caller, probe.writers)
    @test probe.data == expected
    @test next_rng.position == expected_rng.position
end

@testset "R30, R39, R40, and R54 exponential fill validation" begin
    rng = Philox4x32(0x869)
    wrong = WrongDeviceArray(Float32[])
    @test_throws ArgumentError randexp!(rng, wrong)
    @test_throws ArgumentError randexp_next!(rng, wrong)
    @test_throws TypeError randexp!(rng, wrong; threaded = 1)

    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    empty = Float32[]
    @test randexp!(exhausted, empty; threaded = false) === empty
    empty_next, empty_result = randexp_next!(exhausted, empty; threaded = false)
    @test empty_result === empty
    @test empty_next === exhausted
    @test isempty(randexp(exhausted, Float32, 0))
    allocated_empty_next, allocated_empty = randexp_next(exhausted, Float32, 0)
    @test isempty(allocated_empty)
    @test allocated_empty_next === exhausted
    default_empty_next, default_empty = randexp_next(exhausted, 0)
    @test isempty(default_empty)
    @test eltype(default_empty) === Float64
    @test default_empty_next === exhausted

    for F in FAMILY_TYPES, T in EXPONENTIAL_TYPES
        last = _terminal_exponential_rng(F, T)
        destination = fill(one(T), 2)
        before = copy(destination)
        @test_throws ArgumentError randexp!(last, destination; threaded = false)
        @test destination == before
        @test_throws ArgumentError randexp_next!(last, destination; threaded = false)
        @test destination == before

        final = Vector{T}(undef, 1)
        final_next, _ = randexp_next!(last, final; threaded = false)
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

    lookups = Ref(0)
    probe = BackendProbe(Vector{Float64}(undef, 17), lookups)
    randexp!(rng, probe; threaded = false)
    @test lookups[] == 0
    randexp!(rng, probe; threaded = true)
    sync_cpu()
    @test lookups[] == 1
end

@testset "R30 exponential inference, allocation, and IR" begin
    for F in FAMILY_TYPES, T in EXPONENTIAL_TYPES
        rng = F(0x86a)
        destination = Vector{T}(undef, 7)
        @test @inferred(randexp!(rng, destination; threaded = false)) === destination
        @test @inferred(randexp_next!(rng, destination; threaded = false)) isa
              Tuple{typeof(rng),typeof(destination)}
        @test @inferred(randexp(rng, T, 2, 3)) isa Matrix{T}
        @test @inferred(randexp_next(rng, T, 2, 3)) isa Tuple{typeof(rng),Matrix{T}}
        @test _serial_exponential_fill_allocations(rng, destination) == 0
    end

    rng = Philox4x64(0x86b)
    destination = Vector{Float64}(undef, 7)
    signature = Tuple{
        typeof(rng),
        typeof(rng.position),
        typeof(destination),
        Type{Float64},
        Base.OneTo{Int},
        typeof(rng.device),
    }
    for (function_, call_signature) in (
        (randexp_next!, Tuple{typeof(rng),typeof(destination)}),
        (IR._fill_transformed_dense_cpu!, signature),
    )
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

    @test !applicable(randexp!, rng, Vector{Float16}(undef, 1))
    @test !applicable(randexp_next!, rng, Vector{UInt64}(undef, 1))
    @test_throws TypeError randexp!(rng, destination; threaded = 1)
    @test_throws MethodError randexp!(rng, destination; serial = false)
end
