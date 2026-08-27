using InteractiveUtils: code_llvm
using Random: rand!

const IR = PureRNGs
const KA = PureRNGs.KernelAbstractions
const MLD = PureRNGs.MLDataDevices

const SCALAR_UNIFORM_TYPES = (Bool, UInt32, UInt64, Float32, Float64)
const PURE_UNIFORM_TYPES = (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)

const PACKED_GOLDEN_BLOCK = UInt64(0x00123456789abcde)
const PACKED_GOLDEN_BIT = UInt16(61)
const PACKED_GOLDEN_FAMILIES = (
    (Philox2x32, (UInt32(0x01234567),)),
    (Philox4x32, (UInt32(0x01234567), UInt32(0x89abcdef))),
    (Philox2x64, (UInt64(0x0000000001234567),)),
    (Philox4x64, (UInt64(0x0000000001234567), UInt64(0x0000000089abcdef))),
    (Threefry2x32, (UInt32(0x01234567), UInt32(0x89abcdef))),
    (
        Threefry4x32,
        (UInt32(0x01234567), UInt32(0x89abcdef), UInt32(0xfedcba98), UInt32(0x76543210)),
    ),
    (Threefry2x64, (UInt64(0x0000000001234567), UInt64(0x0000000089abcdef))),
    (
        Threefry4x64,
        (
            UInt64(0x0000000001234567),
            UInt64(0x0000000089abcdef),
            UInt64(0x00000000fedcba98),
            UInt64(0x0000000076543210),
        ),
    ),
)

function _packed_golden_rng(F, key)
    base = F(key)
    position =
        base.position isa IR._Position64 ?
        IR._Position64(PACKED_GOLDEN_BLOCK, PACKED_GOLDEN_BIT) :
        IR._Position128(PACKED_GOLDEN_BLOCK, UInt64(0), PACKED_GOLDEN_BIT)
    return IR._rebuild(base, position, base.device)
end

_uniform_width(::Type{Bool}) = 1
_uniform_width(::Type{Float32}) = 24
_uniform_width(::Type{UInt32}) = 32
_uniform_width(::Type{Int32}) = 32
_uniform_width(::Type{Float64}) = 53
_uniform_width(::Type{UInt64}) = 64
_uniform_width(::Type{Int64}) = 64

mutable struct BackendProbe{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    lookups::Base.RefValue{Int}
end

Base.size(array::BackendProbe) = size(array.data)
Base.axes(array::BackendProbe) = axes(array.data)
Base.IndexStyle(::Type{<:BackendProbe{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.getindex(array::BackendProbe, indices...) = getindex(array.data, indices...)
Base.setindex!(array::BackendProbe, value, indices...) =
    setindex!(array.data, value, indices...)
MLD.get_device(array::BackendProbe) = MLD.get_device(array.data)
function KA.get_backend(array::BackendProbe)
    array.lookups[] += 1
    return KA.get_backend(array.data)
end

mutable struct TaskWriteProbe{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    writers::Array{Task,N}
end

TaskWriteProbe(data::AbstractArray{T,N}) where {T,N} =
    TaskWriteProbe(data, Array{Task}(undef, size(data)))
Base.size(array::TaskWriteProbe) = size(array.data)
Base.axes(array::TaskWriteProbe) = axes(array.data)
Base.IndexStyle(::Type{<:TaskWriteProbe{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.getindex(array::TaskWriteProbe, indices...) = getindex(array.data, indices...)
function Base.setindex!(array::TaskWriteProbe, value, indices...)
    array.writers[indices...] = current_task()
    return setindex!(array.data, value, indices...)
end
MLD.get_device(array::TaskWriteProbe) = MLD.get_device(array.data)
KA.get_backend(array::TaskWriteProbe) = KA.get_backend(array.data)

struct WrongDeviceArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end

Base.size(array::WrongDeviceArray) = size(array.data)
Base.axes(array::WrongDeviceArray) = axes(array.data)
Base.IndexStyle(::Type{<:WrongDeviceArray{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.getindex(array::WrongDeviceArray, indices...) = getindex(array.data, indices...)
Base.setindex!(array::WrongDeviceArray, value, indices...) =
    setindex!(array.data, value, indices...)
MLD.get_device_type(::WrongDeviceArray) = MLD.UnknownDevice

_reference_position_block(position::IR._Position64) = position.block
_reference_position_block(position::IR._Position128) = (position.lo, position.hi)

_reference_convert(::Type{Bool}, value::UInt64) = isone(value)
_reference_convert(::Type{UInt32}, value::UInt64) = value % UInt32
_reference_convert(::Type{Int32}, value::UInt64) = reinterpret(Int32, value % UInt32)
_reference_convert(::Type{UInt64}, value::UInt64) = value
_reference_convert(::Type{Int64}, value::UInt64) = reinterpret(Int64, value)
_reference_convert(::Type{Float32}, value::UInt64) =
    Float32(value % UInt32) * Float32(0x1p-24)
_reference_convert(::Type{Float64}, value::UInt64) = Float64(value) * 0x1p-53

function _reference_uniform(rng, ::Type{T}) where {T}
    raw = _reference_extract(
        rng,
        IR.FAMILY_BITS,
        _reference_position_block(rng.position),
        rng.position.bit,
        _uniform_width(T),
    )
    return _reference_convert(T, raw)
end

function _reference_position(rng, additional_bits::Integer)
    block_bits = BigInt(IR._block_bits(rng))
    position = rng.position
    block =
        position isa IR._Position64 ? BigInt(position.block) :
        (BigInt(position.hi) << 64) + BigInt(position.lo)
    total = BigInt(position.bit) + BigInt(additional_bits)
    block_delta, bit = divrem(total, block_bits)
    block += block_delta
    if position isa IR._Position64
        return IR._Position64(UInt64(block), UInt16(bit))
    end
    return IR._Position128(
        UInt64(block & typemax(UInt64)),
        UInt64(block >> 64),
        UInt16(bit),
    )
end

function _reference_chain(rng, ::Type{T}, count::Int) where {T}
    values = Vector{T}(undef, count)
    cursor = rng
    for index in eachindex(values)
        values[index] = _reference_uniform(cursor, T)
        cursor = IR._rebuild(
            cursor,
            _reference_position(cursor, _uniform_width(T)),
            cursor.device,
        )
    end
    return cursor, values
end

function _positioned(F, seed, block::UInt64, bit::UInt16)
    base = F(seed)
    position =
        base.position isa IR._Position64 ? IR._Position64(block, bit) :
        IR._Position128(block, UInt64(7), bit)
    return IR._rebuild(base, position, base.device)
end

function _serial_fill_allocations(rng, destination)
    rand_next!(rng, destination; threaded = false)
    return @allocated rand_next!(rng, destination; threaded = false)
end

sync_cpu() = KA.synchronize(KA.CPU())
_padding_is_zero(bits::BitArray) =
    isempty(bits) ||
    iszero(length(bits) & 63) ||
    iszero(bits.chunks[end] >> (length(bits) & 63))

mutable struct CountingDenseStream{N}
    calls::Base.RefValue{Int}
end

function IR._stream_limbs(stream::CountingDenseStream{N}, ::UInt32, block::UInt64) where {N}
    stream.calls[] += 1
    return ntuple(index -> block + UInt64(index), Val(N))
end

function _reference_counting_extract(block::UInt64, bit::Int, width::Int, limbs::Int)
    value = UInt64(0)
    for _ = 1:width
        block_delta, offset = divrem(bit, 64limbs)
        lane, word_bit = divrem(offset, 64)
        word = block + UInt64(block_delta + lane + 1)
        value = (value << 1) | ((word >> (63 - word_bit)) & UInt64(1))
        bit += 1
    end
    return value
end

@testset "R13 packed uniform golden vectors" begin
    # Independent C++17 oracle using DEShawResearch/random123 v1.14.0:
    # commit 726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13;
    # include/Random123/philox.h SHA-256
    # 6c2ef219a855885499a73b338d5f41dafe079618b2dae2f60ea86ee785d771e2;
    # include/Random123/threefry.h SHA-256
    # 4c210b32b5ba605b059c54d5edd6f01bf04190de49a0abeecec76420cd072a72.
    # Revision-13 layouts and MSB-first extraction were transcribed directly;
    # the oracle never imports package code. Oracle source SHA-256
    # 68a432f195091fff370c858b5ef80617ec3bf3399ff9db8bfa4cc06ca753f5e3;
    # output SHA-256
    # 75cb899f9daf671f3b59be5c7602a96eb969f10188ed71c8d011da484c8069fe.
    # Pinned testbed commit for unchanged cores and layouts:
    # 7a6d2cfe06c610e8437b4d0ac99a5ef208a3464d.
    expected = (
        (true, 0xf39608ad, 0xf39608ad5487335f, 0x3f739608, 0x3fee72c115aa90e6),
        (false, 0x2029c87b, 0x2029c87b4a20bd10, 0x3e00a720, 0x3fc014e43da5105c),
        (false, 0x2a2d596a, 0x2a2d596a681fd002, 0x3e28b564, 0x3fc516acb5340fe8),
        (false, 0x624cb22e, 0x624cb22e36a263ea, 0x3ec49964, 0x3fd8932c8b8da898),
        (true, 0xd4b0fe71, 0xd4b0fe71e0a7135a, 0x3f54b0fe, 0x3fea961fce3c14e2),
        (false, 0x0d7f8e4d, 0x0d7f8e4de91962b4, 0x3d57f8e0, 0x3faaff1c9bd232c0),
        (false, 0x44d3930c, 0x44d3930c5cae4976, 0x3e89a726, 0x3fd134e4c3172b92),
        (false, 0x6dc19acb, 0x6dc19acb71673eb8, 0x3edb8334, 0x3fdb7066b2dc59ce),
    )

    for ((F, key), (bit, word32, word64, float32_bits, float64_bits)) in
        zip(PACKED_GOLDEN_FAMILIES, expected)
        rng = _packed_golden_rng(F, key)
        @test rand(rng, Bool) === bit
        @test rand(rng, UInt32) === word32
        @test rand(rng, Int32) === reinterpret(Int32, word32)
        @test rand(rng, UInt64) === word64
        @test rand(rng, Int64) === reinterpret(Int64, word64)
        @test reinterpret(UInt32, rand(rng, Float32)) === float32_bits
        @test reinterpret(UInt64, rand(rng, Float64)) === float64_bits
    end
end

@testset "R25 and R53 packed primitive widths" begin
    for (T, width) in zip(PURE_UNIFORM_TYPES, (1, 32, 32, 64, 64, 24, 53))
        @test IR._draw_bits(T) === UInt16(width)
    end

    for F in FAMILY_TYPES, T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x521, UInt64(9), UInt16(61))
        position = rng.position
        expected = _reference_uniform(rng, T)
        @test rand(rng, T) === expected
        @test rng.position === position

        next_rng, value = rand_next(rng, T)
        @test value === expected
        @test next_rng.position == _reference_position(rng, _uniform_width(T))
    end
end

@testset "R25 signed primitive bitcasts" begin
    for F in FAMILY_TYPES, (S, U) in ((Int32, UInt32), (Int64, UInt64))
        rng = _positioned(F, 0x5211, UInt64(9), UInt16(61))
        @test reinterpret(U, rand(rng, S)) === rand(rng, U)

        signed_next, signed_value = rand_next(rng, S)
        unsigned_next, unsigned_value = rand_next(rng, U)
        @test reinterpret(U, signed_value) === unsigned_value
        @test signed_next.position == unsigned_next.position

        @test reinterpret(U, randat(rng, S, 7)) === randat(rng, U, 7)

        signed_fill = Vector{S}(undef, 129)
        unsigned_fill = Vector{U}(undef, 129)
        signed_fill_next, _ = rand_next!(rng, signed_fill; threaded = false)
        unsigned_fill_next, _ = rand_next!(rng, unsigned_fill; threaded = false)
        @test reinterpret(U, signed_fill) == unsigned_fill
        @test signed_fill_next.position == unsigned_fill_next.position
    end
end

@testset "R24, R26, and R53 mixed packed continuation" begin
    trace = (Bool, Float64, Float32, UInt64, UInt32, Bool, Float64)
    for F in FAMILY_TYPES
        rng = _positioned(F, 0x522, UInt64(11), UInt16(63))
        cursor = rng
        for T in trace
            expected = _reference_uniform(cursor, T)
            expected_position = _reference_position(cursor, _uniform_width(T))
            cursor, value = rand_next(cursor, T)
            @test value === expected
            @test cursor.position == expected_position
        end
    end
end

@testset "R29 packed addressed draws" begin
    for F in FAMILY_TYPES, T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x523, UInt64(5), UInt16(47))
        cursor = rng
        for i = 1:9
            expected = _reference_uniform(cursor, T)
            @test randat(rng, T, i) === expected
            cursor = IR._rebuild(
                cursor,
                _reference_position(cursor, _uniform_width(T)),
                cursor.device,
            )
        end
        @test rng.position.bit === UInt16(47)
        @test_throws ArgumentError randat(rng, T, 0)
        @test_throws ArgumentError randat(rng, T, -1)
    end

    for F in (Philox2x64, Threefry2x64)
        rng = F(0x523)
        final_index = big(1) << 65
        last = IR._rebuild(
            rng,
            IR._Position64(typemax(UInt64), IR._block_bits(rng) - UInt16(64)),
            rng.device,
        )
        @test randat(rng, UInt64, final_index) === rand(last, UInt64)
        @test_throws ArgumentError randat(rng, UInt64, final_index + 1)
    end

    for F in (Philox4x64, Threefry4x64)
        rng = F(0x523)
        final_index = big(1) << 130
        last = IR._rebuild(
            rng,
            IR._Position128(
                typemax(UInt64),
                typemax(UInt64),
                IR._block_bits(rng) - UInt16(64),
            ),
            rng.device,
        )
        @test randat(rng, UInt64, final_index) === rand(last, UInt64)
        @test_throws ArgumentError randat(rng, UInt64, final_index + 1)

        near_end = IR._rebuild(
            rng,
            IR._Position128(typemax(UInt64), typemax(UInt64), UInt16(0)),
            rng.device,
        )
        @test randat(near_end, UInt64, UInt64(4)) === rand(
            IR._rebuild(
                near_end,
                IR._Position128(typemax(UInt64), typemax(UInt64), UInt16(192)),
                rng.device,
            ),
            UInt64,
        )
        @test_throws ArgumentError randat(near_end, UInt64, UInt64(5))
    end
end

@testset "R26 packed CPU fills, shapes, views, and BitArray" begin
    for F in FAMILY_TYPES, T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x524, UInt64(7), UInt16(61))
        expected_rng, expected = _reference_chain(rng, T, 12)

        destination = Vector{T}(undef, 12)
        @test rand!(rng, destination) === destination
        sync_cpu()
        @test destination == expected
        @test rng.position.bit === UInt16(61)

        replay = similar(destination)
        next_rng, returned = rand_next!(rng, replay)
        sync_cpu()
        @test returned === replay
        @test replay == expected
        @test next_rng.position == expected_rng.position

        matrix = Matrix{T}(undef, 3, 4)
        matrix_next, matrix_result = rand_next!(rng, matrix; threaded = false)
        @test matrix_result === matrix
        @test vec(matrix) == expected
        @test matrix_next.position == expected_rng.position

        storage = fill(zero(T), 24)
        view_destination = @view storage[2:2:24]
        @test rand!(rng, view_destination; threaded = false) === view_destination
        @test collect(view_destination) == expected
        @test all(iszero, @view storage[1:2:23])

        threaded_storage = fill(zero(T), 24)
        threaded_view = @view threaded_storage[2:2:24]
        @test rand!(rng, threaded_view; threaded = true) === threaded_view
        sync_cpu()
        @test collect(threaded_view) == expected
        @test all(iszero, @view threaded_storage[1:2:23])
    end

    for F in FAMILY_TYPES, count in (0, 1, 7, 65, 67)
        rng = _positioned(F, 0x525, UInt64(4), UInt16(63))
        expected_rng, expected = _reference_chain(rng, Bool, count)
        ordinary = BitArray(undef, count)
        explicit = similar(ordinary)
        serial = similar(ordinary)
        ordinary_next, ordinary_result = rand_next!(rng, ordinary)
        explicit_next, explicit_result = rand_next!(rng, explicit; threaded = true)
        serial_next, serial_result = rand_next!(rng, serial; threaded = false)
        sync_cpu()
        @test ordinary_result === ordinary
        @test explicit_result === explicit
        @test serial_result === serial
        @test ordinary == explicit == serial == expected
        @test ordinary_next.position ==
              explicit_next.position ==
              serial_next.position ==
              expected_rng.position
    end

    for F in FAMILY_TYPES
        rng = _positioned(F, 0x525, UInt64(4), UInt16(63))
        expected_rng, expected = _reference_chain(rng, Bool, 12)
        destination = BitArray(undef, 3, 4)
        next_rng, result = rand_next!(rng, destination; threaded = false)
        @test result === destination
        @test vec(destination) == expected
        @test next_rng.position == expected_rng.position
    end
end

@testset "R26 dense codec phases, tails, and cached blocks" begin
    for F in FAMILY_TYPES, T in PURE_UNIFORM_TYPES
        group = IR._dense_fill_group(T)
        counts = unique((0, max(0, group - 1), group, group + 1, 2group + 3))
        block_bits = IR._block_bits(F(0x5250))
        for bit in (UInt16(0), UInt16(1), UInt16(31), UInt16(63), UInt16(block_bits - 1)),
            count in counts

            rng = _positioned(F, 0x5250, UInt64(6), bit)
            expected_rng, expected = _reference_chain(rng, T, count)
            destination = Vector{T}(undef, count)
            IR._fill_uniform_dense_cpu!(
                rng,
                rng.position,
                destination,
                T,
                eachindex(destination),
            )
            @test destination == expected
            @test expected_rng.position ==
                  _reference_position(rng, count * _uniform_width(T))
        end
    end

    for F in (Philox4x64, Threefry4x64), T in PURE_UNIFORM_TYPES
        base = F(0x5250)
        rng = IR._rebuild(
            base,
            IR._Position128(typemax(UInt64), UInt64(7), IR._block_bits(base) - 1),
            base.device,
        )
        count = IR._dense_fill_group(T) + 1
        _, expected = _reference_chain(rng, T, count)
        destination = Vector{T}(undef, count)
        IR._fill_uniform_dense_cpu!(
            rng,
            rng.position,
            destination,
            T,
            eachindex(destination),
        )
        @test destination == expected
    end

    for F in FAMILY_TYPES, count in (1, 63, 64, 65, 129)
        rng = _positioned(F, 0x5250, UInt64(8), UInt16(47))
        _, expected = _reference_chain(rng, Bool, count)
        destination = BitArray(undef, count)
        IR._fill_uniform_dense_cpu!(
            rng,
            rng.position,
            destination,
            Bool,
            eachindex(destination),
        )
        @test destination == expected
        @test _padding_is_zero(destination)
    end

    for (limbs, expected_calls) in ((1, 3), (2, 2), (4, 1))
        calls = Ref(0)
        stream = CountingDenseStream{limbs}(calls)
        cursor = IR._dense_cursor(stream, IR.FAMILY_BITS, UInt64(9), UInt16(13))
        @test isbitstype(typeof(cursor))
        values = UInt64[]
        for width in (Val(53), Val(24), Val(64), Val(32))
            value, cursor =
                IR._take_dense_bits_unchecked(stream, IR.FAMILY_BITS, cursor, width)
            push!(values, value)
        end
        @test calls[] == expected_calls
        @test values == [
            _reference_counting_extract(UInt64(9), 13, 53, limbs),
            _reference_counting_extract(UInt64(9), 66, 24, limbs),
            _reference_counting_extract(UInt64(9), 90, 64, limbs),
            _reference_counting_extract(UInt64(9), 154, 32, limbs),
        ]
    end
end

@testset "R26 parallel packed fills cross CPU chunks" begin
    for F in (Philox2x32, Philox4x32, Philox4x64), T in PURE_UNIFORM_TYPES
        rng = _positioned(F, 0x5251, UInt64(4), UInt16(61))
        chunk_elements = IR._dense_fill_chunk_elements(T)
        count = 4chunk_elements + 3
        serial = Vector{T}(undef, count)
        threaded = similar(serial)

        serial_next, serial_result = rand_next!(rng, serial; threaded = false)
        threaded_next, threaded_result = rand_next!(rng, threaded; threaded = true)
        sync_cpu()

        expected_position = _reference_position(rng, count * _uniform_width(T))
        @test serial_result === serial
        @test threaded_result === threaded
        @test threaded == serial
        @test serial_next.position == threaded_next.position == expected_position

        for index in (
            1,
            chunk_elements,
            chunk_elements + 1,
            2chunk_elements + 1,
            3chunk_elements + 1,
            4chunk_elements + 1,
            count,
        )
            position = _reference_position(rng, (index - 1) * _uniform_width(T))
            cursor = IR._rebuild(rng, position, rng.device)
            @test threaded[index] === _reference_uniform(cursor, T)
        end
    end
end

@testset "R26 dense CPU chunk seams" begin
    for T in PURE_UNIFORM_TYPES, delta in (-1, 0, 1)
        rng = _positioned(Philox4x32, 0x5252, UInt64(4), UInt16(61))
        chunk_elements = IR._dense_fill_chunk_elements(T)
        count = 4chunk_elements + delta
        serial = Vector{T}(undef, count)
        threaded = similar(serial)
        serial_next, _ = rand_next!(rng, serial; threaded = false)
        threaded_next, _ = rand_next!(rng, threaded; threaded = true)
        sync_cpu()
        @test threaded == serial
        @test threaded_next.position == serial_next.position
    end

    chunk_elements = IR._dense_fill_chunk_elements(Bool)
    for delta in (-1, 0, 1)
        rng = _positioned(Philox4x32, 0x5253, UInt64(4), UInt16(63))
        count = 4chunk_elements + delta
        serial = BitArray(undef, count)
        threaded = similar(serial)
        rand_next!(rng, serial; threaded = false)
        rand_next!(rng, threaded; threaded = true)
        sync_cpu()
        @test threaded == serial
        @test _padding_is_zero(threaded)
    end
end

@testset "Philox4x32 four-block dense fills" begin
    rng = Philox4x32(0x5254)
    for (T, count, expected_index, expected_block) in (
        (Bool, 511, 1, 0),
        (Bool, 512, 513, 4),
        (UInt32, 15, 1, 0),
        (UInt32, 16, 17, 4),
        (UInt32, 64, 65, 16),
        (UInt64, 8, 9, 4),
        (Float32, 63, 1, 0),
        (Float32, 64, 65, 12),
    )
        destination = Vector{T}(undef, count)
        index, block = IR._fill_aligned_blocks4!(rng, destination, 1, count, UInt64(0))
        @test (index, block) == (expected_index, UInt64(expected_block))
    end

    terminal_block = IR._max_block(rng) - UInt64(2)
    terminal_destination = Vector{UInt32}(undef, 16)
    @test IR._fill_aligned_blocks4!(rng, terminal_destination, 1, 16, terminal_block) ==
          (1, terminal_block)

    signature =
        Tuple{typeof(rng),typeof(rng.position),Vector{UInt32},Type{UInt32},Base.OneTo{Int}}
    lowered = sprint(show, only(code_lowered(IR._fill_uniform_dense_cpu!, signature)))
    @test occursin("_fill_uniform_blocks4_cpu!", lowered)

    for T in PURE_UNIFORM_TYPES,
        bit in (UInt16(0), UInt16(1), UInt16(31), UInt16(63), UInt16(127)),
        delta in (-1, 0, 1)

        x4_group =
            T === Bool ? 512 :
            T <: Union{Int32,UInt32} ? 16 :
            T <: Union{Int64,UInt64} ? 8 : T === Float32 ? 64 : 1
        group = max(
            IR._dense_fill_group(T),
            cld(4 * 128 - Int(bit), _uniform_width(T)),
            x4_group,
        )
        count = group + delta
        count < 0 && continue
        rng = _positioned(Philox4x32, 0x5254, UInt64(9), bit)
        expected_rng, expected = _reference_chain(rng, T, count)
        destination = Vector{T}(undef, count)
        next_rng, result = rand_next!(rng, destination; threaded = false)
        @test result === destination
        @test destination == expected
        @test next_rng.position == expected_rng.position
    end

    for T in PURE_UNIFORM_TYPES, bit in (UInt16(0), UInt16(61)), delta in (-1, 0, 1)
        rng = _positioned(Philox4x32, 0x5255, UInt64(5), bit)
        chunk = IR._dense_fill_chunk_elements(T)
        count = 4chunk + delta
        serial = Vector{T}(undef, count)
        threaded = similar(serial)
        rand_next!(rng, serial; threaded = false)
        rand_next!(rng, threaded; threaded = true)
        sync_cpu()
        @test threaded == serial
    end

    terminal_base = Philox4x32(0x5255)
    for (T, count, blocks) in (
        (Bool, 512, UInt64(4)),
        (UInt32, 16, UInt64(4)),
        (UInt64, 8, UInt64(4)),
        (Float32, 64, UInt64(12)),
    )
        position =
            IR._Position64(IR._max_block(terminal_base) - blocks + UInt64(1), UInt16(0))
        rng = IR._rebuild(terminal_base, position, terminal_base.device)
        expected = map(1:count) do index
            cursor = IR._rebuild(
                rng,
                _reference_position(rng, (index - 1) * _uniform_width(T)),
                rng.device,
            )
            _reference_uniform(cursor, T)
        end
        destination = Vector{T}(undef, count)
        next_rng, result = rand_next!(rng, destination; threaded = false)
        @test result == expected
        @test next_rng.position == IR._terminal64(IR._max_block(rng))
    end

    base = Philox4x32(0x5256)
    for T in PURE_UNIFORM_TYPES
        width = IR._draw_bits(T)
        position = IR._Position64(IR._max_block(base), IR._block_bits(base) - width)
        rng = IR._rebuild(base, position, base.device)
        destination = Vector{T}(undef, 1)
        next_rng, result = rand_next!(rng, destination; threaded = false)
        @test result[1] === _reference_uniform(rng, T)
        @test next_rng.position == IR._terminal64(IR._max_block(rng))
    end

    rng = Philox4x32(0x5257)
    for T in (Bool, UInt32, Int32, UInt64, Int64, Float32)
        destination = Vector{T}(undef, 1024)
        @test @inferred(rand_next!(rng, destination; threaded = false)) isa
              Tuple{typeof(rng),typeof(destination)}
        rand_next!(rng, destination; threaded = false)
        @test _serial_fill_allocations(rng, destination) == 0
        signature = Tuple{
            typeof(rng),
            typeof(rng.position),
            typeof(destination),
            Type{T},
            Base.OneTo{Int},
        }
        typed_ir = sprint(
            show,
            code_typed(IR._fill_uniform_dense_cpu!, signature; optimize = true),
        )
        llvm_ir = sprint() do io
            code_llvm(
                io,
                IR._fill_uniform_dense_cpu!,
                signature;
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

@testset "R49 serial fills stay on the calling task" begin
    rng = _positioned(Philox4x32, 0x526, UInt64(3), UInt16(29))
    caller = current_task()
    probe = TaskWriteProbe(Vector{UInt32}(undef, 37))
    next_rng, returned = rand_next!(rng, probe; threaded = false)
    @test returned === probe
    @test all(task -> task === caller, probe.writers)
    expected_rng, expected = _reference_chain(rng, UInt32, 37)
    @test probe.data == expected
    @test next_rng.position == expected_rng.position
end

@testset "R39, R40, and R54 packed fill validation and preflight" begin
    rng = Philox4x32(0x527)
    wrong = WrongDeviceArray(UInt32[])
    @test_throws ArgumentError rand!(rng, wrong)
    @test_throws ArgumentError rand_next!(rng, wrong)

    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    empty = UInt32[]
    @test rand!(exhausted, empty; threaded = false) === empty
    empty_next, empty_result = rand_next!(exhausted, empty; threaded = false)
    @test empty_result === empty
    @test empty_next === exhausted

    for F in FAMILY_TYPES, T in PURE_UNIFORM_TYPES
        base = F(0x528)
        width = IR._draw_bits(T)
        position =
            base.position isa IR._Position64 ?
            IR._Position64(IR._max_block(base), IR._block_bits(base) - width) :
            IR._Position128(typemax(UInt64), typemax(UInt64), IR._block_bits(base) - width)
        last = IR._rebuild(base, position, base.device)
        destination = fill(convert(T, T === Bool ? true : 1), 2)
        before = copy(destination)
        @test_throws ArgumentError rand!(last, destination; threaded = false)
        @test destination == before
        @test_throws ArgumentError rand_next!(last, destination; threaded = false)
        @test destination == before
        @test last.position == position

        final = Vector{T}(undef, 1)
        final_next, final_result = rand_next!(last, final; threaded = false)
        @test final_result === final
        @test final[1] === _reference_uniform(last, T)
        expected_terminal =
            last.position isa IR._Position64 ? IR._terminal64(IR._max_block(last)) :
            IR._terminal128()
        @test final_next.position == expected_terminal
    end

    lookups = Ref(0)
    probe = BackendProbe(Vector{UInt32}(undef, 17), lookups)
    _, expected = _reference_chain(rng, UInt32, 17)
    rand!(rng, probe; threaded = false)
    @test probe.data == expected
    @test lookups[] == 0
    rand!(rng, probe; threaded = true)
    sync_cpu()
    @test probe.data == expected
    @test lookups[] == 1
end

@testset "R23 and R30 packed uniform method, inference, allocation, and IR" begin
    for F in FAMILY_TYPES
        rng = F(0x529)
        default_next, default_value = rand_next(rng)
        typed_next, typed_value = rand_next(rng, Float64)
        @test default_next === typed_next
        @test default_value === typed_value

        for T in PURE_UNIFORM_TYPES
            destination = Vector{T}(undef, 7)
            @test which(rand, (typeof(rng), Type{T})).module === IR
            @test which(rand_next, (typeof(rng), Type{T})).module === IR
            @test which(rand!, (typeof(rng), typeof(destination))).module === IR
            @test which(rand_next!, (typeof(rng), typeof(destination))).module === IR
            @test which(randat, (typeof(rng), Type{T}, Int)).module === IR

            @test @inferred(rand(rng, T)) isa T
            @test @inferred(rand_next(rng, T)) isa Tuple{typeof(rng),T}
            @test @inferred(randat(rng, T, 3)) isa T
            @test @inferred(rand!(rng, destination; threaded = false)) === destination
            @test @inferred(rand_next!(rng, destination; threaded = false)) isa
                  Tuple{typeof(rng),typeof(destination)}

            rand(rng, T)
            rand_next(rng, T)
            randat(rng, T, 3)
            rand_next!(rng, destination; threaded = false)
            @test @allocated(rand(rng, T)) == 0
            @test @allocated(rand_next(rng, T)) == 0
            @test @allocated(randat(rng, T, 3)) == 0
            @test _serial_fill_allocations(rng, destination) == 0
        end
    end

    rng = Philox4x64(0x52a)
    @test_throws ArgumentError rand(rng)
    dense_signature = Tuple{
        typeof(rng),
        typeof(rng.position),
        Vector{Float32},
        Type{Float32},
        Base.OneTo{Int},
    }
    for (function_, signature) in (
        (rand, Tuple{typeof(rng),Type{UInt64}}),
        (rand_next, Tuple{typeof(rng),Type{Float64}}),
        (randat, Tuple{typeof(rng),Type{UInt32},Int}),
        (IR._fill_uniform_dense_cpu!, dense_signature),
    )
        typed_ir = sprint(show, code_typed(function_, signature; optimize = true))
        llvm_ir = sprint() do io
            code_llvm(
                io,
                function_,
                signature;
                raw = false,
                dump_module = false,
                optimize = true,
            )
        end
        @test !occursin("BigInt", typed_ir)
        @test !occursin("UInt128", typed_ir)
        @test !occursin(r"\bi128\b", llvm_ir)
    end

    for unsupported in (Union{UInt32,Float32}, Float16)
        @test !applicable(rand, rng, unsupported)
        @test !applicable(rand_next, rng, unsupported)
        @test !applicable(randat, rng, unsupported, 1)
    end


    destination = Vector{UInt32}(undef, 1)
    @test Base.kwarg_decl(which(rand!, (typeof(rng), typeof(destination)))) == [:threaded]
    @test Base.kwarg_decl(which(rand_next!, (typeof(rng), typeof(destination)))) ==
          [:threaded]
    @test !applicable(rand!, rng, destination, false)
    @test !applicable(rand_next!, rng, destination, false)
    @test_throws TypeError rand!(rng, destination; threaded = 1)
    @test_throws TypeError rand_next!(rng, destination; threaded = 1)
    @test_throws MethodError rand!(rng, destination; serial = false)
    @test_throws MethodError rand_next!(rng, destination; serial = false)
    @test_throws MethodError rand(rng, UInt32; threaded = false)
end
