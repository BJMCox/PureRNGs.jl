using CUDA
using Distributions
using Enzyme
using KernelAbstractions
using PureRNGs
using MLDataDevices
using Random
using Test

include(joinpath(@__DIR__, "..", "..", "..", "fixtures.jl"))

const CUDA_EXT = Base.get_extension(IR, :PureRNGsCUDAExt)

const UNIFORM_TYPES = (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
const RANGE_TYPES = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
const PACKED_INTEGER_CASES = (
    (Philox2x64, UInt32, 1 << 20),
    (Philox2x64, UInt64, 1 << 20),
    (Philox4x64, UInt32, 4096),
    (Philox4x64, UInt64, 4096),
    (Threefry4x64, UInt32, 8192),
    (Threefry4x64, UInt64, 4096),
    (ChaCha, UInt32, 4096),
    (ChaCha, UInt64, 4096),
)

mutable struct DeviceAgnosticWeights{T} <: AbstractVector{T}
    values::Vector{T}
    reads::Base.RefValue{Int}
end

Base.size(weights::DeviceAgnosticWeights) = size(weights.values)
function Base.getindex(weights::DeviceAgnosticWeights, index::Int)
    weights.reads[] += 1
    return weights.values[index]
end

MLD.get_device(::DeviceAgnosticWeights) = nothing

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)

# An extension is not a submodule of its parent, so a recursive scan that starts
# at PureRNGs never reaches it. Scan each loaded extension itself.
@testset "extension ambiguities" begin
    for name in (
        :PureRNGsCUDAExt,
        :PureRNGsEnzymeCoreExt,
        :PureRNGsDistributionsExt,
        :PureRNGsKernelAbstractionsExt,
    )
        extension = Base.get_extension(IR, name)
        @testset "$name" begin
            @test isempty(Test.detect_ambiguities(extension; recursive = true))
        end
    end
end

_range(::Type{T}) where {T<:Signed} = T(-31):T(3):T(41)
_range(::Type{T}) where {T<:Unsigned} = T(2):T(3):T(74)

function _chain(rng, draw, count::Int, ::Type{T}) where {T}
    values = Vector{T}(undef, count)
    for index in eachindex(values)
        values[index], rng = draw(rng)
    end
    return rng, values
end

function _device_id(array)
    device = MLD.get_device(array)
    @assert device isa MLD.CUDADevice{<:CUDA.CuDevice}
    return CUDA.deviceid(device.device)
end

function _check_array_draw(
    cpu_rng,
    gpu_rng,
    argument,
    ::Type{T},
    draw,
    next_draw;
    fills = (),
    cpu_parity::Bool = true,
) where {T}
    values = draw(gpu_rng, argument, 19)
    @test values isa CUDA.CuArray{T,1}
    @test _device_id(values) == CUDA.deviceid(primary)
    cpu_parity && @test(isequal(Array(values), draw(cpu_rng, argument, 19)))
    @test isequal(Array(draw(gpu_rng, argument, 7)), Array(values)[1:7])
    @test isequal(vec(Array(draw(gpu_rng, argument, 3, 5))), Array(values)[1:15])

    continued, next_gpu = next_draw(gpu_rng, argument, 19)
    next_scalar, scalar_values = _chain(gpu_rng, rng -> next_draw(rng, argument), 19, T)
    @test continued isa CUDA.CuArray{T,1}
    @test _device_id(continued) == CUDA.deviceid(primary)
    expected = cpu_parity ? scalar_values : Array(values)
    @test isequal(Array(continued), expected)
    @test next_gpu.position == next_scalar.position
    @test next_gpu.device == next_scalar.device == gpu_rng.device

    for (fill, next_fill) in fills
        destination = similar(values)
        @test fill(gpu_rng, destination) === destination
        @test isequal(Array(destination), Array(values))
        returned, fill_next = next_fill(gpu_rng, destination)
        @test returned === destination
        @test fill_next.position == next_gpu.position
        @test fill_next.device == gpu_rng.device
        serial = similar(values)
        _, serial_next = next_fill(gpu_rng, serial; threaded = false)
        @test isequal(Array(serial), Array(values))
        @test serial_next.position == next_gpu.position
        @test serial_next.device == gpu_rng.device
    end

    empty, empty_next = next_draw(gpu_rng, argument, 0)
    @test empty isa CUDA.CuArray{T,1}
    @test isempty(empty)
    @test empty_next.position == gpu_rng.position
    return values
end

function _check_array_continuation(rng, argument, next_draw)
    matrix, matrix_next = next_draw(rng, argument, 2, 3)
    vector, vector_next = next_draw(rng, argument, 6)

    @test size(matrix) == (2, 3)
    @test isequal(vec(Array(matrix)), Array(vector))
    @test matrix_next.position == vector_next.position
end

function _address_kernel!(destination, rng, offset)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index <= length(destination) &&
        (@inbounds destination[index] = rand_at(rng, eltype(destination), offset + index))
    return
end

function _normal_address_kernel!(destination, rng)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index <= length(destination) &&
        (@inbounds destination[index] = randn_at(rng, eltype(destination), index))
    return
end

@inline function _exponential_raw(rng, ::Type{Float32})
    return IR._extract_bits_unchecked(
        rng,
        IR._position_block(rng.position),
        rng.position.bit,
        Val(23),
    )
end

@inline function _exponential_raw(rng, ::Type{Float64})
    return IR._extract_bits_unchecked(
        rng,
        IR._position_block(rng.position),
        rng.position.bit,
        Val(52),
    )
end

function _exponential_api_kernel!(values, raw, lattice, rng)
    if CUDA.threadIdx().x == 1
        T = eltype(values)
        continued, next_rng = randexp_next(rng, T)
        k = _exponential_raw(rng, T)
        u, v = IR._exponential_lattice(T, k)
        @inbounds begin
            values[1] = randexp(rng, T)
            values[2] = continued
            values[3] = randexp_at(rng, T, 1)
            values[4] = randexp(next_rng, T)
            values[5] = randexp_at(rng, T, 2)
            raw[1] = k
            lattice[1] = u
            lattice[2] = v
        end
    end
    return
end

function _device_api_kernel!(uniform, normal32, normal64, ranges, signed32, signed64, rng)
    if CUDA.threadIdx().x == 1
        continued_uniform, next_uniform = rand_next(rng, UInt32)
        continued_signed32, next_signed32 = rand_next(rng, Int32)
        continued_signed64, next_signed64 = rand_next(rng, Int64)
        continued_normal32, next_normal32 = randn_next(rng, Float32)
        continued_normal64, next_normal64 = randn_next(rng, Float64)
        range = UInt64(0):UInt64(1):(UInt64(1)<<40)
        continued_range, next_range = rand_next(rng, range)
        child = subrng(rng, UInt64(0x71))
        children = splitrng(rng, Val(2))
        @inbounds begin
            uniform[1] = rand(rng, UInt32)
            uniform[2] = rand_at(rng, UInt32, 1)
            uniform[3] = continued_uniform
            uniform[4] = rand(next_uniform, UInt32)
            uniform[5] = rand(child, UInt32)
            uniform[6] = rand(children[2], UInt32)
            signed32[1] = rand(rng, Int32)
            signed32[2] = continued_signed32
            signed32[3] = rand(next_signed32, Int32)
            signed64[1] = rand(rng, Int64)
            signed64[2] = continued_signed64
            signed64[3] = rand(next_signed64, Int64)
            normal32[1] = randn(rng, Float32)
            normal32[2] = randn_at(rng, Float32, 1)
            normal32[3] = continued_normal32
            normal32[4] = randn(next_normal32, Float32)
            normal32[5] = randn_at(rng, Float32, 2)
            normal64[1] = randn(rng, Float64)
            normal64[2] = randn_at(rng, Float64, 1)
            normal64[3] = continued_normal64
            normal64[4] = randn(next_normal64, Float64)
            normal64[5] = randn_at(rng, Float64, 2)
            ranges[1] = continued_range
            ranges[2] = rand(next_range, range)
        end
    end
    return
end

function _k64_range_kernel!(destination, rng)
    if CUDA.threadIdx().x == 1
        value, next_rng = rand_next(rng, K64_RANGE)
        @inbounds begin
            destination[1] = value
            destination[2] = rand(next_rng, K64_RANGE)
        end
    end
    return
end

@inline function _write_group!(destination, offset, coefficients)
    @inbounds for index in eachindex(coefficients)
        destination[offset+index] = coefficients[index]
    end
    return offset + length(coefficients)
end

# The Float32 table has four groups and the Float64 table six, so the walk
# recurses over the tuple and unrolls instead of naming a fixed count.
@inline _write_coefficients!(destination, offset, coefficients::Tuple{}) = nothing
@inline _write_coefficients!(destination, offset, coefficients::Tuple) =
    _write_coefficients!(
        destination,
        _write_group!(destination, offset, first(coefficients)),
        Base.tail(coefficients),
    )

@inline _write_coefficients!(destination, coefficients) =
    _write_coefficients!(destination, 0, coefficients)

# The CUDA token's transform at a strided lattice point. The stride is a
# type parameter so the index arithmetic folds.
function _giles_lattice_kernel!(destination, ::Type{T}, ::Val{S}) where {T,S}
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if index <= length(destination)
        k = UInt64(index - 1) * UInt64(S)
        @inbounds destination[index] =
            IR._normal_transform(IR._CUDA_BACKEND, IR._open_midpoint(T, k))
    end
    return
end

function _as241_kernel!(coefficients32, probes32, coefficients64, probes64)
    if CUDA.threadIdx().x == 1
        _write_coefficients!(coefficients32, IR._as241_coefficients(Float32))
        _write_coefficients!(coefficients64, IR._as241_coefficients(Float64))
        @inbounds begin
            probes32[1] = IR._open_midpoint(Float32, UInt64(0))
            probes32[2] = IR._open_midpoint(Float32, (UInt64(1) << 23) - UInt64(1))
            probes32[3] = IR._as241(0.5f0)
            probes32[4] = IR._as241(0.95f0)
            probes32[5] = IR._as241(Float32(0x1p-24))
            probes64[1] = IR._open_midpoint(Float64, UInt64(0))
            probes64[2] = IR._open_midpoint(Float64, (UInt64(1) << 52) - UInt64(1))
            probes64[3] = IR._as241(0.5)
            probes64[4] = IR._as241(0.95)
            probes64[5] = IR._as241(1.0e-20)
        end
    end
    return
end

_flatten(coefficients) = vcat((collect(group) for group in coefficients)...)

function _check_scalar_allocation(call)
    call()
    @test @allocated(call()) == 0
end

function _last_draw_rng(rng, width::UInt16)
    block_bits = IR._block_bits(rng)
    blocks, remainder = divrem(width, block_bits)
    delta = iszero(remainder) ? blocks - UInt16(1) : blocks
    bit = iszero(remainder) ? UInt16(0) : block_bits - remainder
    position = if rng.position isa IR._Position64
        IR._Position64(IR._max_block(rng) - UInt64(delta), bit)
    else
        IR._Position128(typemax(UInt64) - UInt64(delta), typemax(UInt64), bit)
    end
    return IR._rebuild(rng, position, rng.device)
end

_terminal(rng::IR._Position64Generators) = IR._terminal64(IR._max_block(rng))
_terminal(::IR._Position128Generators) = IR._terminal128()

function _positioned_at_bit(rng, block::UInt64, bit::UInt16)
    position = if rng.position isa IR._Position64
        IR._Position64(block, bit)
    else
        IR._Position128(block, UInt64(7), bit)
    end
    return IR._rebuild(rng, position, rng.device)
end

const COOPERATIVE_UNIFORM_TYPES = (Bool, Float32, Float64)
const NATURAL_128_TYPES = (
    (Philox4x32, UInt32),
    (Philox4x32, Int32),
    (Philox4x32, UInt64),
    (Philox4x32, Int64),
    (Threefry4x32, UInt32),
    (Threefry4x32, Int32),
    (Threefry4x32, UInt64),
    (Threefry4x32, Int64),
)
const PACKED_DRAW_SPECS = (
    (Bool, rand_next, rand_next!, UInt16(1), false),
    (UInt32, rand_next, rand_next!, UInt16(32), false),
    (Int32, rand_next, rand_next!, UInt16(32), false),
    (UInt64, rand_next, rand_next!, UInt16(64), false),
    (Int64, rand_next, rand_next!, UInt16(64), false),
    (Float32, rand_next, rand_next!, UInt16(24), false),
    (Float64, rand_next, rand_next!, UInt16(53), false),
    (Float32, randn_next, randn_next!, UInt16(23), true),
    (Float64, randn_next, randn_next!, UInt16(52), true),
)
const K64_RANGE = UInt32(3):UInt32(1003)
const K128_RANGE = UInt64(7):UInt64(3):UInt64(0xfffffffffffffffd)

struct DeviceAgnosticPopulation{T}
    values::Vector{T}
    starts::Base.RefValue{Int}
end

DeviceAgnosticPopulation(values::Vector{T}) where {T} =
    DeviceAgnosticPopulation{T}(values, Ref(0))

Base.IteratorSize(::Type{<:DeviceAgnosticPopulation}) = Base.HasLength()
Base.IteratorEltype(::Type{<:DeviceAgnosticPopulation}) = Base.HasEltype()
Base.eltype(::Type{DeviceAgnosticPopulation{T}}) where {T} = T
Base.length(population::DeviceAgnosticPopulation) = length(population.values)
function Base.iterate(population::DeviceAgnosticPopulation)
    population.starts[] += 1
    return iterate(population.values)
end
Base.iterate(population::DeviceAgnosticPopulation, state) =
    iterate(population.values, state)
MLD.get_device(::DeviceAgnosticPopulation) = nothing

struct DeviceAgnosticArray{T} <: AbstractVector{T}
    values::Vector{T}
    reads::Base.RefValue{Int}
end

DeviceAgnosticArray(values::Vector{T}) where {T} = DeviceAgnosticArray{T}(values, Ref(0))

Base.size(population::DeviceAgnosticArray) = size(population.values)
function Base.getindex(population::DeviceAgnosticArray, index::Int)
    population.reads[] += 1
    return population.values[index]
end
Base.eachindex(population::DeviceAgnosticArray) = reverse(eachindex(population.values))
MLD.get_device(::DeviceAgnosticArray) = nothing

function _check_public_packed_fill(
    rng,
    ::Type{T},
    count,
    next_draw,
    next_fill;
    addressed_normal::Bool = false,
) where {T}
    expected_next, scalar_values = _chain(rng, current -> next_draw(current, T), count, T)
    allocated, allocated_next = next_draw(rng, T, count)
    expected = if addressed_normal
        addressed = similar(allocated)
        threads = min(count, 256)
        CUDA.@sync CUDA.@cuda threads = threads blocks = cld(count, threads) _normal_address_kernel!(
            addressed,
            rng,
        )
        Array(addressed)
    else
        scalar_values
    end
    destination = similar(allocated)
    _, filled_next = next_fill(rng, destination)
    @test isequal(
        (
            Array(allocated),
            allocated_next.position,
            Array(destination),
            filled_next.position,
        ),
        (expected, expected_next.position, expected, expected_next.position),
    )
    return allocated_next
end

function _check_public_packed_addresses(rng, ::Type{T}, count) where {T}
    destination = CUDA.CuArray{T}(undef, count)
    returned, next_rng = rand_next!(rng, destination)
    @test returned === destination

    values = Array(destination)
    indices = (1, 2, count ÷ 2, count)
    @test values[collect(indices)] == map(index -> rand_at(rng, T, index), collect(indices))

    bits_lo, bits_hi = IR._bit_span(UInt64(count), IR._draw_bits(T))
    @test next_rng.position == IR._reserve(rng, bits_lo, bits_hi).position
    return next_rng
end

function _check_cooperative_kernel_code(
    backend,
    rng,
    packed,
    ::Type{T},
    plan,
    stream_aligned;
    check_store::Bool,
    codec = Val(:uniform),
    storage_type = eltype(packed),
) where {T}
    kernel = CUDA_EXT._cooperative_fill_kernel!(backend)
    outputs_per_store = CUDA_EXT._outputs_per_store(plan)
    workgroup = IR._val_count(plan[3])
    block_width = Val(Int(IR._block_bits(rng)))
    typed = KernelAbstractions.@ka_code_typed kernel(
        rng,
        packed,
        Val(T),
        Val(storage_type),
        Val(IR._draw_bits(T)),
        block_width,
        plan[2],
        plan[3],
        outputs_per_store,
        stream_aligned,
        codec,
        ndrange = workgroup,
        workgroupsize = workgroup,
    )
    typed_text = sprint(show, typed)
    llvm = sprint() do io
        CUDA.@device_code_llvm io = io kernel(
            rng,
            packed,
            Val(T),
            Val(storage_type),
            Val(IR._draw_bits(T)),
            block_width,
            plan[2],
            plan[3],
            outputs_per_store,
            stream_aligned,
            codec;
            ndrange = workgroup,
            workgroupsize = workgroup,
        )
    end
    @test !occursin("UInt128", typed_text)
    @test !occursin("BigInt", typed_text)
    @test !occursin(r"\bi128\b", llvm)
    if check_store
        sass = sprint() do io
            CUDA.@device_code_sass io = io kernel(
                rng,
                packed,
                Val(T),
                Val(storage_type),
                Val(IR._draw_bits(T)),
                block_width,
                plan[2],
                plan[3],
                outputs_per_store,
                stream_aligned,
                codec;
                ndrange = workgroup,
                workgroupsize = workgroup,
            )
        end
        @test occursin("STG.E.128", sass)
    end
    return nothing
end

function _check_public_range(rng, range, count)
    T = eltype(range)
    expected_next, expected = _chain(rng, current -> rand_next(current, range), count, T)
    allocated, allocated_next = rand_next(rng, range, count)
    @test (Array(allocated), allocated_next.position) == (expected, expected_next.position)
end

function _device_events(call)
    call()
    CUDA.synchronize()
    profile = CUDA.@profile raw = true begin
        call()
        CUDA.synchronize()
    end
    return _cuda_profile_events(profile)
end

function _cuda_profile_events(profile)
    first_sync = findfirst(==("cuCtxSynchronize"), profile.host.name)
    last_sync = findlast(==("cuCtxSynchronize"), profile.host.name)
    @test first_sync !== nothing
    @test last_sync !== nothing
    @test first_sync != last_sync
    window = findall(
        index ->
            profile.device.start[index] >= profile.host.stop[first_sync] &&
            profile.device.stop[index] <= profile.host.stop[last_sync],
        eachindex(profile.device.name),
    )
    kernels = filter(index -> !ismissing(profile.device.grid[index]), window)
    kernel_names = profile.device.name[kernels]
    memory = filter(index -> !ismissing(profile.device.size[index]), window)
    copies = filter(index -> startswith(profile.device.name[index], "[copy "), memory)
    memsets = filter(index -> startswith(profile.device.name[index], "[set "), memory)
    @test sort!(vcat(kernels, copies, memsets)) == window

    source_is_host(name) = occursin(r"^\[copy (pageable|pinned) to ", name)
    destination_is_host(name) = occursin(r" to (pageable|pinned) memory\]$", name)
    host_to_device = filter(index -> source_is_host(profile.device.name[index]), copies)
    device_to_host =
        filter(index -> destination_is_host(profile.device.name[index]), copies)
    return (; kernels, kernel_names, copies, memsets, host_to_device, device_to_host)
end

_device_kernel_events(call) = length(_device_events(call).kernels)

_event_sizes(profile, indices) = profile.device.size[indices]

devices = collect(CUDA.devices())
primary = first(devices)
CUDA.device!(primary)
device = MLD.CUDADevice(primary)

# Device code takes the widening product from `mul.hi.u64` and the host keeps
# the portable four-product form, so the two must agree bit for bit.
@testset "CUDA high product matches the portable product" begin
    operands = (
        (UInt64(0), UInt64(0)),
        (typemax(UInt64), typemax(UInt64)),
        (0x243f6a8885a308d3, 0xd2b74407b1ce6e93),
        (0x8000000000000000, UInt64(3)),
        (UInt64(1) << 32, UInt64(1) << 32),
    )
    left = CUDA.CuArray(collect(first.(operands)))
    right = CUDA.CuArray(collect(last.(operands)))
    @test Array(map((a, b) -> first(IR._mulhilo64(a, b)), left, right)) ==
          [first(IR._mulhilo64(a, b)) for (a, b) in operands]
    @test Array(map((a, b) -> last(IR._mulhilo64(a, b)), left, right)) ==
          [last(IR._mulhilo64(a, b)) for (a, b) in operands]
end

@testset "CUDA Bool packed paths preserve every generator stream" begin
    for F in GENERATOR_TYPES
        rng = device(F(0x787))
        block_bits = Int(IR._block_bits(rng))

        count = 2block_bits
        _check_public_packed_fill(rng, Bool, count, rand_next, rand_next!)
        positioned = _positioned_at_bit(rng, UInt64(9), UInt16(0))
        _check_public_packed_fill(positioned, Bool, count, rand_next, rand_next!)

        offset = _positioned_at_bit(rng, UInt64(9), UInt16(5))
        _check_public_packed_fill(offset, Bool, block_bits, rand_next, rand_next!)
        _check_public_packed_fill(rng, Bool, block_bits + 1, rand_next, rand_next!)

        storage = CUDA.CuArray{Bool}(undef, block_bits + 16)
        expected_next, expected =
            _chain(rng, current -> rand_next(current, Bool), block_bits, Bool)
        for first in (2, 17)
            view = @view storage[first:(first+block_bits-1)]
            returned, filled_next = rand_next!(rng, view)
            @test returned === view
            @test Array(view) == expected
            @test filled_next.position == expected_next.position
        end

        terminal_position =
            rng.position isa IR._Position64 ?
            IR._Position64(IR._max_block(rng), UInt16(0)) :
            IR._Position128(typemax(UInt64), typemax(UInt64), UInt16(0))
        terminal_rng = IR._rebuild(rng, terminal_position, rng.device)
        terminal =
            _check_public_packed_fill(terminal_rng, Bool, block_bits, rand_next, rand_next!)
        @test terminal.position == _terminal(rng)
    end

    rng = device(Philox4x32(0x788))
    destination = CUDA.CuArray{Bool}(undef, 4096)
    events = _device_events(() -> rand_next!(rng, destination))
    @test !isempty(events.kernels)
    @test isempty(events.host_to_device)
    @test isempty(events.device_to_host)
end

@testset "CUDA Philox4x32 Float32 vector stores preserve the dense stream" begin
    rng = device(Philox4x32(0x786))

    _check_public_packed_fill(rng, Float32, 4096, rand_next, rand_next!)
    positioned = _positioned_at_bit(rng, UInt64(9), UInt16(0))
    _check_public_packed_fill(positioned, Float32, 4096, rand_next, rand_next!)

    offset = _positioned_at_bit(rng, UInt64(9), UInt16(5))
    _check_public_packed_fill(offset, Float32, 2048, rand_next, rand_next!)
    _check_public_packed_fill(rng, Float32, 2052, rand_next, rand_next!)

    storage = CUDA.CuArray{Float32}(undef, 2052)
    expected_next, expected =
        _chain(rng, current -> rand_next(current, Float32), 2048, Float32)
    for first in (1, 2)
        view = @view storage[first:(first+2047)]
        returned, filled_next = rand_next!(rng, view)
        @test returned === view
        @test Array(view) == expected
        @test filled_next.position == expected_next.position
    end

    terminal_rng = _positioned_at_bit(rng, typemax(UInt64) - UInt64(383), UInt16(0))
    terminal = _check_public_packed_fill(terminal_rng, Float32, 2048, rand_next, rand_next!)
    @test terminal.position == _terminal(rng)
end

@testset "CUDA non-Philox float vector stores preserve the dense stream" begin
    for F in GENERATOR_TYPES
        F === Philox4x32 && continue
        for T in (Float32, Float64)
            rng = device(F(0x788))
            outputs = T === Float32 ? 2048 : 1024
            outputs_per_store = T === Float32 ? 4 : 2

            _check_public_packed_fill(rng, T, outputs, rand_next, rand_next!)
            block_bits = IR._block_bits(rng)
            block = rng.position isa IR._Position128 ? typemax(UInt64) : UInt64(9)
            boundary_bits = unique(
                filter(
                    bit -> bit < block_bits,
                    UInt16[
                        1,
                        31,
                        63,
                        64,
                        127,
                        128,
                        block_bits-UInt16(5),
                        block_bits-UInt16(1),
                    ],
                ),
            )
            for bit in boundary_bits
                offset = _positioned_at_bit(rng, block, bit)
                _check_public_packed_fill(
                    offset,
                    T,
                    8outputs_per_store,
                    rand_next,
                    rand_next!,
                )
            end

            storage = CUDA.CuArray{T}(undef, outputs + 1)
            expected_next, expected =
                _chain(rng, current -> rand_next(current, T), outputs, T)
            for first in (1, 2)
                view = @view storage[first:(first+outputs-1)]
                returned, filled_next = rand_next!(rng, view)
                @test returned === view
                @test Array(view) == expected
                @test filled_next.position == expected_next.position
            end

            packed_count = 8outputs_per_store
            packed_next, packed_expected =
                _chain(rng, current -> rand_next(current, T), packed_count, T)
            matrix = CUDA.CuArray{T}(undef, 1, packed_count)
            returned_matrix, matrix_next = rand_next!(rng, matrix)
            @test returned_matrix === matrix
            @test vec(Array(matrix)) == packed_expected
            @test matrix_next.position == packed_next.position

            strided_storage = CUDA.CuArray{T}(undef, 2packed_count)
            strided = @view strided_storage[1:2:(2packed_count)]
            returned_strided, strided_next = rand_next!(rng, strided)
            @test returned_strided === strided
            @test Array(strided) == packed_expected
            @test strided_next.position == packed_next.position

            last_pack_rng =
                _last_draw_rng(rng, UInt16(outputs_per_store) * IR._draw_bits(T))
            unchanged = CUDA.fill(T(0.25), 2outputs_per_store)
            @test_throws StreamExhausted rand_next!(last_pack_rng, unchanged)
            @test Array(unchanged) == fill(T(0.25), 2outputs_per_store)

            terminal_rng = _last_draw_rng(rng, UInt16(outputs_per_store) * IR._draw_bits(T))
            terminal = _check_public_packed_fill(
                terminal_rng,
                T,
                outputs_per_store,
                rand_next,
                rand_next!,
            )
            @test terminal.position == _terminal(rng)
        end
    end
end

@testset "CUDA natural 128-bit stores preserve the stream" begin
    count = 4080

    for (F, T) in NATURAL_128_TYPES
        rng = device(F(0x784))
        outputs_per_pack = 16 ÷ sizeof(T)
        expected_next, expected = _chain(rng, current -> rand_next(current, T), count, T)

        allocated, allocated_next = rand_next(rng, T, 3, 1360)
        @test Array(allocated) == reshape(expected, 3, 1360)
        @test allocated_next.position == expected_next.position

        positioned = _positioned_at_bit(rng, UInt64(9), UInt16(0))
        _check_public_packed_fill(positioned, T, count, rand_next, rand_next!)
        for bit in (UInt16(5), UInt16(64))
            offset = _positioned_at_bit(rng, UInt64(9), bit)
            _check_public_packed_fill(offset, T, 64, rand_next, rand_next!)
        end

        _check_public_packed_fill(rng, T, 65, rand_next, rand_next!)

        aligned_first = outputs_per_pack + 1
        storage = CUDA.CuArray{T}(undef, aligned_first + 63)
        expected_fallback_next = _chain(rng, current -> rand_next(current, T), 64, T)[1]
        for first in (2, aligned_first)
            view = @view storage[first:(first+63)]
            returned, fallback_next = rand_next!(rng, view)
            @test returned === view
            @test Array(view) == expected[1:64]
            @test fallback_next.position == expected_fallback_next.position
        end

        terminal_rng = _positioned_at_bit(rng, typemax(UInt64), UInt16(0))
        terminal_expected, terminal_values =
            _chain(terminal_rng, current -> rand_next(current, T), outputs_per_pack, T)
        terminal_destination = CUDA.fill(zero(T), outputs_per_pack)
        _, terminal_next = rand_next!(terminal_rng, terminal_destination)
        @test Array(terminal_destination) == terminal_values
        @test terminal_next.position == terminal_expected.position == _terminal(rng)
    end
end

@testset "public fill plan classes launch CUDA kernels" begin
    cases = (
        () -> rand(device(Threefry2x32(0x771)), Bool, 64),
        () -> randn(device(Philox4x32(0x772)), Float64, 9),
        () -> rand(device(Threefry2x32(0x773)), UInt32, 9),
        () -> randn(device(Threefry2x32(0x774)), Float32, 9),
        () -> rand(device(Philox4x32(0x775)), K64_RANGE, 9),
        () -> rand(device(Philox4x32(0x776)), K128_RANGE, 9),
    )
    for call in cases
        @test _device_kernel_events(call) > 0
    end
end

@testset "cooperative public fills cover offsets and workgroups" begin
    rng = device(Philox4x32(0x781))
    block_bits = IR._block_bits(rng)
    for (T, next_draw, next_fill, width, addressed_normal) in PACKED_DRAW_SPECS
        next_draw === rand_next && T ∉ COOPERATIVE_UNIFORM_TYPES && continue
        outputs =
            next_draw === randn_next ? 512 : T === Bool ? 4096 : T === Float32 ? 2048 : 1024
        cases = (
            (UInt16(0), 1),
            (UInt16(61), outputs + 3),
            (block_bits - UInt16(5), 5 ÷ Int(width) + 2),
        )
        for (bit, count) in cases
            positioned = _positioned_at_bit(rng, UInt64(9), bit)
            _check_public_packed_fill(
                positioned,
                T,
                count,
                next_draw,
                next_fill;
                addressed_normal,
            )
        end
    end
end

@testset "grouped public fallbacks cover every remaining generator and type" begin
    for F in GENERATOR_TYPES, T in UNIFORM_TYPES
        F === Philox4x32 && T in COOPERATIVE_UNIFORM_TYPES && continue
        rng = device(F(0x782))
        block = rng.position isa IR._Position128 ? typemax(UInt64) : UInt64(9)
        positioned = _positioned_at_bit(rng, block, IR._block_bits(rng) - UInt16(5))
        _check_public_packed_fill(positioned, T, 9, rand_next, rand_next!)
    end

    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        F === Philox4x32 && continue
        rng = device(F(0x783))
        block = rng.position isa IR._Position128 ? typemax(UInt64) : UInt64(9)
        positioned = _positioned_at_bit(rng, block, IR._block_bits(rng) - UInt16(5))
        _check_public_packed_fill(
            positioned,
            T,
            9,
            randn_next,
            randn_next!;
            addressed_normal = true,
        )
    end
end

@testset "non-Philox normal packed fills preserve every generator stream" begin
    backend = CUDA.CUDABackend()
    for F in GENERATOR_TYPES, T in NORMAL_TYPES
        F === Philox4x32 && continue
        rng = device(F(0x782))
        plan = IR._device_fill_plan(backend, rng, IR._NormalCodec(rng.device), T)
        @test plan[1] == Val(:cooperative)
        _check_public_packed_fill(
            rng,
            T,
            IR._val_count(plan[2]),
            randn_next,
            randn_next!;
            addressed_normal = true,
        )
    end
end

@testset "packed integer stores preserve the dense stream" begin
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    for (F, T, count) in PACKED_INTEGER_CASES
        rng = device(F(0x784))
        @test extension_module._packed_integer_min_length(rng, T) == count

        _check_public_packed_addresses(rng, T, count)
        positioned = _positioned_at_bit(rng, UInt64(7), UInt16(5))
        _check_public_packed_addresses(positioned, T, count)
    end

    carry_rng = _positioned_at_bit(device(Philox4x64(0x784)), typemax(UInt64), UInt16(0))
    _check_public_packed_fill(carry_rng, Float64, 4096, rand_next, rand_next!)

    for (unsigned, signed, count) in ((UInt32, Int32, 8192), (UInt64, Int64, 4096))
        rng = device(Threefry4x64(0x785))
        unsigned_values = CUDA.CuArray{unsigned}(undef, count)
        signed_values = CUDA.CuArray{signed}(undef, count)
        _, unsigned_next = rand_next!(rng, unsigned_values)
        _, signed_next = rand_next!(rng, signed_values)
        @test Array(signed_values) == reinterpret(signed, Array(unsigned_values))
        @test signed_next.position == unsigned_next.position
    end

    for (T, count) in ((UInt32, 8192), (UInt64, 4096))
        rng = device(Threefry4x64(0x786))
        outputs_per_store = 16 ÷ sizeof(T)
        expected_next, expected = _chain(rng, current -> rand_next(current, T), count, T)

        dims = T === UInt32 ? (2, count ÷ 2) : (1, count)
        matrix = CUDA.CuArray{T}(undef, dims)
        returned, matrix_next = rand_next!(rng, matrix)
        @test returned === matrix
        @test vec(Array(matrix)) == expected
        @test matrix_next.position == expected_next.position

        fallback_count = count
        fallback_next, fallback_expected =
            _chain(rng, current -> rand_next(current, T), fallback_count, T)
        storage = CUDA.CuArray{T}(undef, fallback_count + 1)
        offset = @view storage[2:end]
        returned_offset, offset_next = rand_next!(rng, offset)
        @test returned_offset === offset
        @test Array(offset) == fallback_expected
        @test offset_next.position == fallback_next.position

        strided_storage = CUDA.CuArray{T}(undef, 2fallback_count)
        strided = @view strided_storage[1:2:end]
        returned_strided, strided_next = rand_next!(rng, strided)
        @test returned_strided === strided
        @test Array(strided) == fallback_expected
        @test strided_next.position == fallback_next.position

        last_pack_rng = _last_draw_rng(rng, UInt16(outputs_per_store) * IR._draw_bits(T))
        unchanged = CUDA.fill(typemax(T), 2outputs_per_store)
        @test_throws StreamExhausted rand_next!(last_pack_rng, unchanged)
        @test Array(unchanged) == fill(typemax(T), 2outputs_per_store)

        terminal = _check_public_packed_fill(
            last_pack_rng,
            T,
            outputs_per_store,
            rand_next,
            rand_next!,
        )
        @test terminal.position == _terminal(rng)
    end

    rng = device(Philox4x64(0x787))
    below = CUDA.CuArray{UInt32}(undef, 4092)
    destination = CUDA.CuArray{UInt32}(undef, 4096)
    below_events = _device_events(() -> rand_next!(rng, below))
    events = _device_events(() -> rand_next!(rng, destination))
    @test occursin("grouped_fill_kernel", only(below_events.kernel_names))
    @test occursin("cooperative_fill_kernel", only(events.kernel_names))
    @test length(events.kernels) == 1
    @test isempty(events.memsets)
    @test isempty(events.host_to_device)
    @test isempty(events.device_to_host)
end

@testset "Threefry4x32 aligned 64-bit fills use natural 128-bit packs" begin
    backend = CUDA.CUDABackend()
    rng = _positioned_at_bit(device(Threefry4x32(0x784)), UInt64(7), UInt16(0))
    for T in (UInt64, Int64)
        @test IR._device_fill_plan(backend, rng, Val(:uniform), T) ==
              (Val(:natural128_packed),)
    end
end

@testset "public K=64 grouped and K=128 generic ranges" begin
    for F in GENERATOR_TYPES, range in (K64_RANGE, K128_RANGE)
        rng = device(F(0x785))
        block = rng.position isa IR._Position128 ? typemax(UInt64) : UInt64(11)
        positioned = _positioned_at_bit(rng, block, IR._block_bits(rng) - UInt16(5))
        _check_public_range(positioned, range, 5)
    end
end

@testset "public packed tails consume the exact terminal draw" begin
    for F in (Philox4x32, Threefry4x64)
        rng = device(F(0x786))
        for (T, next_draw, next_fill, width, addressed_normal) in PACKED_DRAW_SPECS
            last_rng = _last_draw_rng(rng, width)
            terminal = _check_public_packed_fill(
                last_rng,
                T,
                1,
                next_draw,
                next_fill;
                addressed_normal,
            )
            @test terminal.position == _terminal(rng)
        end
    end
end

@testset "all array draws: residency, parity, shape, and fixed work" begin
    uniform_cases = (
        ((Philox4x32, T) for T in UNIFORM_TYPES)...,
        ((F, UInt64) for F in GENERATOR_TYPES if F !== Philox4x32)...,
    )
    for (F, T) in uniform_cases
        cpu_rng = F(0x123456)
        gpu_rng = device(cpu_rng)
        values = _check_array_draw(
            cpu_rng,
            gpu_rng,
            T,
            T,
            rand,
            rand_next;
            fills = ((rand!, rand_next!),),
        )
        @test gpu_rng.position == cpu_rng.position
        if F === Philox4x32 && T === UInt64
            serial, parallel = similar(values), similar(values)
            CUDA.@sync CUDA.@cuda threads = 1 blocks = 19 _address_kernel!(
                serial,
                gpu_rng,
                0,
            )
            CUDA.@sync CUDA.@cuda threads = 19 blocks = 1 _address_kernel!(
                parallel,
                gpu_rng,
                0,
            )
            @test Array(serial) == Array(values) == Array(parallel)
        end
    end

    normal_cases = (
        ((Philox4x32, T) for T in NORMAL_TYPES)...,
        ((F, Float64) for F in GENERATOR_TYPES if F !== Philox4x32)...,
    )
    for (F, T) in normal_cases
        cpu_rng = F(0x123456)
        gpu_rng = device(cpu_rng)
        values = _check_array_draw(
            cpu_rng,
            gpu_rng,
            T,
            T,
            randn,
            randn_next;
            fills = ((randn!, randn_next!),),
            cpu_parity = false,
        )
        if F === Philox4x32 && T === Float64
            serial, parallel = similar(values), similar(values)
            CUDA.@sync CUDA.@cuda threads = 1 blocks = 19 _normal_address_kernel!(
                serial,
                gpu_rng,
            )
            CUDA.@sync CUDA.@cuda threads = 19 blocks = 1 _normal_address_kernel!(
                parallel,
                gpu_rng,
            )
            @test isequal(Array(serial), Array(values))
            @test isequal(Array(parallel), Array(values))
        end
    end

    range_cases = (
        ((Philox4x32, T) for T in RANGE_TYPES)...,
        ((F, UInt64) for F in GENERATOR_TYPES if F !== Philox4x32)...,
    )
    for (F, T) in range_cases
        cpu_rng = F(0x123456)
        _check_array_draw(cpu_rng, device(cpu_rng), _range(T), T, rand, rand_next)
    end

    # The matrix above covers every type on Philox4x32 and one type on the
    # other generators. One further type per generator checks that the
    # continuation does not depend on the type the matrix chose.
    for F in GENERATOR_TYPES
        F === Philox4x32 && continue
        _check_array_continuation(device(F(0x123459)), Float32, rand_next)
        _check_array_continuation(device(F(0x12345a)), Float32, randn_next)
        _check_array_continuation(device(F(0x12345b)), _range(UInt16), rand_next)
    end

    wide = UInt64(0):UInt64(1):(UInt64(1)<<40)
    for F in GENERATOR_TYPES
        cpu_rng = F(0x123456)
        @test Array(rand(device(cpu_rng), wide, 17)) == rand(cpu_rng, wide, 17)
    end
end

@testset "CUDA AS241 coefficients, midpoint endpoints, and branches" begin
    reference32 = _flatten(IR._as241_coefficients(Float32))
    reference64 = _flatten(IR._as241_coefficients(Float64))
    coefficients32 = CUDA.CuArray{Float32}(undef, length(reference32))
    coefficients64 = CUDA.CuArray{Float64}(undef, length(reference64))
    probes32 = CUDA.CuArray{Float32}(undef, 5)
    probes64 = CUDA.CuArray{Float64}(undef, 5)
    CUDA.@sync CUDA.@cuda threads = 1 blocks = 1 _as241_kernel!(
        coefficients32,
        probes32,
        coefficients64,
        probes64,
    )

    @test Array(coefficients32) == reference32
    @test Array(coefficients64) == reference64
    result32, result64 = Array(probes32), Array(probes64)
    @test result32[1:2] == [Float32(0x1p-24), one(Float32) - Float32(0x1p-24)]
    @test result64[1:2] == [Float64(0x1p-53), one(Float64) - Float64(0x1p-53)]
    for (T, result, inputs) in (
        (Float32, result32, (0.5f0, 0.95f0, Float32(0x1p-24))),
        (Float64, result64, (0.5, 0.95, 1.0e-20)),
    )
        @test result[3] === zero(T)
        for index = 2:3
            @test isapprox(result[index+2], IR._as241(inputs[index]); rtol = 16eps(T))
        end
    end
end

@testset "device normal ulp over a strided lattice" begin
    # Float32: the reference is the same formula evaluated in Float64 at the
    # same lattice point. Its own error is about 1e-16 relative, eight decades
    # below a Float32 ulp, and the core suite gates the formula itself against a
    # BigFloat quantile. A toolkit change that moves the device `log` or `sqrt`
    # past the 5.0 ulp bound fails here.
    # Float64: the reference is the host evaluation of the same formula. The two
    # may differ through `log` and `sqrt` only, so a device build that
    # substitutes an approximate one for either fails here. An approximate
    # `Float64` square root alone costs about 6e9 ulp, nine decades past this
    # bound; the measured worst is 3 ulp.
    for (T, stride, points, bound) in
        ((Float32, 8, 1 << 20, 5.0), (Float64, 1 << 32, 1 << 20, 6.0))
        destination = CUDA.CuArray{T}(undef, points)
        CUDA.@sync CUDA.@cuda threads = 256 blocks = cld(points, 256) _giles_lattice_kernel!(
            destination,
            T,
            Val(stride),
        )
        values = Array(destination)
        worst = 0.0
        for index in eachindex(values)
            u = IR._open_midpoint(T, UInt64(index - 1) * UInt64(stride))
            reference =
                T === Float32 ? IR._normal_transform(IR._CUDA_BACKEND, Float64(u)) :
                Float64(IR._normal_transform(IR._CUDA_BACKEND, u))
            worst = max(
                worst,
                abs(Float64(values[index]) - reference) / Float64(eps(T(reference))),
            )
        end
        @test worst <= bound
    end
end

@testset "device compilation, typed IR, and launch independence" begin
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    backend = CUDA.CUDABackend()
    kernel = extension_module._natural128_fill_kernel!(backend)
    for (F, T) in NATURAL_128_TYPES
        packed_rng = device(F(0x785))
        destination = CUDA.CuArray{T}(undef, 1024)
        storage_type = extension_module._CUDANatural128Pack{T}
        packed_typed = KernelAbstractions.@ka_code_typed kernel(
            packed_rng,
            destination,
            Val(storage_type),
            ndrange = extension_module._CUDA_FILL_THREADS,
            workgroupsize = extension_module._CUDA_FILL_THREADS,
        )
        packed_typed_text = sprint(show, packed_typed)
        packed_llvm_text = sprint() do io
            CUDA.@device_code_llvm io = io kernel(
                packed_rng,
                destination,
                Val(storage_type);
                ndrange = extension_module._CUDA_FILL_THREADS,
                workgroupsize = extension_module._CUDA_FILL_THREADS,
            )
        end
        @test !occursin("UInt128", packed_typed_text)
        @test !occursin("BigInt", packed_typed_text)
        @test !occursin(r"\bi128\b", packed_llvm_text)
        packed_sass = sprint() do io
            CUDA.@device_code_sass io = io kernel(
                packed_rng,
                destination,
                Val(storage_type);
                ndrange = extension_module._CUDA_FILL_THREADS,
                workgroupsize = extension_module._CUDA_FILL_THREADS,
            )
        end
        @test occursin("STG.E.128", packed_sass)
    end

    bool_kernel = CUDA_EXT._bool_blocks_fill_kernel!(backend)
    for F in GENERATOR_TYPES
        bool_rng = device(F(0x787))
        plan = IR._device_fill_plan(backend, bool_rng, Val(:uniform), Bool)
        expected = F === Philox4x32 ? Val(:cooperative) : Val(:bool_blocks)
        @test plan[1] === expected
        plan[1] === Val(:bool_blocks) || continue
        packs_per_block = plan[2]
        packed = reinterpret(
            extension_module._CUDA_B8X16,
            CUDA.CuArray{Bool}(undef, Int(IR._block_bits(bool_rng))),
        )
        bool_typed = KernelAbstractions.@ka_code_typed bool_kernel(
            bool_rng,
            packed,
            packs_per_block,
            ndrange = 1,
            workgroupsize = 1,
        )
        bool_typed_text = sprint(show, bool_typed)
        bool_llvm_text = sprint() do io
            CUDA.@device_code_llvm io = io bool_kernel(
                bool_rng,
                packed,
                packs_per_block;
                ndrange = 1,
                workgroupsize = 1,
            )
        end
        @test !occursin("UInt128", bool_typed_text)
        @test !occursin("BigInt", bool_typed_text)
        @test !occursin(r"\bi128\b", bool_llvm_text)
        bool_sass = sprint() do io
            CUDA.@device_code_sass io = io bool_kernel(
                bool_rng,
                packed,
                packs_per_block;
                ndrange = 1,
                workgroupsize = 1,
            )
        end
        @test occursin("STG.E.128", bool_sass)
    end

    packed_rng = device(Philox4x32(0x785))
    float_plan = IR._device_fill_plan(backend, packed_rng, Val(:uniform), Float32)
    float_destination = CUDA.CuArray{Float32}(undef, IR._val_count(float_plan[2]))
    for stream_aligned in (Val(false), Val(true))
        _check_cooperative_kernel_code(
            backend,
            packed_rng,
            float_destination,
            Float32,
            float_plan,
            stream_aligned,
            check_store = stream_aligned === Val(true),
            storage_type = extension_module._CUDA_F32X4,
        )
    end

    for F in GENERATOR_TYPES
        F === Philox4x32 && continue
        for T in (Float32, Float64)
            rng = device(F(0x788))
            plan = IR._device_fill_plan(backend, rng, Val(:uniform), T)
            destination = CUDA.CuArray{T}(undef, IR._val_count(plan[2]))
            _check_cooperative_kernel_code(
                backend,
                rng,
                destination,
                T,
                plan,
                Val(false),
                check_store = true,
                storage_type = extension_module._packed_type(T),
            )
        end
    end

    for T in (UInt32, UInt64)
        rng = device(Threefry4x64(0x789))
        plan = IR._device_fill_plan(backend, rng, Val(:uniform), T)
        destination = CUDA.CuArray{T}(undef, IR._val_count(plan[2]))
        _check_cooperative_kernel_code(
            backend,
            rng,
            destination,
            T,
            plan,
            Val(false),
            check_store = true,
            storage_type = extension_module._packed_type(T),
        )
    end

    for T in (Float32, Float64)
        rng = device(Threefry4x32(0x788))
        plan = IR._device_fill_plan(backend, rng, IR._NormalCodec(rng.device), T)
        destination = CUDA.CuArray{T}(undef, IR._val_count(plan[2]))
        _check_cooperative_kernel_code(
            backend,
            rng,
            destination,
            T,
            plan,
            Val(false),
            check_store = true,
            codec = IR._NormalCodec(rng.device),
            storage_type = extension_module._packed_type(T),
        )
    end

    for F in GENERATOR_TYPES
        rng = device(F(0x123456))
        signed32 = CUDA.zeros(Int32, 3)
        signed64 = CUDA.zeros(Int64, 3)
        args = (
            CUDA.CuArray{UInt32}(undef, 6),
            CUDA.CuArray{Float32}(undef, 5),
            CUDA.CuArray{Float64}(undef, 5),
            CUDA.CuArray{UInt64}(undef, 2),
            signed32,
            signed64,
            rng,
        )
        CUDA.@sync CUDA.@cuda threads = 1 blocks = 1 _device_api_kernel!(args...)
        signature = Tuple{map(typeof, args)...}
        typed_text = sprint(show, CUDA.code_typed(_device_api_kernel!, signature))
        llvm_text = sprint(io -> CUDA.code_llvm(io, _device_api_kernel!, signature))
        @test !occursin("UInt128", typed_text)
        @test !occursin("BigInt", typed_text)
        @test !occursin(r"\bi128\b", llvm_text)

        continued_uniform, next_uniform = rand_next(rng, UInt32)
        @test Array(args[1]) == UInt32[
            rand(rng, UInt32),
            rand_at(rng, UInt32, 1),
            continued_uniform,
            rand(next_uniform, UInt32),
            rand(subrng(rng, UInt64(0x71)), UInt32),
            rand(splitrng(rng, Val(2))[2], UInt32),
        ]
        normal32 = Array(args[2])
        normal64 = Array(args[3])
        @test normal32[1] === normal32[2] === normal32[3]
        @test normal32[4] === normal32[5]
        @test normal64[1] === normal64[2] === normal64[3]
        @test normal64[4] === normal64[5]
        continued_unsigned32, next_unsigned32 = rand_next(rng, UInt32)
        @test Array(signed32) == Int32[
            reinterpret(Int32, rand(rng, UInt32)),
            reinterpret(Int32, continued_unsigned32),
            reinterpret(Int32, rand(next_unsigned32, UInt32)),
        ]
        continued_unsigned64, next_unsigned64 = rand_next(rng, UInt64)
        @test Array(signed64) == Int64[
            reinterpret(Int64, rand(rng, UInt64)),
            reinterpret(Int64, continued_unsigned64),
            reinterpret(Int64, rand(next_unsigned64, UInt64)),
        ]
        range = UInt64(0):UInt64(1):(UInt64(1)<<40)
        continued_range, next_range = rand_next(rng, range)
        @test Array(args[4]) == [continued_range, rand(next_range, range)]

        k64_values = CUDA.CuArray{UInt32}(undef, 2)
        k64_args = (k64_values, rng)
        CUDA.@sync CUDA.@cuda threads = 1 blocks = 1 _k64_range_kernel!(k64_args...)
        k64_signature = Tuple{map(typeof, k64_args)...}
        k64_typed_text = sprint(show, CUDA.code_typed(_k64_range_kernel!, k64_signature))
        k64_llvm_text = sprint(io -> CUDA.code_llvm(io, _k64_range_kernel!, k64_signature))
        @test !occursin("UInt128", k64_typed_text)
        @test !occursin("BigInt", k64_typed_text)
        @test !occursin(r"\bi128\b", k64_llvm_text)
        k64_value, k64_next = rand_next(rng, K64_RANGE)
        @test Array(k64_values) == [k64_value, rand(k64_next, K64_RANGE)]
    end

end

@testset "exponential CUDA scalar, array, fill, and IR smoke" begin
    exponential_cases = (
        (Philox4x32, Float32),
        (Philox4x32, Float64),
        ((F, Float64) for F in GENERATOR_TYPES if F !== Philox4x32)...,
    )
    for (F, T) in exponential_cases
        cpu_rng = F(0x123456)
        rng = device(cpu_rng)
        values = CUDA.CuArray{T}(undef, 5)
        raw = CUDA.CuArray{UInt64}(undef, 1)
        lattice = CUDA.CuArray{T}(undef, 2)
        args = (values, raw, lattice, rng)

        CUDA.@sync CUDA.@cuda threads = 1 blocks = 1 _exponential_api_kernel!(args...)
        signature = Tuple{map(typeof, args)...}
        typed_text = sprint(show, CUDA.code_typed(_exponential_api_kernel!, signature))
        llvm_text = sprint(io -> CUDA.code_llvm(io, _exponential_api_kernel!, signature))
        for forbidden in ("StatefulRNG", "Task", "RefValue", "BigInt", "UInt128")
            @test !occursin(forbidden, typed_text)
        end
        @test !occursin(r"\bi128\b", llvm_text)

        allocated = _check_array_draw(
            cpu_rng,
            rng,
            T,
            T,
            randexp,
            randexp_next,
            fills = ((randexp!, randexp_next!),),
            cpu_parity = false,
        )
        host_allocated = Array(allocated)
        @test Array(values) == [
            host_allocated[1],
            host_allocated[1],
            host_allocated[1],
            host_allocated[2],
            host_allocated[2],
        ]

        expected_raw = _exponential_raw(cpu_rng, T)
        @test only(Array(raw)) === expected_raw
        @test Tuple(Array(lattice)) === IR._exponential_lattice(T, expected_raw)

        last_rng = _last_draw_rng(rng, IR._exponential_bits(T))
        _, terminal = randexp_next(last_rng, T)
        @test terminal.position == _terminal(rng)
        terminal_array, terminal_array_next = randexp_next(last_rng, T, 1)
        @test terminal_array_next.position == terminal.position
        terminal_destination = similar(terminal_array)
        _, terminal_fill_next = randexp_next!(last_rng, terminal_destination)
        @test terminal_fill_next.position == terminal.position
        @test Array(terminal_destination) == Array(terminal_array)
        @test_throws StreamExhausted randexp(terminal, T)

        for operation in (randexp!, randexp_next!), source in (last_rng, terminal)
            failed = CUDA.fill(T(-1), 2)
            before = Array(failed)
            @test_throws StreamExhausted operation(source, failed)
            @test Array(failed) == before
        end

        empty = CUDA.CuArray{T}(undef, 0)
        for operation in (randexp!, randexp_next!)
            empty_profile = CUDA.@profile raw = true operation(terminal, empty)
            @test count(value -> !ismissing(value), empty_profile.device.grid) == 0
        end
        returned_empty, empty_next = randexp_next!(terminal, empty)
        @test returned_empty === empty
        @test empty_next.position == terminal.position
    end
end

@testset "CUDA exponential packed stores match grouped fallback" begin
    for F in GENERATOR_TYPES, T in (Float32, Float64)
        F === Philox4x32 && T === Float64 && continue
        base = device(F(0x78a))
        outputs_per_store = T === Float32 ? 4 : 2
        count = 16outputs_per_store
        for rng in (base, _positioned_at_bit(base, UInt64(9), UInt16(5)))
            packed = CUDA.CuArray{T}(undef, count)
            storage = CUDA.CuArray{T}(undef, count + 1)
            grouped = @view storage[2:end]

            _, packed_next = randexp_next!(rng, packed)
            _, grouped_next = randexp_next!(rng, grouped)
            @test isequal(Array(packed), Array(grouped))
            @test packed_next.position == grouped_next.position
        end
    end
end

@testset "mixed widths, capacity, terminal, and failed preflight" begin
    block_bits = (64, 128, 128, 256, 64, 128, 128, 256)
    capacity_exponents = (62, 71, 71, 136, 62, 71, 71, 136)
    for (F, expected_block_bits, capacity_exponent) in
        zip(GENERATOR_TYPES, block_bits, capacity_exponents)

        cpu_rng = F(0x123456)
        gpu_rng = device(cpu_rng)
        @test IR._block_bits(gpu_rng) == expected_block_bits
        block_index_bits = if gpu_rng.position isa IR._Position128
            128
        elseif IR._max_block(gpu_rng) == UInt64(0x00ffffffffffffff)
            56
        else
            64
        end
        @test block_index_bits + trailing_zeros(expected_block_bits) == capacity_exponent

        cross_cpu = _positioned_at_bit(cpu_rng, UInt64(9), UInt16(expected_block_bits - 1))
        cross_gpu = device(cross_cpu)
        cross_values, cross_next = rand_next(cross_gpu, UInt64, 2)
        expected_values, expected_next = rand_next(cross_cpu, UInt64, 2)
        @test Array(cross_values) == expected_values
        @test cross_next.position == expected_next.position

        if gpu_rng.position isa IR._Position128
            carry_position =
                IR._Position128(typemax(UInt64), UInt64(7), UInt16(expected_block_bits - 1))
            carry = IR._rebuild(gpu_rng, carry_position, gpu_rng.device)
            _, carry_next = rand_next(carry, UInt32, 1)
            @test carry_next.position == IR._Position128(UInt64(0), UInt64(8), UInt16(31))
        end

        range = UInt16(2):UInt16(3):UInt16(74)
        bools, gpu_1 = rand_next(gpu_rng, Bool, 3)
        u64s, gpu_2 = rand_next(gpu_1, UInt64, 2)
        normals, gpu_3 = randn_next(gpu_2, Float32, 5)
        ranges, gpu_4 = rand_next(gpu_3, range, 4)
        expected_bools, cpu_1 = rand_next(cpu_rng, Bool, 3)
        expected_u64s, cpu_2 = rand_next(cpu_1, UInt64, 2)
        _, cpu_3 = randn_next(cpu_2, Float32, 5)
        expected_ranges, cpu_4 = rand_next(cpu_3, range, 4)
        @test Array(bools) == expected_bools
        @test Array(u64s) == expected_u64s
        @test Array(ranges) == expected_ranges
        @test gpu_4.position == cpu_4.position

        last_rng = _last_draw_rng(gpu_rng, UInt16(32))
        final_value, exhausted = rand_next(last_rng, UInt32)
        @test exhausted.position.bit == IR._EXHAUSTED_BIT
        @test_throws StreamExhausted rand_next(exhausted, UInt32)
        final_array, array_exhausted = rand_next(last_rng, UInt32, 1)
        @test array_exhausted.position == exhausted.position
        @test Array(final_array) == [final_value]
        empty, empty_next = rand_next(exhausted, UInt32, 0)
        @test isempty(empty)
        @test empty_next.position == exhausted.position
        destination = CUDA.fill(UInt32(0xdeadbeef), 2)
        @test_throws StreamExhausted rand_next!(last_rng, destination)
        @test Array(destination) == fill(UInt32(0xdeadbeef), 2)

        for (capacity_range, width) in (
            (UInt16(2):UInt16(3):UInt16(74), UInt16(64)),
            (UInt64(7):UInt64(3):UInt64(0xfffffffffffffffd), UInt16(128)),
        )
            last_range = _last_draw_rng(gpu_rng, width)
            range_value, terminal = rand_next(last_range, capacity_range, 1)
            cpu_last = MLD.CPUDevice()(last_range)
            @test Array(range_value) == rand(cpu_last, capacity_range, 1)
            @test terminal.position == _terminal(gpu_rng)
            @test_throws StreamExhausted rand_next(terminal, capacity_range, 1)
            insufficient_position =
                IR._advance_position_unchecked(last_range, UInt64(1), UInt64(0))
            insufficient = IR._rebuild(last_range, insufficient_position, last_range.device)
            @test_throws StreamExhausted rand_next(insufficient, capacity_range, 1)
            @test insufficient.position == insufficient_position
        end
    end
end

@testset "context, empty validation, and active-device placement" begin
    before = CUDA.device()
    rng = device(Philox4x32(0x123456))
    @test CUDA.device() == before
    rand(rng, UInt32, 4)
    @test CUDA.device() == before

    empty = CUDA.CuArray{UInt32}(undef, 0)
    returned, empty_next = rand_next!(rng, empty)
    @test returned === empty
    @test empty_next.position == rng.position
    empty_profile = CUDA.@profile raw = true rand_next!(rng, empty)
    @test count(value -> !ismissing(value), empty_profile.device.grid) == 0

    empty_normal = CUDA.CuArray{Float32}(undef, 0)
    @test randn!(rng, empty_normal) === empty_normal
    returned_normal, empty_normal_next = randn_next!(rng, empty_normal)
    @test returned_normal === empty_normal
    @test empty_normal_next.position == rng.position
    for operation in (randn!, randn_next!)
        empty_normal_profile = CUDA.@profile raw = true operation(rng, empty_normal)
        @test count(value -> !ismissing(value), empty_normal_profile.device.grid) == 0
    end

    exhausted = IR._rebuild(rng, _terminal(rng), rng.device)
    for destination in (fill(UInt32(0xdeadbeef), 4), BitVector([true, false, true, false])),
        operation in (rand!, rand_next!)

        before_values = copy(destination)
        error = try
            operation(exhausted, destination)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test destination == before_values
    end

    wrong_normal = fill(-123.5f0, 4)
    for operation in (randn!, randn_next!)
        before_values = copy(wrong_normal)
        error = try
            operation(exhausted, wrong_normal)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test wrong_normal == before_values
    end

    nonempty = CUDA.CuArray{UInt32}(undef, 1024)
    rand_next!(rng, nonempty)
    CUDA.synchronize()
    nonempty_profile = CUDA.@profile raw = true rand_next!(rng, nonempty)
    @test count(value -> !ismissing(value), nonempty_profile.device.grid) > 0

    if length(devices) < 2
        @info "only one CUDA device; active-device switching capability-skipped"
        @test_skip false
    else
        first, second = devices[1:2]
        try
            CUDA.device!(first)
            second_bound = MLD.CUDADevice(second)(Philox4x32(0x123456))
            first_values = rand(second_bound, UInt32, 4)
            @test _device_id(first_values) == CUDA.deviceid(first)
            @test CUDA.device() == first

            CUDA.device!(second)
            first_bound = MLD.CUDADevice(first)(Philox4x32(0x123456))
            second_values = rand(first_bound, UInt32, 4)
            @test _device_id(second_values) == CUDA.deviceid(second)
            @test CUDA.device() == second
        finally
            CUDA.device!(before)
        end
        @test CUDA.device() == before
    end
end

@testset "CUDA unweighted sampling" begin
    for F in GENERATOR_TYPES
        cpu_rng = F(0x91a)
        gpu_rng = device(cpu_rng)
        host_population = reshape(collect(Int32(-11):Int32(12)), 4, 6)
        population = CuArray(host_population)

        values, next_gpu = randsample_next(gpu_rng, population, 19)
        expected, next_cpu = randsample_next(cpu_rng, host_population, 19)
        @test values isa CuArray{Int32,1}
        @test Array(values) == expected
        @test next_gpu.position == next_cpu.position
        @test _device_id(values) == CUDA.deviceid(primary)
        @test Array(randsample(gpu_rng, population, 7)) == expected[1:7]

        no_k, no_k_next = randsample_next(gpu_rng, population)
        expected_no_k, expected_no_k_next = randsample_next(cpu_rng, host_population)
        @test Array(no_k) == expected_no_k
        @test no_k_next.position == expected_no_k_next.position

        range = UInt64(0):(UInt64(1)<<32)
        range_values, range_next = randsample_next(gpu_rng, range, 5)
        expected_range, expected_range_next = randsample_next(cpu_rng, range, 5)
        @test Array(range_values) == expected_range
        @test range_next.position == expected_range_next.position

        iterable = DeviceAgnosticPopulation(collect(Int16(3):Int16(13)))
        iterable_values, iterable_next = randsample_next(gpu_rng, iterable, 9)
        expected_iterable, expected_iterable_next =
            randsample_next(cpu_rng, iterable.values, 9)
        @test iterable_values isa CuArray{Int16,1}
        @test Array(iterable_values) == expected_iterable
        @test iterable_next.position == expected_iterable_next.position
        @test iterable.starts[] == 1

        agnostic_array = DeviceAgnosticArray(collect(Int16(21):Int16(31)))
        array_values, array_next = randsample_next(gpu_rng, agnostic_array, 9)
        expected_array, expected_array_next =
            randsample_next(cpu_rng, agnostic_array.values, 9)
        @test array_values isa CuArray{Int16,1}
        @test Array(array_values) == expected_array
        @test array_next.position == expected_array_next.position
        @test agnostic_array.reads[] == length(agnostic_array)

        empty = CuArray{Int32}(undef, 0)
        empty_values, empty_next = randsample_next(gpu_rng, empty, 0)
        @test empty_values isa CuArray{Int32,1}
        @test isempty(empty_values)
        @test empty_next.position == gpu_rng.position

        error = try
            randsample(gpu_rng, vec(host_population), -1)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
    end

    limit_rng = device(Philox4x32(0x91b))
    limit_population = CUDA.CuArray(Int32[1])
    too_many = big(typemax(Int)) + 1
    @test_throws ArgumentError randsample(limit_rng, limit_population, too_many)
    @test_throws ArgumentError randsample_next(limit_rng, limit_population, too_many)

    audit_population = DeviceAgnosticPopulation(collect(Int32(1):Int32(13)))
    randsample(limit_rng, DeviceAgnosticPopulation(copy(audit_population.values)), 9)
    profile = CUDA.Profile.profile_internally(; concurrent = false, trace = true) do
        randsample(limit_rng, audit_population, 9)
    end
    @test audit_population.starts[] == 1
    events = _cuda_profile_events(profile)
    h2d_sizes = _event_sizes(profile, events.host_to_device)
    d2h_sizes = _event_sizes(profile, events.device_to_host)
    @test !isempty(events.kernels)
    @test sort(h2d_sizes) == [sizeof(Int32) * length(audit_population)]
    @test isempty(d2h_sizes)
end

@testset "GPU-bound scalar allocation" begin
    for F in GENERATOR_TYPES
        rng = device(F(0x123456))
        range = UInt32(2):UInt32(3):UInt32(74)
        _check_scalar_allocation(() -> rand_next(rng, UInt32))
        _check_scalar_allocation(() -> randn_next(rng, Float32))
        _check_scalar_allocation(() -> rand_next(rng, range))
        _check_scalar_allocation(() -> rand_at(rng, UInt64, 2))
        _check_scalar_allocation(() -> randn_at(rng, Float64, 2))
        _check_scalar_allocation(() -> subrng(rng, UInt64(0x71)))
        _check_scalar_allocation(() -> splitrng(rng, Val(2)))
    end
end

@testset "CUDA weighted sampling values, residency, and validation" begin
    cpu_population = reshape(collect(Int32(11):Int32(22)), 3, 4)
    cpu_weights = Float64[0, 1, 7, 2, 0, 4, 3, 9, 1, 5, 0, 6]
    gpu_population = CUDA.CuArray(cpu_population)
    gpu_weights = CUDA.CuArray(cpu_weights)

    for F in (Philox4x32, Threefry4x64)
        cpu_rng = F(0x791)
        gpu_rng = device(cpu_rng)
        expected, expected_next = randsample_next(cpu_rng, cpu_population, cpu_weights, 17)
        values, next_rng = randsample_next(gpu_rng, gpu_population, gpu_weights, 17)
        @test _device_id(values) == CUDA.deviceid(primary)
        @test Array(values) == expected
        @test next_rng.position == expected_next.position
        @test next_rng.device == gpu_rng.device
        @test Array(randsample(gpu_rng, gpu_population, gpu_weights, 9)) == expected[1:9]

        cursor = gpu_rng
        chained = Int32[]
        for _ = 1:17
            value, cursor = randsample_next(cursor, gpu_population, gpu_weights, 1)
            push!(chained, only(Array(value)))
        end
        @test chained == expected
        @test cursor.position == next_rng.position

        no_k, no_k_next = randsample_next(gpu_rng, gpu_population, gpu_weights)
        cpu_no_k, cpu_no_k_next = randsample_next(cpu_rng, cpu_population, cpu_weights)
        @test Array(no_k) == cpu_no_k
        @test no_k_next.position == cpu_no_k_next.position

        empty, empty_next = randsample_next(gpu_rng, gpu_population, gpu_weights, 0)
        @test empty isa CUDA.CuArray{Int32,1}
        @test isempty(empty)
        @test empty_next.position == gpu_rng.position

        last_rng = _last_draw_rng(gpu_rng, UInt16(53))
        final_value, terminal = randsample_next(last_rng, gpu_population, gpu_weights, 1)
        @test length(final_value) == 1
        @test terminal.position == _terminal(gpu_rng)
        @test_throws StreamExhausted randsample(last_rng, gpu_population, gpu_weights, 2)
        @test_throws StreamExhausted randsample_next(
            last_rng,
            gpu_population,
            gpu_weights,
            2,
        )
        @test last_rng.position.bit != IR._EXHAUSTED_BIT
    end

    range_population = UInt16(10):UInt16(3):UInt16(43)
    range_weights = CUDA.fill(1.0, length(range_population))
    range_rng = device(Philox4x32(0x792))
    range_values, range_next =
        randsample_next(range_rng, range_population, range_weights, 9)
    cpu_range_values, cpu_range_next = randsample_next(
        Philox4x32(0x792),
        range_population,
        ones(length(range_population)),
        9,
    )
    @test Array(range_values) == cpu_range_values
    @test range_next.position == cpu_range_next.position

    agnostic_weights = 1:length(range_population)
    agnostic_values, agnostic_next =
        randsample_next(range_rng, range_population, agnostic_weights, 9)
    cpu_agnostic_values, cpu_agnostic_next =
        randsample_next(Philox4x32(0x792), range_population, agnostic_weights, 9)
    @test Array(agnostic_values) == cpu_agnostic_values
    @test agnostic_next.position == cpu_agnostic_next.position
    @test_throws ArgumentError randsample(range_rng, UInt16(1):UInt16(3), -1:1, 2)

    audit_population = UInt16(1):UInt16(13)
    randsample(range_rng, audit_population, 1:13, 9)
    counted_weights = DeviceAgnosticWeights(collect(Float32, 1:13), Ref(0))
    profile = CUDA.Profile.profile_internally(; concurrent = false, trace = true) do
        randsample(range_rng, audit_population, counted_weights, 9)
    end
    @test counted_weights.reads[] == length(counted_weights)
    events = _cuda_profile_events(profile)
    h2d_sizes = _event_sizes(profile, events.host_to_device)
    d2h_sizes = _event_sizes(profile, events.device_to_host)
    @test !isempty(events.kernels)
    @test sort(h2d_sizes) == [sizeof(Float64) * length(counted_weights)]
    @test d2h_sizes == [1]

    fold_weights = CUDA.CuArray(
        Float64[Float64(0x000f5d057718d3b7), Float64(0x0010a2fa88e72c49), 1.0, 1.0],
    )
    fold_rng = device(Philox4x32(0x9750))
    @test Array(
        randsample(fold_rng, CUDA.CuArray(Int32[10, 20, 30, 40]), fold_weights, 1),
    ) == Int32[10]

    scan_destination = CUDA.CuArray{Int32}(undef, 1)
    scan_weights_host = Float64[0x1p53, fill(1.0, 1023)..., 2.0]
    scan_expected = similar(scan_weights_host)
    scan_total = 0.0
    for index in eachindex(scan_weights_host)
        scan_total += scan_weights_host[index]
        scan_expected[index] = scan_total
    end
    scan_weights = CUDA.CuArray(scan_weights_host)
    scan_population = CUDA.CuArray(Int32.(1:length(scan_weights_host)))
    _, _, scan_cumulative = IR._prepare_weight_scan(range_rng, scan_weights, false)
    @test reinterpret.(UInt64, Array(scan_cumulative)) ==
          reinterpret.(UInt64, scan_expected)
    IR._launch_weighted_scan!(
        KernelAbstractions.get_backend(scan_destination),
        scan_population,
        scan_cumulative,
        CUDA.CuArray(Float64[0x1p53]),
        scan_destination,
    )
    @test Array(scan_destination) == Int32[1025]

    invalid_scan_weights = copy(scan_weights_host)
    invalid_scan_weights[end] = NaN
    @test_throws ArgumentError randsample(
        range_rng,
        scan_population,
        CUDA.CuArray(invalid_scan_weights),
        1,
    )

    wrong_weights = copy(cpu_weights)
    error = try
        randsample(range_rng, gpu_population, wrong_weights, -1)
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError

    too_many = big(typemax(Int)) + 1
    @test_throws ArgumentError randsample(range_rng, gpu_population, gpu_weights, too_many)
    @test_throws ArgumentError randsample_next(
        range_rng,
        gpu_population,
        gpu_weights,
        too_many,
    )

    for invalid in (
        CUDA.CuArray([1.0, -1.0, 2.0]),
        CUDA.CuArray([1.0, Inf, 2.0]),
        CUDA.zeros(Float64, 3),
    )
        @test_throws ArgumentError randsample(
            range_rng,
            CUDA.CuArray(Int32[1, 2, 3]),
            invalid,
            2,
        )
    end
end

@testset "CUDA population destination fills" begin
    rng = device(Philox4x32(0x9767))
    population = CUDA.CuArray(Int32[11, 13, 17, 19])
    expected, after = randsample_next(rng, population, 33)
    storage = CUDA.CuArray{Int32}(undef, 34)
    destination = @view storage[2:end]

    returned, next_rng = randsample_next!(rng, population, destination)
    @test returned === destination
    CUDA.synchronize()
    @test Array(destination) == Array(expected)
    @test next_rng.position == after.position

    @test randsample!(rng, population, destination) === destination
    CUDA.synchronize()
    @test Array(destination) == Array(expected)

    aliased_population = CUDA.CuArray(Int32[23, 29, 31])
    before = Array(aliased_population)
    @test_throws ArgumentError randsample!(rng, aliased_population, aliased_population)
    @test Array(aliased_population) == before

    weighted_population = CUDA.CuArray(Float64.(1:33))
    weights = CUDA.CuArray(Float64.(1:33))
    weighted_expected, weighted_after =
        randsample_next(rng, weighted_population, weights, 33)
    weighted_returned, weighted_next =
        randsample_next!(rng, weighted_population, weights, weights)
    @test weighted_returned === weights
    CUDA.synchronize()
    @test Array(weights) == Array(weighted_expected)
    @test weighted_next.position == weighted_after.position
end

include("fixed_distributions.jl")
include("enzyme.jl")
