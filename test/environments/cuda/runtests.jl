using CUDA
using Distributions
using Enzyme
using PureRNGs
using MLDataDevices
using Random
using Test

const IR = PureRNGs
const MLD = MLDataDevices
const FAMILIES = (
    Philox2x32,
    Philox4x32,
    Philox2x64,
    Philox4x64,
    Threefry2x32,
    Threefry4x32,
    Threefry2x64,
    Threefry4x64,
)
const UNIFORM_TYPES = (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
const NORMAL_TYPES = (Float32, Float64)
const RANGE_TYPES = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)

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

_range(::Type{T}) where {T<:Signed} = T(-31):T(3):T(41)
_range(::Type{T}) where {T<:Unsigned} = T(2):T(3):T(74)

function _chain(rng, draw, count::Int, ::Type{T}) where {T}
    values = Vector{T}(undef, count)
    for index in eachindex(values)
        rng, values[index] = draw(rng)
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

    next_gpu, continued = next_draw(gpu_rng, argument, 19)
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
        fill_next, returned = next_fill(gpu_rng, destination)
        @test returned === destination
        @test fill_next.position == next_gpu.position
        @test fill_next.device == gpu_rng.device
        serial = similar(values)
        serial_next, _ = next_fill(gpu_rng, serial; threaded = false)
        @test isequal(Array(serial), Array(values))
        @test serial_next.position == next_gpu.position
        @test serial_next.device == gpu_rng.device
    end

    empty_next, empty = next_draw(gpu_rng, argument, 0)
    @test empty isa CUDA.CuArray{T,1}
    @test isempty(empty)
    @test empty_next.position == gpu_rng.position
    return values
end

function _address_kernel!(destination, rng, offset)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index <= length(destination) &&
        (@inbounds destination[index] = randat(rng, eltype(destination), offset + index))
    return
end

function _normal_address_kernel!(destination, rng)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index <= length(destination) &&
        (@inbounds destination[index] = randnat(rng, eltype(destination), index))
    return
end

@inline function _exponential_raw(rng, ::Type{Float32})
    return IR._extract_bits_unchecked(
        rng,
        IR.FAMILY_EXP,
        IR._position_block(rng.position),
        rng.position.bit,
        Val(24),
    )
end

@inline function _exponential_raw(rng, ::Type{Float64})
    return IR._extract_bits_unchecked(
        rng,
        IR.FAMILY_EXP,
        IR._position_block(rng.position),
        rng.position.bit,
        Val(53),
    )
end

function _exponential_api_kernel!(values, raw, lattice, rng)
    if CUDA.threadIdx().x == 1
        T = eltype(values)
        next_rng, continued = randexp_next(rng, T)
        k = _exponential_raw(rng, T)
        u, v = IR._exponential_lattice(T, k)
        @inbounds begin
            values[1] = randexp(rng, T)
            values[2] = continued
            values[3] = randexpat(rng, T, 1)
            values[4] = randexp(next_rng, T)
            values[5] = randexpat(rng, T, 2)
            raw[1] = k
            lattice[1] = u
            lattice[2] = v
        end
    end
    return
end

function _device_api_kernel!(uniform, normal32, normal64, ranges, signed32, signed64, rng)
    if CUDA.threadIdx().x == 1
        next_uniform, continued_uniform = rand_next(rng, UInt32)
        next_signed32, continued_signed32 = rand_next(rng, Int32)
        next_signed64, continued_signed64 = rand_next(rng, Int64)
        next_normal32, continued_normal32 = randn_next(rng, Float32)
        next_normal64, continued_normal64 = randn_next(rng, Float64)
        range = UInt64(0):UInt64(1):(UInt64(1)<<40)
        next_range, continued_range = rand_next(rng, range)
        child = subrng(rng, UInt64(0x71))
        children = splitrng(rng, Val(2))
        @inbounds begin
            uniform[1] = rand(rng, UInt32)
            uniform[2] = randat(rng, UInt32, 1)
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
            normal32[2] = randnat(rng, Float32, 1)
            normal32[3] = continued_normal32
            normal32[4] = randn(next_normal32, Float32)
            normal32[5] = randnat(rng, Float32, 2)
            normal64[1] = randn(rng, Float64)
            normal64[2] = randnat(rng, Float64, 1)
            normal64[3] = continued_normal64
            normal64[4] = randn(next_normal64, Float64)
            normal64[5] = randnat(rng, Float64, 2)
            ranges[1] = continued_range
            ranges[2] = rand(next_range, range)
        end
    end
    return
end

function _k64_range_kernel!(destination, rng)
    if CUDA.threadIdx().x == 1
        next_rng, value = rand_next(rng, K64_RANGE)
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

@inline function _write_coefficients!(destination, coefficients)
    offset = _write_group!(destination, 0, coefficients[1])
    offset = _write_group!(destination, offset, coefficients[2])
    offset = _write_group!(destination, offset, coefficients[3])
    offset = _write_group!(destination, offset, coefficients[4])
    offset = _write_group!(destination, offset, coefficients[5])
    _write_group!(destination, offset, coefficients[6])
    return
end

function _as241_kernel!(coefficients32, probes32, coefficients64, probes64)
    if CUDA.threadIdx().x == 1
        _write_coefficients!(coefficients32, IR._as241_coefficients(Float32))
        _write_coefficients!(coefficients64, IR._as241_coefficients(Float64))
        @inbounds begin
            probes32[1] = IR._normal_midpoint(Float32, UInt64(0))
            probes32[2] = IR._normal_midpoint(Float32, (UInt64(1)<<23) - UInt64(1))
            probes32[3] = IR._as241(0.5f0)
            probes32[4] = IR._as241(0.95f0)
            probes32[5] = IR._as241(1.0f-12)
            probes64[1] = IR._normal_midpoint(Float64, UInt64(0))
            probes64[2] = IR._normal_midpoint(Float64, (UInt64(1)<<52) - UInt64(1))
            probes64[3] = IR._as241(0.5)
            probes64[4] = IR._as241(0.95)
            probes64[5] = IR._as241(1.0e-20)
        end
    end
    return
end

_flatten(coefficients) = vcat((collect(group) for group in coefficients)...)

function _check_scalar(call, ::Type{T}) where {T}
    @test @inferred(call()) isa T
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

_terminal(rng::IR._Position64Family) = IR._terminal64(IR._max_block(rng))
_terminal(::IR._Position128Family) = IR._terminal128()

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
    allocated_next, allocated = next_draw(rng, T, count)
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
    filled_next, _ = next_fill(rng, destination)
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

function _check_cooperative_kernel_code(
    backend,
    rng,
    packed,
    ::Type{T},
    plan,
    stream_aligned;
    check_store::Bool,
) where {T}
    kernel = IR._fill_cooperative_kernel!(backend)
    workgroup = IR._fill_group_size(plan[3])
    block_width = Val(Int(IR._block_bits(rng)))
    typed = IR.KernelAbstractions.@ka_code_typed kernel(
        rng,
        packed,
        T,
        Val(IR._draw_bits(T)),
        block_width,
        plan[2],
        plan[3],
        plan[4],
        stream_aligned,
        IR.FAMILY_BITS,
        Val(:uniform),
        ndrange = workgroup,
        workgroupsize = workgroup,
    )
    typed_text = sprint(show, typed)
    llvm = sprint() do io
        CUDA.@device_code_llvm io = io kernel(
            rng,
            packed,
            T,
            Val(IR._draw_bits(T)),
            block_width,
            plan[2],
            plan[3],
            plan[4],
            stream_aligned,
            IR.FAMILY_BITS,
            Val(:uniform);
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
                T,
                Val(IR._draw_bits(T)),
                block_width,
                plan[2],
                plan[3],
                plan[4],
                stream_aligned,
                IR.FAMILY_BITS,
                Val(:uniform);
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
    allocated_next, allocated = rand_next(rng, range, count)
    @test (Array(allocated), allocated_next.position) == (expected, expected_next.position)
end

function _device_kernel_events(call)
    call()
    CUDA.synchronize()
    profile = CUDA.@profile raw = true begin
        call()
        CUDA.synchronize()
    end
    return count(value -> !ismissing(value), profile.device.grid)
end

devices = collect(CUDA.devices())
primary = first(devices)
CUDA.device!(primary)
device = MLD.CUDADevice(primary)

@testset "CUDA extension, binding, and capabilities" begin
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    @test extension_module !== nothing
    for F in FAMILIES
        cpu_rng = F(0x123456)
        gpu_rng = device(cpu_rng)
        @test isbits(gpu_rng)
        @test (gpu_rng.key, gpu_rng.position) == (cpu_rng.key, cpu_rng.position)
        @test gpu_rng.device === IR._CUDA_BACKEND
        @test MLD.get_device_type(gpu_rng.device) === MLD.CUDADevice
        @test which(device, Tuple{typeof(cpu_rng)}).module === IR
        @test MLD.CUDADevice()(cpu_rng).device === gpu_rng.device

        range = UInt32(1):UInt32(3)
        @test which(rand, (typeof(gpu_rng), Type{UInt32}, Int)).module === IR
        @test which(rand_next, (typeof(gpu_rng), Type{UInt32}, Int)).module === IR
        @test which(randn, (typeof(gpu_rng), Type{Float32}, Int)).module === IR
        @test which(randn_next, (typeof(gpu_rng), Type{Float32}, Int)).module === IR
        @test which(rand, (typeof(gpu_rng), typeof(range), Int)).module === IR
        @test which(rand_next, (typeof(gpu_rng), typeof(range), Int)).module === IR
        @test which(rand_next, (typeof(gpu_rng), Int)).module === IR
        @test which(randn_next, (typeof(gpu_rng), Int)).module === IR

        advanced, _ = rand_next(gpu_rng, UInt64)
        child = subrng(advanced, UInt64(0x71))
        children = splitrng(advanced, Val(2))
        dynamic_children = splitrng(advanced, 2)
        @test_throws ArgumentError splitrng(advanced, big(typemax(Int)) + 1)
        @test advanced.position != gpu_rng.position
        @test child.key == subrng(gpu_rng, UInt64(0x71)).key
        @test getfield.(children, :key) == getfield.(splitrng(gpu_rng, Val(2)), :key)
        @test Tuple(dynamic_children) == children
        @test child.position == cpu_rng.position
        @test advanced.device == child.device == gpu_rng.device
        @test all(
            rng -> rng.position == cpu_rng.position && rng.device == gpu_rng.device,
            (children..., dynamic_children...),
        )

    end
end

@testset "CUDA Bool scalar and array agreement" begin
    for F in FAMILIES
        rng = device(F(0x123456))
        pure = rand(rng, Bool)
        next_rng, continued = rand_next(rng, Bool)
        array = rand(rng, Bool, 1)
        @test pure === continued === only(Array(array))
        @test next_rng.position != rng.position
    end
end

@testset "CUDA owns packed fill plans" begin
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    backend = CUDA.CUDABackend()
    for F in FAMILIES, T in UNIFORM_TYPES
        rng = device(F(0x123456))
        plan = IR._device_uniform_fill_plan(backend, rng, T)
        cooperative = IR._cooperative_uniform_fill(rng, T)
        packed_float = F !== Philox4x32 && T in (Float32, Float64)
        expected_kind =
            cooperative !== nothing || packed_float ? Val(:cooperative) :
            T === Bool ? Val(:bool_blocks) :
            (F, T) in NATURAL_128_TYPES ? Val(:natural128_packed) : Val(:grouped)
        dispatch =
            which(IR._device_uniform_fill_plan, Tuple{typeof(backend),typeof(rng),Type{T}})
        @test (dispatch.module, plan[1]) === (extension_module, expected_kind)
        if plan[1] === Val(:cooperative)
            if packed_float
                @test plan[4] === (T === Float32 ? Val(4) : Val(2))
            else
                T === Float32 ? @test(plan[4] === Val(4)) : @test(length(plan) == 3)
            end
        end
    end

    for F in FAMILIES, T in NORMAL_TYPES
        rng = device(F(0x123456))
        plan = IR._device_normal_fill_plan(backend, rng, T)
        dispatch =
            which(IR._device_normal_fill_plan, Tuple{typeof(backend),typeof(rng),Type{T}})
        expected_kind = F === Philox4x32 ? Val(:cooperative) : Val(:grouped)
        @test (dispatch.module, plan[1]) === (extension_module, expected_kind)
        plan[1] === Val(:cooperative) && @test(length(plan) == 3)
    end

    for F in FAMILIES, T in (Float32, Float64)
        rng = device(F(0x123456))
        plan = IR._transformed_fill_plan(rng.device, backend, rng, T)
        uniform_plan = IR._device_uniform_fill_plan(backend, rng, T)
        dispatch = which(
            IR._transformed_fill_plan,
            Tuple{typeof(rng.device),typeof(backend),typeof(rng),Type{T}},
        )
        @test dispatch.module === extension_module
        @test plan == uniform_plan
    end

    for F in FAMILIES,
        (range, expected_kind) in ((K64_RANGE, Val(:grouped)), (K128_RANGE, nothing))

        rng = device(F(0x123456))
        span = IR._range_span(range)
        plan = IR._device_range_fill_plan(backend, rng, span)
        dispatch =
            which(IR._device_range_fill_plan, Tuple{typeof(backend),typeof(rng),UInt64})
        actual_kind = plan === nothing ? nothing : plan[1]
        @test (dispatch.module, actual_kind) === (extension_module, expected_kind)
    end
end

@testset "CUDA Bool packed paths preserve every family stream" begin
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    backend = CUDA.CUDABackend()
    for F in FAMILIES
        rng = device(F(0x787))
        block_bits = Int(IR._block_bits(rng))
        packs_per_block = extension_module._bool_packs_per_block(rng)
        plan = IR._device_uniform_fill_plan(backend, rng, Bool)
        cooperative = IR._cooperative_uniform_fill(rng, Bool)
        expected_plan =
            cooperative === nothing ? (Val(:bool_blocks), packs_per_block) :
            (Val(:cooperative), cooperative...)
        @test plan == expected_plan

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
        for (first, eligible) in ((2, false), (17, true))
            view = @view storage[first:(first+block_bits-1)]
            plan[1] === Val(:bool_blocks) &&
                @test(extension_module._aligned_bool_block_fill(rng, view) === eligible)
            filled_next, returned = rand_next!(rng, view)
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
end

@testset "CUDA Philox4x32 Float32 vector stores preserve the dense stream" begin
    rng = device(Philox4x32(0x786))
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    outputs = IR._device_uniform_fill_plan(CUDA.CUDABackend(), rng, Float32)[2]

    _check_public_packed_fill(rng, Float32, 4096, rand_next, rand_next!)
    positioned = _positioned_at_bit(rng, UInt64(9), UInt16(0))
    _check_public_packed_fill(positioned, Float32, 4096, rand_next, rand_next!)

    offset = _positioned_at_bit(rng, UInt64(9), UInt16(5))
    _check_public_packed_fill(offset, Float32, 2048, rand_next, rand_next!)
    _check_public_packed_fill(rng, Float32, 2052, rand_next, rand_next!)

    storage = CUDA.CuArray{Float32}(undef, 2052)
    expected_next, expected =
        _chain(rng, current -> rand_next(current, Float32), 2048, Float32)
    full_group = @view storage[1:2048]
    @test extension_module._stream_aligned_philox4x32_f32_fill(rng, full_group, outputs)
    @test !extension_module._stream_aligned_philox4x32_f32_fill(offset, full_group, outputs)
    for (first, layout_eligible) in ((1, true), (2, false))
        view = @view storage[first:(first+2047)]
        @test extension_module._packed_float_layout(view, Float32, Val(4)) ===
              layout_eligible
        filled_next, returned = rand_next!(rng, view)
        @test returned === view
        @test Array(view) == expected
        @test filled_next.position == expected_next.position
    end

    terminal_rng = _positioned_at_bit(rng, typemax(UInt64) - UInt64(383), UInt16(0))
    terminal = _check_public_packed_fill(terminal_rng, Float32, 2048, rand_next, rand_next!)
    @test terminal.position == _terminal(rng)
end

@testset "CUDA non-Philox float vector stores preserve the dense stream" begin
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    backend = CUDA.CUDABackend()
    for F in FAMILIES
        F === Philox4x32 && continue
        for T in (Float32, Float64)
            rng = device(F(0x788))
            plan = IR._device_uniform_fill_plan(backend, rng, T)
            outputs = IR._fill_group_size(plan[2])
            outputs_per_store = IR._fill_group_size(plan[4])
            @test plan[1] === Val(:cooperative)
            @test outputs_per_store == (T === Float32 ? 4 : 2)

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
            for (first, eligible) in ((1, true), (2, false))
                view = @view storage[first:(first+outputs-1)]
                @test extension_module._packed_float_layout(view, T, plan[4]) === eligible
                filled_next, returned = rand_next!(rng, view)
                @test returned === view
                @test Array(view) == expected
                @test filled_next.position == expected_next.position
            end

            packed_count = 8outputs_per_store
            packed_next, packed_expected =
                _chain(rng, current -> rand_next(current, T), packed_count, T)
            matrix = CUDA.CuArray{T}(undef, outputs_per_store, 8)
            @test extension_module._packed_float_layout(matrix, T, plan[4])
            matrix_next, returned_matrix = rand_next!(rng, matrix)
            @test returned_matrix === matrix
            @test vec(Array(matrix)) == packed_expected
            @test matrix_next.position == packed_next.position

            strided_storage = CUDA.CuArray{T}(undef, 2packed_count)
            strided = @view strided_storage[1:2:(2packed_count)]
            @test !extension_module._packed_float_layout(strided, T, plan[4])
            strided_next, returned_strided = rand_next!(rng, strided)
            @test returned_strided === strided
            @test Array(strided) == packed_expected
            @test strided_next.position == packed_next.position

            last_pack_rng =
                _last_draw_rng(rng, UInt16(outputs_per_store) * IR._draw_bits(T))
            unchanged = CUDA.fill(T(0.25), 2outputs_per_store)
            @test_throws ArgumentError rand_next!(last_pack_rng, unchanged)
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
    count = 4096
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    backend = CUDA.CUDABackend()
    kernel = extension_module._natural128_packed_kernel!(backend)

    for (F, T) in NATURAL_128_TYPES
        rng = device(F(0x784))
        outputs_per_pack = extension_module._CUDA_FILL_ALIGNMENT ÷ sizeof(T)
        expected_next, expected = _chain(rng, current -> rand_next(current, T), count, T)

        allocated_next, allocated = rand_next(rng, T, 64, 64)
        @test Array(allocated) == reshape(expected, 64, 64)
        @test allocated_next.position == expected_next.position

        direct_count = 512outputs_per_pack
        direct = CUDA.CuArray{T}(undef, direct_count)
        @test extension_module._aligned_natural128_fill(rng, direct, T)
        packed = reinterpret(extension_module._CUDANatural128Pack{T}, direct)
        kernel(
            rng,
            packed;
            ndrange = extension_module._CUDA_FILL_THREADS,
            workgroupsize = extension_module._CUDA_FILL_THREADS,
        )
        IR.KernelAbstractions.synchronize(backend)
        @test Array(direct) == expected[1:direct_count]

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
        for (first, eligible) in ((2, false), (aligned_first, true))
            view = @view storage[first:(first+63)]
            @test extension_module._aligned_natural128_fill(rng, view, T) === eligible
            fallback_next, returned = rand_next!(rng, view)
            @test returned === view
            @test Array(view) == expected[1:64]
            @test fallback_next.position == expected_fallback_next.position
        end

        terminal_rng = _positioned_at_bit(rng, typemax(UInt64), UInt16(0))
        terminal_expected, terminal_values =
            _chain(terminal_rng, current -> rand_next(current, T), outputs_per_pack, T)
        terminal_destination = CUDA.fill(zero(T), outputs_per_pack)
        terminal_next, _ = rand_next!(terminal_rng, terminal_destination)
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
        plan =
            next_draw === rand_next ?
            IR._device_uniform_fill_plan(CUDA.CUDABackend(), rng, T) :
            IR._device_normal_fill_plan(CUDA.CUDABackend(), rng, T)
        outputs = IR._fill_group_size(plan[2])
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

@testset "grouped public fallbacks cover every remaining family and type" begin
    for F in FAMILIES, T in UNIFORM_TYPES
        F === Philox4x32 && T in COOPERATIVE_UNIFORM_TYPES && continue
        rng = device(F(0x782))
        block = rng.position isa IR._Position128 ? typemax(UInt64) : UInt64(9)
        positioned = _positioned_at_bit(rng, block, IR._block_bits(rng) - UInt16(5))
        _check_public_packed_fill(positioned, T, 9, rand_next, rand_next!)
    end

    for F in FAMILIES, T in NORMAL_TYPES
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

@testset "aligned public integer fills preserve packed results" begin
    for F in FAMILIES, T in (UInt32, UInt64)
        rng = _positioned_at_bit(device(F(0x784)), UInt64(7), UInt16(0))
        _check_public_packed_fill(rng, T, 9, rand_next, rand_next!)
    end
end

@testset "public K=64 grouped and K=128 generic ranges" begin
    for F in FAMILIES, range in (K64_RANGE, K128_RANGE)
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
    for F in FAMILIES, T in UNIFORM_TYPES
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
        serial, parallel = similar(values), similar(values)
        CUDA.@sync CUDA.@cuda threads = 1 blocks = 19 _address_kernel!(serial, gpu_rng, 0)
        CUDA.@sync CUDA.@cuda threads = 19 blocks = 1 _address_kernel!(parallel, gpu_rng, 0)
        @test Array(serial) == Array(values) == Array(parallel)
    end

    for F in FAMILIES, T in NORMAL_TYPES
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

    for F in FAMILIES, T in RANGE_TYPES
        cpu_rng = F(0x123456)
        _check_array_draw(cpu_rng, device(cpu_rng), _range(T), T, rand, rand_next)
    end
    wide = UInt64(0):UInt64(1):(UInt64(1)<<40)
    for F in FAMILIES
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
        (Float32, result32, (0.5f0, 0.95f0, 1.0f-12)),
        (Float64, result64, (0.5, 0.95, 1.0e-20)),
    )
        @test result[3] === zero(T)
        for index = 2:3
            @test isapprox(result[index+2], IR._as241(inputs[index]); rtol = 16eps(T))
        end
    end
end

@testset "device compilation, typed IR, and launch independence" begin
    extension_module = Base.get_extension(IR, :PureRNGsCUDAExt)
    backend = CUDA.CUDABackend()
    kernel = extension_module._natural128_packed_kernel!(backend)
    for (F, T) in NATURAL_128_TYPES
        packed_rng = device(F(0x785))
        packed = reinterpret(
            extension_module._CUDANatural128Pack{T},
            CUDA.CuArray{T}(undef, 1024),
        )
        packed_typed = IR.KernelAbstractions.@ka_code_typed kernel(
            packed_rng,
            packed,
            ndrange = extension_module._CUDA_FILL_THREADS,
            workgroupsize = extension_module._CUDA_FILL_THREADS,
        )
        packed_typed_text = sprint(show, packed_typed)
        packed_llvm_text = sprint() do io
            CUDA.@device_code_llvm io = io kernel(
                packed_rng,
                packed;
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
                packed;
                ndrange = extension_module._CUDA_FILL_THREADS,
                workgroupsize = extension_module._CUDA_FILL_THREADS,
            )
        end
        @test occursin("STG.E.128", packed_sass)
    end

    bool_kernel = IR._uniform_fill_bool_blocks_kernel!(backend)
    for F in FAMILIES
        bool_rng = device(F(0x787))
        plan = IR._device_uniform_fill_plan(backend, bool_rng, Bool)
        plan[1] === Val(:bool_blocks) || continue
        packs_per_block = plan[2]
        packed = reinterpret(
            NTuple{16,VecElement{Bool}},
            CUDA.CuArray{Bool}(undef, Int(IR._block_bits(bool_rng))),
        )
        bool_typed = IR.KernelAbstractions.@ka_code_typed bool_kernel(
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
    float_plan = IR._device_uniform_fill_plan(backend, packed_rng, Float32)
    float_packed = reinterpret(
        extension_module._CUDA_F32X4,
        CUDA.CuArray{Float32}(undef, IR._fill_group_size(float_plan[2])),
    )
    for stream_aligned in (Val(false), Val(true))
        _check_cooperative_kernel_code(
            backend,
            packed_rng,
            float_packed,
            Float32,
            float_plan,
            stream_aligned,
            check_store = stream_aligned === Val(true),
        )
    end

    for F in FAMILIES
        F === Philox4x32 && continue
        for T in (Float32, Float64)
            rng = device(F(0x788))
            plan = IR._device_uniform_fill_plan(backend, rng, T)
            packed = reinterpret(
                extension_module._packed_float_type(T),
                CUDA.CuArray{T}(undef, IR._fill_group_size(plan[2])),
            )
            _check_cooperative_kernel_code(
                backend,
                rng,
                packed,
                T,
                plan,
                Val(false),
                check_store = true,
            )
        end
    end

    for F in FAMILIES
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

        next_uniform, continued_uniform = rand_next(rng, UInt32)
        @test Array(args[1]) == UInt32[
            rand(rng, UInt32),
            randat(rng, UInt32, 1),
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
        next_unsigned32, continued_unsigned32 = rand_next(rng, UInt32)
        @test Array(signed32) == Int32[
            reinterpret(Int32, rand(rng, UInt32)),
            reinterpret(Int32, continued_unsigned32),
            reinterpret(Int32, rand(next_unsigned32, UInt32)),
        ]
        next_unsigned64, continued_unsigned64 = rand_next(rng, UInt64)
        @test Array(signed64) == Int64[
            reinterpret(Int64, rand(rng, UInt64)),
            reinterpret(Int64, continued_unsigned64),
            reinterpret(Int64, rand(next_unsigned64, UInt64)),
        ]
        range = UInt64(0):UInt64(1):(UInt64(1)<<40)
        next_range, continued_range = rand_next(rng, range)
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
        k64_next, k64_value = rand_next(rng, K64_RANGE)
        @test Array(k64_values) == [k64_value, rand(k64_next, K64_RANGE)]
    end

end

@testset "exponential CUDA scalar, array, fill, and IR smoke" begin
    for F in FAMILIES, T in (Float32, Float64)
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
        terminal, _ = randexp_next(last_rng, T)
        @test terminal.position == _terminal(rng)
        terminal_array_next, terminal_array = randexp_next(last_rng, T, 1)
        @test terminal_array_next.position == terminal.position
        terminal_destination = similar(terminal_array)
        terminal_fill_next, _ = randexp_next!(last_rng, terminal_destination)
        @test terminal_fill_next.position == terminal.position
        @test Array(terminal_destination) == Array(terminal_array)
        @test_throws ArgumentError randexp(terminal, T)

        for operation in (randexp!, randexp_next!), source in (last_rng, terminal)
            failed = CUDA.fill(T(-1), 2)
            before = Array(failed)
            @test_throws ArgumentError operation(source, failed)
            @test Array(failed) == before
        end

        empty = CUDA.CuArray{T}(undef, 0)
        for operation in (randexp!, randexp_next!)
            empty_profile = CUDA.@profile raw = true operation(terminal, empty)
            @test count(value -> !ismissing(value), empty_profile.device.grid) == 0
        end
        empty_next, returned_empty = randexp_next!(terminal, empty)
        @test returned_empty === empty
        @test empty_next.position == terminal.position
    end
end

@testset "mixed widths, capacity, terminal, and failed preflight" begin
    @test fieldtypes(IR._Position64) === (UInt64, UInt16)
    @test fieldtypes(IR._Position128) === (UInt64, UInt64, UInt16)
    block_bits = (64, 128, 128, 256, 64, 128, 128, 256)
    capacity_exponents = (62, 71, 71, 136, 62, 71, 71, 136)
    for (F, expected_block_bits, capacity_exponent) in
        zip(FAMILIES, block_bits, capacity_exponents)

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

        for bit in (UInt16(expected_block_bits), typemax(UInt16) - UInt16(1))
            invalid = _positioned_at_bit(gpu_rng, UInt64(0), bit)
            @test_throws ArgumentError IR._reserve(invalid, UInt64(0), UInt64(0))
        end
        if gpu_rng.position isa IR._Position64 && IR._max_block(gpu_rng) != typemax(UInt64)
            invalid =
                _positioned_at_bit(gpu_rng, IR._max_block(gpu_rng) + UInt64(1), UInt16(0))
            @test_throws ArgumentError IR._reserve(invalid, UInt64(0), UInt64(0))
        end

        cross_cpu = _positioned_at_bit(cpu_rng, UInt64(9), UInt16(expected_block_bits - 1))
        cross_gpu = device(cross_cpu)
        cross_next, cross_values = rand_next(cross_gpu, UInt64, 2)
        expected_next, expected_values = rand_next(cross_cpu, UInt64, 2)
        @test Array(cross_values) == expected_values
        @test cross_next.position == expected_next.position

        if gpu_rng.position isa IR._Position128
            carry_position =
                IR._Position128(typemax(UInt64), UInt64(7), UInt16(expected_block_bits - 1))
            carry = IR._rebuild(gpu_rng, carry_position, gpu_rng.device)
            carry_next, _ = rand_next(carry, UInt32, 1)
            @test carry_next.position == IR._Position128(UInt64(0), UInt64(8), UInt16(31))
        end

        range = UInt16(2):UInt16(3):UInt16(74)
        gpu_1, bools = rand_next(gpu_rng, Bool, 3)
        gpu_2, u64s = rand_next(gpu_1, UInt64, 2)
        gpu_3, normals = randn_next(gpu_2, Float32, 5)
        gpu_4, ranges = rand_next(gpu_3, range, 4)
        cpu_1, expected_bools = rand_next(cpu_rng, Bool, 3)
        cpu_2, expected_u64s = rand_next(cpu_1, UInt64, 2)
        cpu_3, _ = randn_next(cpu_2, Float32, 5)
        cpu_4, expected_ranges = rand_next(cpu_3, range, 4)
        @test Array(bools) == expected_bools
        @test Array(u64s) == expected_u64s
        @test normals isa CUDA.CuArray{Float32,1}
        @test Array(ranges) == expected_ranges
        @test gpu_4.position == cpu_4.position

        last_rng = _last_draw_rng(gpu_rng, UInt16(32))
        exhausted, final_value = rand_next(last_rng, UInt32)
        @test exhausted.position.bit == IR._EXHAUSTED_BIT
        @test_throws ArgumentError rand_next(exhausted, UInt32)
        array_exhausted, final_array = rand_next(last_rng, UInt32, 1)
        @test array_exhausted.position == exhausted.position
        @test Array(final_array) == [final_value]
        empty_next, empty = rand_next(exhausted, UInt32, 0)
        @test isempty(empty)
        @test empty_next.position == exhausted.position
        destination = CUDA.fill(UInt32(0xdeadbeef), 2)
        @test_throws ArgumentError rand_next!(last_rng, destination)
        @test Array(destination) == fill(UInt32(0xdeadbeef), 2)

        for (capacity_range, width) in (
            (UInt16(2):UInt16(3):UInt16(74), UInt16(64)),
            (UInt64(7):UInt64(3):UInt64(0xfffffffffffffffd), UInt16(128)),
        )
            last_range = _last_draw_rng(gpu_rng, width)
            terminal, range_value = rand_next(last_range, capacity_range, 1)
            cpu_last = MLD.CPUDevice()(last_range)
            @test Array(range_value) == rand(cpu_last, capacity_range, 1)
            @test terminal.position == _terminal(gpu_rng)
            @test_throws ArgumentError rand_next(terminal, capacity_range, 1)
            insufficient_position =
                IR._advance_position_unchecked(last_range, UInt64(1), UInt64(0))
            insufficient = IR._rebuild(last_range, insufficient_position, last_range.device)
            @test_throws ArgumentError rand_next(insufficient, capacity_range, 1)
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
    @test_throws TypeError rand_next!(rng, empty; threaded = 1)
    empty_next, returned = rand_next!(rng, empty)
    @test returned === empty
    @test empty_next.position == rng.position
    empty_profile = CUDA.@profile raw = true rand_next!(rng, empty)
    @test count(value -> !ismissing(value), empty_profile.device.grid) == 0

    empty_normal = CUDA.CuArray{Float32}(undef, 0)
    @test_throws TypeError randn_next!(rng, empty_normal; threaded = 1)
    @test randn!(rng, empty_normal) === empty_normal
    empty_normal_next, returned_normal = randn_next!(rng, empty_normal)
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
        @test sprint(showerror, error) ==
              "ArgumentError: destination device differs from the generator device"
        @test destination == before_values
    end

    wrong_normal = fill(-123.5f0, 4)
    for operation in (randn!, randn_next!)
        before_values = copy(wrong_normal)
        @test_throws TypeError operation(exhausted, wrong_normal; threaded = 1)
        error = try
            operation(exhausted, wrong_normal)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test sprint(showerror, error) ==
              "ArgumentError: destination device differs from the generator device"
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

@testset "R56-R58 and R60 CUDA unweighted sampling" begin
    for F in FAMILIES
        cpu_rng = F(0x91a)
        gpu_rng = device(cpu_rng)
        host_population = reshape(collect(Int32(-11):Int32(12)), 4, 6)
        population = CuArray(host_population)

        next_gpu, values = randsample_next(gpu_rng, population, 19)
        next_cpu, expected = randsample_next(cpu_rng, host_population, 19)
        @test values isa CuArray{Int32,1}
        @test Array(values) == expected
        @test next_gpu.position == next_cpu.position
        @test _device_id(values) == CUDA.deviceid(primary)
        @test Array(randsample(gpu_rng, population, 7)) == expected[1:7]

        no_k_next, no_k = randsample_next(gpu_rng, population)
        expected_no_k_next, expected_no_k = randsample_next(cpu_rng, host_population)
        @test Array(no_k) == expected_no_k
        @test no_k_next.position == expected_no_k_next.position

        range = UInt64(0):(UInt64(1)<<32)
        range_next, range_values = randsample_next(gpu_rng, range, 5)
        expected_range_next, expected_range = randsample_next(cpu_rng, range, 5)
        @test Array(range_values) == expected_range
        @test range_next.position == expected_range_next.position

        iterable = DeviceAgnosticPopulation(collect(Int16(3):Int16(13)))
        iterable_next, iterable_values = randsample_next(gpu_rng, iterable, 9)
        expected_iterable_next, expected_iterable =
            randsample_next(cpu_rng, iterable.values, 9)
        @test iterable_values isa CuArray{Int16,1}
        @test Array(iterable_values) == expected_iterable
        @test iterable_next.position == expected_iterable_next.position
        @test iterable.starts[] == 1

        agnostic_array = DeviceAgnosticArray(collect(Int16(21):Int16(31)))
        array_next, array_values = randsample_next(gpu_rng, agnostic_array, 9)
        expected_array_next, expected_array =
            randsample_next(cpu_rng, agnostic_array.values, 9)
        @test array_values isa CuArray{Int16,1}
        @test Array(array_values) == expected_array
        @test array_next.position == expected_array_next.position
        @test agnostic_array.reads[] == length(agnostic_array)

        empty = CuArray{Int32}(undef, 0)
        empty_next, empty_values = randsample_next(gpu_rng, empty, 0)
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
        @test occursin("device", sprint(showerror, error))
    end

    limit_rng = device(Philox4x32(0x91b))
    limit_population = CUDA.CuArray(Int32[1])
    too_many = big(typemax(Int)) + 1
    @test_throws ArgumentError randsample(limit_rng, limit_population, too_many)
    @test_throws ArgumentError randsample_next(limit_rng, limit_population, too_many)

    audit_population = DeviceAgnosticPopulation(collect(Int32(1):Int32(13)))
    randsample(limit_rng, DeviceAgnosticPopulation(copy(audit_population.values)), 9)
    profiled_result = Ref{Any}()
    profile = CUDA.Profile.profile_internally(; concurrent = false, trace = true) do
        profiled_result[] = randsample(limit_rng, audit_population, 9)
    end
    @test audit_population.starts[] == 1
    @test profiled_result[] isa CUDA.CuArray{Int32,1}
    # The integrated profiler adds its own eight-byte H2D warm-up copy.
    h2d_sizes = [
        profile.device.size[index] for index in eachindex(profile.device.name) if
        profile.device.name[index] == "[copy pageable to device memory]"
    ]
    d2h_sizes = [
        profile.device.size[index] for index in eachindex(profile.device.name) if
        profile.device.name[index] == "[copy device to pageable memory]"
    ]
    @test sort(h2d_sizes) == [8, sizeof(Int32) * length(audit_population)]
    @test isempty(d2h_sizes)
end

@testset "GPU-bound scalar inference, allocation, and IR" begin
    for F in FAMILIES
        rng = device(F(0x123456))
        range = UInt32(2):UInt32(3):UInt32(74)
        _check_scalar(() -> rand_next(rng, UInt32), Tuple{typeof(rng),UInt32})
        _check_scalar(() -> randn_next(rng, Float32), Tuple{typeof(rng),Float32})
        _check_scalar(() -> rand_next(rng, range), Tuple{typeof(rng),UInt32})
        _check_scalar(() -> randat(rng, UInt64, 2), UInt64)
        _check_scalar(() -> randnat(rng, Float64, 2), Float64)
        _check_scalar(() -> subrng(rng, UInt64(0x71)), typeof(rng))
        _check_scalar(() -> splitrng(rng, Val(2)), NTuple{2,typeof(rng)})
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
        expected_next, expected = randsample_next(cpu_rng, cpu_population, cpu_weights, 17)
        next_rng, values = randsample_next(gpu_rng, gpu_population, gpu_weights, 17)
        @test values isa CUDA.CuArray{Int32,1}
        @test _device_id(values) == CUDA.deviceid(primary)
        @test Array(values) == expected
        @test next_rng.position == expected_next.position
        @test next_rng.device == gpu_rng.device
        @test Array(randsample(gpu_rng, gpu_population, gpu_weights, 9)) == expected[1:9]

        cursor = gpu_rng
        chained = Int32[]
        for _ = 1:17
            cursor, value = randsample_next(cursor, gpu_population, gpu_weights, 1)
            push!(chained, only(Array(value)))
        end
        @test chained == expected
        @test cursor.position == next_rng.position

        no_k_next, no_k = randsample_next(gpu_rng, gpu_population, gpu_weights)
        cpu_no_k_next, cpu_no_k = randsample_next(cpu_rng, cpu_population, cpu_weights)
        @test Array(no_k) == cpu_no_k
        @test no_k_next.position == cpu_no_k_next.position

        empty_next, empty = randsample_next(gpu_rng, gpu_population, gpu_weights, 0)
        @test empty isa CUDA.CuArray{Int32,1}
        @test isempty(empty)
        @test empty_next.position == gpu_rng.position

        last_rng = _last_draw_rng(gpu_rng, UInt16(53))
        terminal, final_value = randsample_next(last_rng, gpu_population, gpu_weights, 1)
        @test length(final_value) == 1
        @test terminal.position == _terminal(gpu_rng)
        @test_throws ArgumentError randsample(last_rng, gpu_population, gpu_weights, 2)
        @test_throws ArgumentError randsample_next(last_rng, gpu_population, gpu_weights, 2)
        @test last_rng.position.bit != IR._EXHAUSTED_BIT
    end

    range_population = UInt16(10):UInt16(3):UInt16(43)
    range_weights = CUDA.fill(1.0, length(range_population))
    range_rng = device(Philox4x32(0x792))
    range_next, range_values =
        randsample_next(range_rng, range_population, range_weights, 9)
    cpu_range_next, cpu_range_values = randsample_next(
        Philox4x32(0x792),
        range_population,
        ones(length(range_population)),
        9,
    )
    @test Array(range_values) == cpu_range_values
    @test range_next.position == cpu_range_next.position

    agnostic_weights = 1:length(range_population)
    agnostic_next, agnostic_values =
        randsample_next(range_rng, range_population, agnostic_weights, 9)
    cpu_agnostic_next, cpu_agnostic_values =
        randsample_next(Philox4x32(0x792), range_population, agnostic_weights, 9)
    @test agnostic_values isa CUDA.CuArray{UInt16,1}
    @test Array(agnostic_values) == cpu_agnostic_values
    @test agnostic_next.position == cpu_agnostic_next.position
    @test_throws ArgumentError randsample(range_rng, UInt16(1):UInt16(3), -1:1, 2)

    audit_population = UInt16(1):UInt16(13)
    randsample(range_rng, audit_population, 1:13, 9)
    counted_weights = DeviceAgnosticWeights(collect(Float32, 1:13), Ref(0))
    profiled_result = Ref{Any}()
    profile = CUDA.Profile.profile_internally(; concurrent = false, trace = true) do
        profiled_result[] = randsample(range_rng, audit_population, counted_weights, 9)
    end
    @test counted_weights.reads[] == length(counted_weights)
    @test profiled_result[] isa CUDA.CuArray{UInt16,1}
    # The integrated profiler adds its own eight-byte H2D warm-up copy.
    h2d_sizes = [
        profile.device.size[index] for index in eachindex(profile.device.name) if
        profile.device.name[index] == "[copy pageable to device memory]"
    ]
    d2h_sizes = [
        profile.device.size[index] for index in eachindex(profile.device.name) if
        profile.device.name[index] == "[copy device to pageable memory]"
    ]
    @test sort(h2d_sizes) == [8, sizeof(Float64) * length(counted_weights)]
    @test d2h_sizes == [1]

    converted, total = IR._prepare_weights(range_rng, 1:13, true)
    @test converted isa CUDA.CuArray{Float64,1}
    @test total isa CUDA.CuArray{Float64,1}
    @test length(total) == 1
    thresholds = CUDA.CuArray{Float64}(undef, 9)
    destination = CUDA.CuArray{UInt16}(undef, 9)
    backend = IR._fill_backend(thresholds)
    IR._fill_weighted_thresholds!(backend, range_rng, total, thresholds)
    frozen_thresholds = Array(thresholds)
    order = IR._weighted_sortperm(range_rng.device, thresholds)
    @test order isa CUDA.CuArray{Int,1}
    @test Array(thresholds) == frozen_thresholds
    IR._launch_weighted_scan!(
        backend,
        audit_population,
        converted,
        thresholds,
        order,
        destination,
    )
    @test destination isa CUDA.CuArray{UInt16,1}

    fold_weights = CUDA.CuArray(
        Float64[Float64(0x000f5d057718d3b7), Float64(0x0010a2fa88e72c49), 1.0, 1.0],
    )
    fold_rng = device(Philox4x32(0x9750))
    _, fold_total, _ = IR._prepare_weight_scan(fold_rng, fold_weights, false)
    fold_total_host = only(Array(fold_total))
    @test reinterpret(UInt64, fold_total_host) == 0x4340000000000000
    @test IR._weighted_threshold(fold_rng, fold_rng.position, fold_total_host) ==
          Float64(0x000f5d057718d3b6)
    reordered_total = reinterpret(Float64, UInt64(0x4340000000000001))
    @test IR._weighted_threshold(fold_rng, fold_rng.position, reordered_total) ==
          Float64(0x000f5d057718d3b7)
    @test Array(
        randsample(fold_rng, CUDA.CuArray(Int32[10, 20, 30, 40]), fold_weights, 1),
    ) == Int32[10]
    scan_destination = CUDA.CuArray{Int32}(undef, 1)
    scan_weights = CUDA.CuArray(Float64[0x1p53, 1.0, 1.0, 2.0])
    _, _, scan_cumulative = IR._prepare_weight_scan(range_rng, scan_weights, false)
    @test reinterpret.(UInt64, Array(scan_cumulative)) == UInt64[
        0x4340000000000000,
        0x4340000000000000,
        0x4340000000000000,
        0x4340000000000001,
    ]
    backend = IR._fill_backend(scan_destination)
    IR._launch_weighted_scan!(
        range_rng.device,
        backend,
        CUDA.CuArray(Int32[10, 20, 30, 40]),
        scan_weights,
        CUDA.CuArray(Float64[0x1p53]),
        CUDA.CuArray([1]),
        scan_cumulative,
        scan_destination,
    )
    @test Array(scan_destination) == Int32[40]
    equal_thresholds = CUDA.CuArray([0.5, 0.1, 0.5, 0.1])
    equal_order = IR._weighted_sortperm(range_rng.device, equal_thresholds)
    @test Array(equal_order) == [2, 4, 1, 3]
    @test Array(equal_thresholds) == [0.5, 0.1, 0.5, 0.1]

    wrong_weights = copy(cpu_weights)
    error = try
        randsample(range_rng, gpu_population, wrong_weights, -1)
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin("weights device", sprint(showerror, error))

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

include("fixed_distributions.jl")
include("enzyme.jl")
