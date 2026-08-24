using CUDA
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
const UNIFORM_TYPES = (Bool, UInt32, UInt64, Float32, Float64)
const NORMAL_TYPES = (Float32, Float64)
const RANGE_TYPES = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)

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

function _device_api_kernel!(uniform, normal32, normal64, ranges, rng)
    if CUDA.threadIdx().x == 1
        next_uniform, continued_uniform = rand_next(rng, UInt32)
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
const PACKED_DRAW_SPECS = (
    (Bool, rand_next, rand_next!, UInt16(1), false),
    (UInt32, rand_next, rand_next!, UInt16(32), false),
    (UInt64, rand_next, rand_next!, UInt16(64), false),
    (Float32, rand_next, rand_next!, UInt16(24), false),
    (Float64, rand_next, rand_next!, UInt16(53), false),
    (Float32, randn_next, randn_next!, UInt16(23), true),
    (Float64, randn_next, randn_next!, UInt16(52), true),
)
const K64_RANGE = UInt32(3):UInt32(1003)
const K128_RANGE = UInt64(7):UInt64(3):UInt64(0xfffffffffffffffd)

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

        advanced, _ = rand_next(gpu_rng, UInt64)
        child = subrng(advanced, UInt64(0x71))
        children = splitrng(advanced, Val(2))
        dynamic_children = splitrng(advanced, 2)
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
        expected_kind =
            F === Philox4x32 && T in COOPERATIVE_UNIFORM_TYPES ? Val(:cooperative) :
            Val(:grouped)
        dispatch =
            which(IR._device_uniform_fill_plan, Tuple{typeof(backend),typeof(rng),Type{T}})
        @test (dispatch.module, plan[1]) === (extension_module, expected_kind)
    end

    for F in FAMILIES, T in NORMAL_TYPES
        rng = device(F(0x123456))
        plan = IR._device_normal_fill_plan(backend, rng, T)
        dispatch =
            which(IR._device_normal_fill_plan, Tuple{typeof(backend),typeof(rng),Type{T}})
        expected_kind = F === Philox4x32 ? Val(:cooperative) : Val(:grouped)
        @test (dispatch.module, plan[1]) === (extension_module, expected_kind)
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

@testset "public fill plan classes launch CUDA kernels" begin
    cases = (
        () -> rand(device(Philox4x32(0x771)), Bool, 9),
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
    for F in FAMILIES
        rng = device(F(0x123456))
        args = (
            CUDA.CuArray{UInt32}(undef, 6),
            CUDA.CuArray{Float32}(undef, 5),
            CUDA.CuArray{Float64}(undef, 5),
            CUDA.CuArray{UInt64}(undef, 2),
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
    IR._with_device(rng.device) do
        @test CUDA.device() == before
    end
    @test CUDA.device() == before
    @test_throws ErrorException IR._with_device(rng.device) do
        error("context probe")
    end
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
