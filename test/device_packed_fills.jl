const PackedDeviceIR = PureRNGs
const PackedDeviceKA = PureRNGs.KernelAbstractions

function _packed_device_position(rng)
    bit = PackedDeviceIR._block_bits(rng) - UInt16(5)
    return rng.position isa PackedDeviceIR._Position64 ?
           PackedDeviceIR._Position64(UInt64(3), bit) :
           PackedDeviceIR._Position128(typemax(UInt64), UInt64(7), bit)
end
function _packed_device_rng(::Type{F}) where {F}
    rng = F(0x781)
    return PackedDeviceIR._rebuild(rng, _packed_device_position(rng), rng.device)
end

function _packed_cooperative_values(
    rng,
    ::Type{T},
    cooperative,
    width,
    family,
    codec,
    kernel = PackedDeviceIR._fill_cooperative_kernel!,
) where {T}
    outputs, workgroup = cooperative
    output_count = PackedDeviceIR._fill_group_size(outputs)
    workgroup_size = PackedDeviceIR._fill_group_size(workgroup)
    count = output_count + 3
    destination = Vector{T}(undef, count)
    groups = cld(count, output_count)
    kernel(PackedDeviceKA.CPU())(
        rng,
        destination,
        T,
        Val(width),
        outputs,
        workgroup,
        family,
        codec;
        ndrange = groups * workgroup_size,
        workgroupsize = workgroup_size,
    )
    PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
    return destination
end

@testset "cooperative Philox4x32 codec preserves packed stream" begin
    rng = _packed_device_rng(Philox4x32)
    for T in (Bool, Float32, Float64)
        cooperative = PackedDeviceIR._cooperative_uniform_fill(rng, T)
        destination = _packed_cooperative_values(
            rng,
            T,
            cooperative,
            PackedDeviceIR._draw_bits(T),
            PackedDeviceIR.FAMILY_BITS,
            Val(:uniform),
        )
        @test destination == _reference_chain(rng, T, length(destination))[2]
    end

    destination = _packed_cooperative_values(
        rng,
        Float32,
        PackedDeviceIR._cooperative_uniform_fill(rng, Float32),
        UInt16(24),
        PackedDeviceIR.FAMILY_BITS,
        Val(:uniform),
        PackedDeviceIR._fill_cooperative_float32_kernel!,
    )
    @test destination == _reference_chain(rng, Float32, length(destination))[2]

    for T in (Float32, Float64)
        cooperative = PackedDeviceIR._cooperative_normal_fill(rng, T)
        destination = _packed_cooperative_values(
            rng,
            T,
            cooperative,
            PackedDeviceIR._normal_bits(T),
            PackedDeviceIR.FAMILY_NORMAL,
            Val(:normal),
        )
        @test destination == _reference_normal_chain(rng, T, length(destination))[2]
    end

    base = Philox4x32(0x783)
    terminal = PackedDeviceIR._rebuild(
        base,
        PackedDeviceIR._Position64(
            PackedDeviceIR._max_block(base),
            PackedDeviceIR._block_bits(base) - UInt16(1),
        ),
        base.device,
    )
    destination = Vector{Bool}(undef, 1)
    outputs, workgroup = PackedDeviceIR._cooperative_uniform_fill(terminal, Bool)
    workgroup_size = PackedDeviceIR._fill_group_size(workgroup)
    PackedDeviceIR._fill_cooperative_kernel!(PackedDeviceKA.CPU())(
        terminal,
        destination,
        Bool,
        Val(1),
        outputs,
        workgroup,
        PackedDeviceIR.FAMILY_BITS,
        Val(:uniform);
        ndrange = workgroup_size,
        workgroupsize = workgroup_size,
    )
    PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
    @test only(destination) == PackedDeviceIR._draw_unchecked(terminal, Bool)
end

@testset "grouped device uniform codec preserves packed stream" begin
    for F in FAMILY_TYPES, T in (Bool, UInt32, UInt64, Float32, Float64)
        rng = _packed_device_rng(F)
        group = PackedDeviceIR._device_uniform_fill_group(rng, T)
        count = PackedDeviceIR._fill_group_size(group) + 3
        destination = Vector{T}(undef, count)
        workitems = cld(count, PackedDeviceIR._fill_group_size(group))
        PackedDeviceIR._uniform_fill_grouped_kernel!(PackedDeviceKA.CPU())(
            rng,
            destination,
            T,
            group;
            ndrange = workitems,
            workgroupsize = 1,
        )
        PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
        @test destination == _reference_chain(rng, T, count)[2]
    end
end

@testset "grouped device normal codec preserves packed stream" begin
    for F in FAMILY_TYPES, T in (Float32, Float64)
        rng = _packed_device_rng(F)
        group = PackedDeviceIR._device_normal_fill_group(T)
        count = PackedDeviceIR._fill_group_size(group) + 3
        destination = Vector{T}(undef, count)
        workitems = cld(count, PackedDeviceIR._fill_group_size(group))
        PackedDeviceIR._normal_fill_grouped_kernel!(PackedDeviceKA.CPU())(
            rng,
            destination,
            T,
            group;
            ndrange = workitems,
            workgroupsize = 1,
        )
        PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
        @test destination == _reference_normal_chain(rng, T, count)[2]
    end
end

@testset "grouped device range codec preserves fixed work" begin
    for F in FAMILY_TYPES
        rng = _packed_device_rng(F)
        range = UInt32(3):UInt32(1003)
        span = PackedDeviceIR._range_span(range)
        group = Val(2)
        count = PackedDeviceIR._fill_group_size(group) + 3
        destination = Vector{eltype(range)}(undef, count)
        workitems = cld(count, PackedDeviceIR._fill_group_size(group))
        PackedDeviceIR._range_fill_grouped_kernel!(PackedDeviceKA.CPU())(
            rng,
            destination,
            range,
            span,
            group;
            ndrange = workitems,
            workgroupsize = 1,
        )
        PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
        expected = eltype(range)[]
        cursor = rng
        for _ = 1:count
            cursor, value = rand_next(cursor, range)
            push!(expected, value)
        end
        @test destination == expected
    end
end

@testset "grouped device tails do not fetch past terminal capacity" begin
    for F in FAMILY_TYPES
        base = F(0x782)
        block_bits = PackedDeviceIR._block_bits(base)
        position = if base.position isa PackedDeviceIR._Position64
            PackedDeviceIR._Position64(PackedDeviceIR._max_block(base), block_bits - 1)
        else
            PackedDeviceIR._Position128(typemax(UInt64), typemax(UInt64), block_bits - 1)
        end
        rng = PackedDeviceIR._rebuild(base, position, base.device)
        destination = Vector{Bool}(undef, 1)
        group = PackedDeviceIR._device_uniform_fill_group(rng, Bool)
        PackedDeviceIR._uniform_fill_grouped_kernel!(PackedDeviceKA.CPU())(
            rng,
            destination,
            Bool,
            group;
            ndrange = 1,
            workgroupsize = 1,
        )
        PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
        @test only(destination) == PackedDeviceIR._draw_unchecked(rng, Bool)
    end
end
