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
) where {T}
    outputs, workgroup = cooperative
    output_count = PackedDeviceIR._fill_group_size(outputs)
    workgroup_size = PackedDeviceIR._fill_group_size(workgroup)
    count = output_count + 3
    destination = Vector{T}(undef, count)
    groups = cld(count, output_count)
    PackedDeviceIR._fill_cooperative_kernel!(PackedDeviceKA.CPU())(
        rng,
        destination,
        T,
        Val(width),
        Val(Int(PackedDeviceIR._block_bits(rng))),
        outputs,
        workgroup,
        Val(1),
        Val(false),
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

    for T in (Float32, Float64)
        cooperative = PackedDeviceIR._cooperative_exponential_fill(rng, T)
        destination = _packed_cooperative_values(
            rng,
            T,
            cooperative,
            PackedDeviceIR._exponential_bits(T),
            PackedDeviceIR.FAMILY_EXP,
            rng.device,
        )
        @test destination == _scalar_exponential_chain(rng, T, length(destination))[2]
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
        Val(Int(PackedDeviceIR._block_bits(terminal))),
        outputs,
        workgroup,
        Val(1),
        Val(false),
        PackedDeviceIR.FAMILY_BITS,
        Val(:uniform);
        ndrange = workgroup_size,
        workgroupsize = workgroup_size,
    )
    PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
    @test only(destination) == PackedDeviceIR._draw_unchecked(terminal, Bool)
end

@testset "packed cooperative float stores preserve non-Philox streams" begin
    @test PackedDeviceIR._cooperative_shared_limbs(Val(256), Val(16), Val(24)) == 12
    @test PackedDeviceIR._cooperative_shared_limbs(Val(256), Val(16), Val(53)) == 20
    @test PackedDeviceIR._cooperative_shared_limbs(Val(128), Val(2048), Val(24)) == 770

    kernel = PackedDeviceIR._fill_cooperative_kernel!(PackedDeviceKA.CPU())
    for F in FAMILY_TYPES
        F === Philox4x32 && continue
        rng = _packed_device_rng(F)
        for (T, outputs_per_store) in ((Float32, Val(4)), (Float64, Val(2)))
            outputs = Val(16)
            workgroup = Val(4)
            count =
                PackedDeviceIR._fill_group_size(outputs) +
                PackedDeviceIR._fill_group_size(outputs_per_store)
            values = Vector{T}(undef, count)
            packed = reinterpret(
                NTuple{PackedDeviceIR._fill_group_size(outputs_per_store),VecElement{T}},
                values,
            )
            groups = cld(count, PackedDeviceIR._fill_group_size(outputs))
            kernel(
                rng,
                packed,
                T,
                Val(PackedDeviceIR._draw_bits(T)),
                Val(Int(PackedDeviceIR._block_bits(rng))),
                outputs,
                workgroup,
                outputs_per_store,
                Val(false),
                PackedDeviceIR.FAMILY_BITS,
                Val(:uniform);
                ndrange = groups * PackedDeviceIR._fill_group_size(workgroup),
                workgroupsize = PackedDeviceIR._fill_group_size(workgroup),
            )
            PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
            @test values == _reference_chain(rng, T, count)[2]
        end
    end
end

@testset "natural-block Bool stores preserve every family stream" begin
    kernel = PackedDeviceIR._uniform_fill_bool_blocks_kernel!(PackedDeviceKA.CPU())
    for F in FAMILY_TYPES
        base = F(0x787)
        position =
            base.position isa PackedDeviceIR._Position64 ?
            PackedDeviceIR._Position64(UInt64(3), UInt16(0)) :
            PackedDeviceIR._Position128(UInt64(3), UInt64(7), UInt16(0))
        rng = PackedDeviceIR._rebuild(base, position, base.device)
        block_bits = Int(PackedDeviceIR._block_bits(rng))
        packs_per_block = Val(block_bits ÷ 16)
        values = Vector{Bool}(undef, 2block_bits)
        packed = reinterpret(NTuple{16,VecElement{Bool}}, values)
        kernel(rng, packed, packs_per_block; ndrange = 1, workgroupsize = 1)
        PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
        @test values == _reference_chain(rng, Bool, length(values))[2]
    end
end

@testset "grouped device uniform codec preserves packed stream" begin
    for F in FAMILY_TYPES, T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
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

@testset "grouped device exponential codec preserves packed stream" begin
    for F in FAMILY_TYPES, T in (Float32, Float64)
        rng = _packed_device_rng(F)
        group = PackedDeviceIR._device_exponential_fill_group(T)
        count = PackedDeviceIR._fill_group_size(group) + 3
        destination = Vector{T}(undef, count)
        workitems = cld(count, PackedDeviceIR._fill_group_size(group))
        PackedDeviceIR._exponential_fill_grouped_kernel!(PackedDeviceKA.CPU())(
            rng,
            destination,
            T,
            group;
            ndrange = workitems,
            workgroupsize = 1,
        )
        PackedDeviceKA.synchronize(PackedDeviceKA.CPU())
        @test destination == _scalar_exponential_chain(rng, T, count)[2]
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
