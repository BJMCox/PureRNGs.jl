KernelAbstractions.@kernel function _uniform_fill_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds indices[firstindex(indices)+ordinal-1]
    bits_lo, bits_hi = _bit_span(UInt64(ordinal - 1), _draw_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = _draw_unchecked(rng, position, T)
end

KernelAbstractions.@kernel function _uniform_fill_grouped_kernel!(
    rng,
    destination,
    ::Type{T},
    group::Val{N},
) where {T,N}
    workitem = @index(Global, Linear)
    first = (workitem - 1) * N + 1
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _draw_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_uniform_grouped_unchecked!(rng, position, destination, T, first, group)
end

@inline _cooperative_value(::Val{:uniform}, ::Type{T}, raw) where {T} = _from_bits(T, raw)
@inline _fill_width(::Val{:uniform}, ::Type{T}) where {T} = _draw_bits(T)
@inline _fill_family(::Val{:uniform}) = FAMILY_BITS
@inline _fill_kernel(::Val{:uniform}) = _uniform_fill_kernel!
@inline _fill_grouped_kernel(::Val{:uniform}) = _uniform_fill_grouped_kernel!

@inline _stream_block_offset(block::UInt64, offset::UInt64) = block + offset
@inline function _stream_block_offset(block::NTuple{2,UInt64}, offset::UInt64)
    lo = block[1] + offset
    return lo, block[2] + UInt64(lo < block[1])
end

@inline _cooperative_shared_limbs(::Val{B}, ::Val{O}, ::Val{W}) where {B,O,W} =
    (B ÷ 64) * cld((B - 1) + O * W, B)

KernelAbstractions.@kernel function _fill_cooperative_kernel!(
    rng,
    destination,
    ::Type{T},
    ::Val{W},
    block_width::Val{B},
    ::Val{O},
    ::Val{L},
    ::Val{P},
    ::Val{S},
    family::UInt32,
    codec,
) where {T,W,B,O,L,P,S}
    group = @index(Group, Linear)
    lane = @index(Local, Linear)
    shared = @localmem UInt64 (
        S ? 2 * cld(O * W, 128) : _cooperative_shared_limbs(block_width, Val(O), Val(W)),
    )
    if S
        blocks = cld(O * W, 128)
        first_block = rng.position.block + UInt64((group - 1) * blocks)
        block_offset = lane - 1
        while block_offset < blocks
            limbs = _stream_limbs(rng, family, first_block + UInt64(block_offset))
            @inbounds begin
                shared[2block_offset+1] = limbs[1]
                shared[2block_offset+2] = limbs[2]
            end
            block_offset += L
        end
    else
        first = (group - 1) * O + 1
        outputs = min(O, P * length(destination) - first + 1)
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), UInt16(W))
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        block_bits = B
        blocks = cld(Int(position.bit) + outputs * W, block_bits)

        block_offset = lane - 1
        while block_offset < blocks
            block = _stream_block_offset(_position_block(position), UInt64(block_offset))
            limbs = _stream_limbs(rng, family, block)
            shared_first = block_offset * length(limbs)
            @inbounds for limb in eachindex(limbs)
                shared[shared_first+limb] = limbs[limb]
            end
            block_offset += L
        end
    end
    @synchronize

    write_group = @index(Group, Linear)
    write_lane = @index(Local, Linear)
    if S
        pack = write_lane - 1
        packs = O ÷ P
        first_pack = (write_group - 1) * packs + 1
        while pack < packs
            @inbounds destination[first_pack+pack] =
                _cooperative_pack(codec, T, shared, P * pack * W, Val(P), Val(W))
            pack += L
        end
    else
        write_first = (write_group - 1) * O + 1
        write_outputs = min(O, P * length(destination) - write_first + 1)
        write_bits_lo, write_bits_hi = _bit_span(UInt64(write_first - 1), UInt16(W))
        write_position = _advance_position_unchecked(rng, write_bits_lo, write_bits_hi)
        if P == 1
            output = write_lane - 1
            while output < write_outputs
                raw =
                    _local_dense_bits(shared, Int(write_position.bit) + output * W, Val(W))
                @inbounds destination[write_first+output] =
                    _cooperative_value(codec, T, raw)
                output += L
            end
        else
            output = P * (write_lane - 1)
            while output < write_outputs
                bit = Int(write_position.bit) + output * W
                index = ((write_first - 1) + output) ÷ P + 1
                @inbounds destination[index] =
                    _cooperative_pack(codec, T, shared, bit, Val(P), Val(W))
                output += P * L
            end
        end
    end
end

@inline function _cooperative_pack(
    codec,
    T,
    shared,
    bit,
    outputs_per_store::Val{P},
    ::Val{W},
) where {P,W}
    return ntuple(outputs_per_store) do lane
        raw = _local_dense_bits(shared, bit + (lane - 1) * W, Val(W))
        VecElement(_cooperative_value(codec, T, raw))
    end
end

KernelAbstractions.@kernel function _uniform_fill_bool_blocks_kernel!(
    rng,
    destination,
    ::Val{P},
) where {P}
    block_ordinal = @index(Global, Linear)
    stride = KernelAbstractions.@ndrange()[1]
    block_count = length(destination) ÷ P
    while block_ordinal <= block_count
        bits_lo, bits_hi = _bit_span(UInt64(block_ordinal - 1), _block_bits(rng))
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        limbs = _stream_limbs(rng, FAMILY_BITS, _position_block(position))
        first_pack = (block_ordinal - 1) * P + 1
        pack = 0
        while pack < P
            @inbounds destination[first_pack+pack] =
                _cooperative_pack(Val(:uniform), Bool, limbs, 16pack, Val(16), Val(1))
            pack += 1
        end
        block_ordinal += stride
    end
end

const _CPU_FILL_CHUNK_BITS = UInt64(4096 * 32)
const _CPU_FILL_MIN_WORKITEMS = 4

@inline function _dense_fill_chunk_elements(::Type{T}) where {T}
    raw = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_draw_bits(T)))
    group = _dense_fill_group(T)
    return raw - raw % group
end
@inline function _dense_fill_bounds(workitem::Int, count::Int, chunk_elements::Int)
    first = (workitem - 1) * chunk_elements + 1
    chunk_count = min(chunk_elements, count - first + 1)
    return first, first + chunk_count - 1
end

KernelAbstractions.@kernel function _uniform_fill_dense_kernel!(
    rng,
    destination,
    ::Type{T},
    chunk_elements,
) where {T}
    workitem = @index(Global, Linear)
    first, last = _dense_fill_bounds(workitem, length(destination), chunk_elements)
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _draw_bits(T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_uniform_dense_cpu!(rng, position, destination, T, first:last)
end

KernelAbstractions.@kernel function _uniform_fill_dense_serial_kernel!(
    rng,
    destination,
    ::Type{T},
) where {T}
    _fill_uniform_dense_cpu!(rng, rng.position, destination, T, eachindex(destination))
end

@inline _fill_backend(destination) = KernelAbstractions.get_backend(destination)
@inline _fill_backend(destination::BitArray) =
    KernelAbstractions.get_backend(destination.chunks)

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    ::Nothing,
) where {T}
    _fill_kernel(codec)(backend)(rng, destination, T; ndrange = length(destination))
    return destination
end

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    plan::Tuple{Val{:grouped},Val{N}},
) where {T,N}
    group = plan[2]
    workitems = cld(length(destination), _fill_group_size(group))
    _fill_grouped_kernel(codec)(backend)(rng, destination, T, group; ndrange = workitems)
    return destination
end

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    plan::Tuple{Val{:cooperative},Val{O},Val{L}},
) where {T,O,L}
    return _launch_cooperative_fill!(backend, rng, destination, T, codec, plan, Val(false))
end

@inline _outputs_per_store(::Tuple{Val{:cooperative},Val{O},Val{L}}) where {O,L} = Val(1)
@inline _outputs_per_store(
    plan::Tuple{Val{:cooperative},Val{O},Val{L},Val{P}},
) where {O,L,P} = plan[4]

@inline function _launch_cooperative_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec,
    plan,
    stream_aligned::Val{S},
) where {T,S}
    outputs, workgroup = plan[2], plan[3]
    outputs_per_store = _outputs_per_store(plan)
    output_count = _fill_group_size(outputs)
    workgroup_size = _fill_group_size(workgroup)
    groups = cld(_fill_group_size(outputs_per_store) * length(destination), output_count)
    _fill_cooperative_kernel!(backend)(
        rng,
        destination,
        T,
        Val(_fill_width(codec, T)),
        Val(Int(_block_bits(rng))),
        outputs,
        workgroup,
        outputs_per_store,
        stream_aligned,
        _fill_family(codec),
        codec;
        ndrange = groups * workgroup_size,
        workgroupsize = workgroup_size,
    )
    return destination
end
