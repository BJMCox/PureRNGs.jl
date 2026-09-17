# A codec selects the width, mapping, and fill plan through dispatch.
abstract type _MappedFillCodec end
const _TransformedFillCodec = Union{Val{:normal},_BackendToken,_MappedFillCodec}

@inline _transformed_draw_unchecked(::Val{:normal}, rng, position, T) =
    _draw_normal_unchecked(rng, position, T)
@inline _transformed_draw_unchecked(::_BackendToken, rng, position, T) =
    _draw_exponential_unchecked(rng, position, T)
@inline function _transformed_draw_unchecked(
    codec::_MappedFillCodec,
    rng,
    position,
    ::Type{T},
) where {T}
    raw = _extract_bits_unchecked(
        rng,
        _position_block(position),
        position.bit,
        Val(_fill_width(codec, T)),
    )
    return _cooperative_value(codec, T, raw)
end

@inline function _fill_transformed_dense_cpu!(
    rng,
    position,
    destination,
    ::Type{T},
    indices,
    codec::_TransformedFillCodec,
) where {T}
    isempty(indices) && return nothing
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    width = Val(_fill_width(codec, T))
    @inbounds for index in indices
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, width)
        destination[index] = _cooperative_value(codec, T, raw)
    end
    return nothing
end

@inline function _fill_transformed_grouped_unchecked!(
    rng,
    position,
    destination,
    ::Type{T},
    first::Int,
    group::Val{N},
    codec::_TransformedFillCodec,
) where {T,N}
    return _fill_grouped_cursor!(
        rng,
        position,
        destination,
        T,
        first,
        group,
        Val(_fill_width(codec, T)),
        codec,
    )
end

KernelAbstractions.@kernel function _transformed_fill_kernel!(
    rng,
    destination,
    ::Val{T},
    codec,
) where {T}
    ordinal = @index(Global, Linear)
    indices = eachindex(destination)
    index = @inbounds indices[firstindex(indices)+ordinal-1]
    bits_lo, bits_hi = _bit_span(UInt64(ordinal - 1), _fill_width(codec, T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    @inbounds destination[index] = _transformed_draw_unchecked(codec, rng, position, T)
end

KernelAbstractions.@kernel function _transformed_fill_grouped_kernel!(
    rng,
    destination,
    ::Val{T},
    group::Val{N},
    codec,
) where {T,N}
    workitem = @index(Global, Linear)
    first = (workitem - 1) * N + 1
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _fill_width(codec, T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_transformed_grouped_unchecked!(rng, position, destination, T, first, group, codec)
end

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::_TransformedFillCodec,
    ::Nothing,
) where {T}
    _transformed_fill_kernel!(backend)(
        rng,
        destination,
        Val(T),
        codec;
        ndrange = length(destination),
    )
    return destination
end

@inline function _launch_device_fill!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::_TransformedFillCodec,
    plan::Tuple{Val{:grouped},Val{N}},
) where {T,N}
    group = plan[2]
    workitems = cld(length(destination), _fill_group_size(group))
    _transformed_fill_grouped_kernel!(backend)(
        rng,
        destination,
        Val(T),
        group,
        codec;
        ndrange = workitems,
    )
    return destination
end

const _CPU_TRANSFORMED_FILL_CHUNK_BITS = UInt64(8192 * 32)

@inline _transformed_fill_chunk_elements(codec, ::Type{T}) where {T} =
    Int(_CPU_TRANSFORMED_FILL_CHUNK_BITS ÷ UInt64(_fill_width(codec, T)))

KernelAbstractions.@kernel function _transformed_fill_dense_kernel!(
    rng,
    destination,
    ::Val{T},
    chunk_elements,
    codec,
) where {T}
    workitem = @index(Global, Linear)
    first, last = _dense_fill_bounds(workitem, length(destination), chunk_elements)
    bits_lo, bits_hi = _bit_span(UInt64(first - 1), _fill_width(codec, T))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_transformed_dense_cpu!(rng, position, destination, T, first:last, codec)
end

@inline function _launch_transformed!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::_TransformedFillCodec,
) where {T}
    plan = _transformed_fill_plan(codec, backend, rng, T)
    return _launch_device_fill!(backend, rng, destination, T, codec, plan)
end

function _launch_transformed!(
    backend::KernelAbstractions.CPU,
    rng,
    destination::Array{T},
    ::Type{T},
    codec::_TransformedFillCodec,
) where {T}
    chunk_elements = _transformed_fill_chunk_elements(codec, T)
    workitems = cld(length(destination), chunk_elements)
    if workitems < _CPU_FILL_MIN_WORKITEMS
        _fill_transformed_dense_cpu!(
            rng,
            rng.position,
            destination,
            T,
            eachindex(destination),
            codec,
        )
        return destination
    end
    _transformed_fill_dense_kernel!(backend)(
        rng,
        destination,
        Val(T),
        chunk_elements,
        codec;
        ndrange = workitems,
        workgroupsize = 1,
    )
    return destination
end

function _launch_transformed!(
    ::KernelAbstractions.CPU,
    rng,
    destination::BitArray,
    ::Type{Bool},
    codec::_MappedFillCodec,
)
    _fill_transformed_dense_cpu!(
        rng,
        rng.position,
        destination,
        Bool,
        eachindex(destination),
        codec,
    )
    return destination
end

@inline function _fill_transformed_prevalidated!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded::Bool,
    codec::_TransformedFillCodec,
) where {T}
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _fill_width(codec, T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_transformed_dense_cpu!(
            rng,
            rng.position,
            destination,
            T,
            eachindex(destination),
            codec,
        )
        return destination, next_rng
    end
    backend = _fill_backend(destination)
    _launch_transformed!(backend, rng, destination, T, codec)
    return destination, next_rng
end

@inline function _rand_transformed_next_fill!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded::Bool,
    codec::_TransformedFillCodec,
) where {T}
    _check_fill_device(rng, destination)
    _check_serviceability(rng, T)
    return _fill_transformed_prevalidated!(rng, destination, threaded, codec)
end

@inline function _rand_transformed_next_array(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Tuple,
    codec::_TransformedFillCodec,
) where {T}
    _check_serviceability(rng, T)
    destination = _allocate_draw_array(rng.device, T, dims)
    return _fill_transformed_prevalidated!(rng, destination, true, codec)
end
