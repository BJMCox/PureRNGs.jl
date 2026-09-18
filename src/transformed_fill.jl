# A codec selects the width, mapping, and fill plan through dispatch.
abstract type _MappedFillCodec end
# The exponential codec carries the backend because the log approximation is per backend.
struct _ExponentialCodec{B<:_BackendToken}
    backend::B
end
const _TransformedFillCodec = Union{Val{:normal},_ExponentialCodec,_MappedFillCodec}

# The single seam a backend extension overrides to select a tuned device kernel.
# `nothing` keeps the generic one-work-item-per-element kernel.
@inline _device_fill_plan(backend, rng, codec, ::Type{T}) where {T} = nothing

@inline _transformed_draw_unchecked(::Val{:normal}, rng, position, T) =
    _draw_normal_unchecked(rng, position, T)
@inline _transformed_draw_unchecked(::_ExponentialCodec, rng, position, T) =
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

const _CPU_TRANSFORMED_FILL_CHUNK_BITS = UInt64(8192 * 32)

@inline _transformed_fill_chunk_elements(codec, ::Type{T}) where {T} =
    Int(_CPU_TRANSFORMED_FILL_CHUNK_BITS ÷ UInt64(_fill_width(codec, T)))

@inline function _launch_transformed!(
    backend,
    rng,
    destination,
    ::Type{T},
    codec::_TransformedFillCodec,
) where {T}
    plan = _device_fill_plan(backend, rng, codec, T)
    return _launch_device_fill!(backend, rng, destination, T, codec, plan)
end

function _launch_transformed!(
    ::_CPUBackend,
    rng,
    destination::AbstractArray{T},
    ::Type{T},
    codec::_TransformedFillCodec,
) where {T}
    chunk_elements = _transformed_fill_chunk_elements(codec, T)
    indices = eachindex(destination)
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), _fill_width(codec, T))
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        chunk = _chunk_indices(indices, first, last)
        _fill_transformed_dense_cpu!(rng, position, destination, T, chunk, codec)
    end
    return destination
end

function _launch_transformed!(
    ::_CPUBackend,
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
    backend = _fill_backend(rng.device, destination)
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
