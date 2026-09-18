@inline function _launch_uniform!(backend, rng, destination, ::Type{T}) where {T}
    plan = _device_uniform_fill_plan(backend, rng, T)
    return _launch_device_fill!(backend, rng, destination, T, Val(:uniform), plan)
end

function _launch_uniform!(
    ::KernelAbstractions.CPU,
    rng,
    destination::AbstractArray{T},
    ::Type{T},
) where {T}
    chunk_elements = _dense_fill_chunk_elements(T)
    indices = eachindex(destination)
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), _draw_bits(T))
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        chunk = _chunk_indices(indices, first, last)
        _fill_uniform_dense_cpu!(rng, position, destination, T, chunk)
    end
    return destination
end

@inline function _fill_uniform_prevalidated!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded::Bool,
) where {T}
    bits_lo, bits_hi = _bit_span(UInt64(length(destination)), _draw_bits(T))
    next_rng = _reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_uniform_dense_cpu!(rng, rng.position, destination, T, eachindex(destination))
        return destination, next_rng
    end
    backend = _fill_backend(destination)
    _launch_uniform!(backend, rng, destination, T)
    return destination, next_rng
end

@inline function _rand_next_fill!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T},
    threaded::Bool,
) where {T}
    _check_fill_device(rng, destination)
    _check_serviceability(rng, T)
    return _fill_uniform_prevalidated!(rng, destination, threaded)
end

@inline function Random.rand!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded = true,
) where {T<:_UniformResult}
    result, _ = _rand_next_fill!(rng, destination, _check_threaded(threaded))
    return result
end

@inline function rand_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded = true,
) where {T<:_UniformResult}
    return _rand_next_fill!(rng, destination, _check_threaded(threaded))
end

@doc """
    rand_next!(rng, destination; threaded=true) -> (destination, next_rng)
    rand_next!(rng, destination, range; threaded=true) -> (destination, next_rng)

Fill `destination` from `rng` and return the advanced immutable generator with
the same destination. The destination element type must be `Bool`, `UInt32`,
`Int32`, `UInt64`, `Int64`, `Float32`, or `Float64`, or with `range` the
integer element type of that range, and its device must match the generator.

Set `threaded=false` to request the serial CPU fill path. The keyword does not
change the generated stream. The input generator never changes.
""" rand_next!

@inline function _rand_next_uniform_array(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Tuple,
) where {T}
    _check_serviceability(rng, T)
    destination = _allocate_draw_array(rng.device, T, dims)
    return _fill_uniform_prevalidated!(rng, destination, true)
end

@inline function rand_next(rng::_ScalarUniformGenerators, dim1::Integer, dims::Integer...)
    return _rand_next_uniform_array(rng, Float64, (dim1, dims...))
end
@inline rand_next(rng::_ScalarUniformGenerators, dims::Dims) =
    _rand_next_uniform_array(rng, Float64, dims)

@inline function Random.rand(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...,
) where {T<:_UniformResult}
    destination, _ = _rand_next_uniform_array(rng, T, (dim1, dims...))
    return destination
end
@inline Random.rand(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims,
) where {T<:_UniformResult} = first(_rand_next_uniform_array(rng, T, dims))

@inline function rand_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...,
) where {T<:_UniformResult}
    return _rand_next_uniform_array(rng, T, (dim1, dims...))
end
@inline rand_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims,
) where {T<:_UniformResult} = _rand_next_uniform_array(rng, T, dims)
