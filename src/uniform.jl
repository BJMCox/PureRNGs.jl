# Uniform draws run the transformed scaffold under `Val(:uniform)`, whose
# `_fill_width` and `_cooperative_value` methods live in `fill_hooks.jl`.
@inline function Random.rand!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded = true,
) where {T<:_UniformResult}
    result, _ = _rand_transformed_next_fill!(rng, destination, threaded, Val(:uniform))
    return result
end

@inline function rand_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded = true,
) where {T<:_UniformResult}
    return _rand_transformed_next_fill!(rng, destination, threaded, Val(:uniform))
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

@inline function rand_next(rng::_ScalarUniformGenerators, dim1::Integer, dims::Integer...)
    return _rand_transformed_next_array(rng, Float64, (dim1, dims...), Val(:uniform))
end
@inline rand_next(rng::_ScalarUniformGenerators, dims::Dims) =
    _rand_transformed_next_array(rng, Float64, dims, Val(:uniform))

@inline function Random.rand(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...,
) where {T<:_UniformResult}
    destination, _ = _rand_transformed_next_array(rng, T, (dim1, dims...), Val(:uniform))
    return destination
end
@inline Random.rand(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims,
) where {T<:_UniformResult} =
    first(_rand_transformed_next_array(rng, T, dims, Val(:uniform)))

@inline function rand_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...,
) where {T<:_UniformResult}
    return _rand_transformed_next_array(rng, T, (dim1, dims...), Val(:uniform))
end
@inline rand_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims,
) where {T<:_UniformResult} = _rand_transformed_next_array(rng, T, dims, Val(:uniform))
