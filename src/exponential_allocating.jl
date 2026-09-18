@inline function _randexp_next_array(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Tuple,
) where {T}
    return _rand_transformed_next_array(rng, T, dims, _ExponentialCodec(rng.device))
end

@inline function randexp_next(
    rng::_ScalarUniformGenerators,
    dim1::Integer,
    dims::Integer...,
)
    return _randexp_next_array(rng, Float64, (dim1, dims...))
end
@inline randexp_next(rng::_ScalarUniformGenerators, dims::Dims) =
    _randexp_next_array(rng, Float64, dims)

@inline function Random.randexp(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...,
) where {T<:_UniformFloat}
    destination, _ = _randexp_next_array(rng, T, (dim1, dims...))
    return destination
end
@inline Random.randexp(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims,
) where {T<:_UniformFloat} = first(_randexp_next_array(rng, T, dims))
@inline randexp_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims,
) where {T<:_UniformFloat} = _randexp_next_array(rng, T, dims)

@inline function randexp_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...,
) where {T<:_UniformFloat}
    return _randexp_next_array(rng, T, (dim1, dims...))
end
