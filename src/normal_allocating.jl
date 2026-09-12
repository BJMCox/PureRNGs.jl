@inline function _randn_next_array(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Tuple,
) where {T}
    return _rand_transformed_next_array(rng, T, dims, Val(:normal))
end

@inline function randn_next(rng::_ScalarUniformGenerators, dim1::Integer, dims::Integer...)
    return _randn_next_array(rng, Float64, (dim1, dims...))
end
@inline randn_next(rng::_ScalarUniformGenerators, dims::Dims) =
    _randn_next_array(rng, Float64, dims)

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randn(
            rng::_ScalarUniformGenerators,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            destination, _ = _randn_next_array(rng, $T, (dim1, dims...))
            return destination
        end
        @inline Random.randn(rng::_ScalarUniformGenerators, ::Type{$T}, dims::Dims) =
            first(_randn_next_array(rng, $T, dims))
        @inline randn_next(rng::_ScalarUniformGenerators, ::Type{$T}, dims::Dims) =
            _randn_next_array(rng, $T, dims)

        @inline function randn_next(
            rng::_ScalarUniformGenerators,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return _randn_next_array(rng, $T, (dim1, dims...))
        end
    end
end
