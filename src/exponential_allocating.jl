@inline function _randexp_next_array(
    rng::_ScalarUniformFamily,
    ::Type{T},
    dims::Tuple,
) where {T}
    return _rand_transformed_next_array(rng, T, dims, rng.device)
end

@inline function randexp_next(rng::_ScalarUniformFamily, dim1::Integer, dims::Integer...)
    return _randexp_next_array(rng, Float64, (dim1, dims...))
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randexp(
            rng::_ScalarUniformFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = _randexp_next_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function randexp_next(
            rng::_ScalarUniformFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return _randexp_next_array(rng, $T, (dim1, dims...))
        end
    end
end
