@inline function _rand_next_uniform_array(
    rng::_ScalarUniformFamily,
    ::Type{T},
    dims::Tuple,
) where {T}
    _check_serviceability(rng, T)
    destination = _allocate_array(rng.device, T, dims)
    return _rand_next_fill!(rng, destination, true)
end

@inline function rand_next(rng::_ScalarUniformFamily, dim1::Integer, dims::Integer...)
    return _rand_next_uniform_array(rng, Float64, (dim1, dims...))
end

for T in (Bool, UInt32, Int32, UInt64, Int64, Float32, Float64)
    @eval begin
        @inline function Random.rand(
            rng::_ScalarUniformFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = _rand_next_uniform_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function rand_next(
            rng::_ScalarUniformFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return _rand_next_uniform_array(rng, $T, (dim1, dims...))
        end
    end
end
