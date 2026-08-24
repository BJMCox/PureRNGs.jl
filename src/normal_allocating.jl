@inline function _randn_next_array(rng::_CPUFamily, ::Type{T}, dims::Tuple) where {T}
    destination = _allocate_array(rng.device, T, dims)
    return _randn_next_fill!(rng, destination, true)
end

@inline function randn_next(rng::_CPUFamily, dim1::Integer, dims::Integer...)
    return _randn_next_array(rng, Float64, (dim1, dims...))
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randn(
            rng::_CPUFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = _randn_next_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function randn_next(
            rng::_CPUFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return _randn_next_array(rng, $T, (dim1, dims...))
        end
    end
end
