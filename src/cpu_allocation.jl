const _CPUFamily = _BackendFamily{_CPUBackend}

@inline _allocate_array(::_CPUBackend, ::Type{T}, dims::Tuple) where {T} =
    Array{T}(undef, dims)

@noinline _invalid_dimensions() = throw(ArgumentError("dimensions must be non-negative"))

@inline function _allocate_draw_array(device, ::Type{T}, dims::Tuple) where {T}
    for dim in dims
        dim < 0 && _invalid_dimensions()
    end
    return _allocate_array(device, T, dims)
end
