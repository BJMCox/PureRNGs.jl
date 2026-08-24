const _CPUFamily = _BackendFamily{_CPUBackend}

@inline _allocate_array(::_CPUBackend, ::Type{T}, dims::Tuple) where {T} =
    Array{T}(undef, dims)
