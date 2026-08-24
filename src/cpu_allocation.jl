const _CPUFamily = Union{
    Philox2x32{_CPUBackend},
    Philox4x32{_CPUBackend},
    Philox2x64{_CPUBackend},
    Philox4x64{_CPUBackend},
    Threefry2x32{_CPUBackend},
    Threefry4x32{_CPUBackend},
    Threefry2x64{_CPUBackend},
    Threefry4x64{_CPUBackend},
}

@inline _allocate_array(::_CPUBackend, ::Type{T}, dims::Tuple) where {T} =
    Array{T}(undef, dims)
