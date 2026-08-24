const _CPUFamily = Union{
    Philox2x32{<:MLDataDevices.CPUDevice},
    Philox4x32{<:MLDataDevices.CPUDevice},
    Philox2x64{<:MLDataDevices.CPUDevice},
    Philox4x64{<:MLDataDevices.CPUDevice},
    Threefry2x32{<:MLDataDevices.CPUDevice},
    Threefry4x32{<:MLDataDevices.CPUDevice},
    Threefry2x64{<:MLDataDevices.CPUDevice},
    Threefry4x64{<:MLDataDevices.CPUDevice},
}

@inline _allocate_array(::MLDataDevices.CPUDevice, ::Type{T}, dims::Tuple) where {T} =
    Array{T}(undef, dims)
