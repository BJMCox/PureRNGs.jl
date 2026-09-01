const AxesKA = PureRNGs.KernelAbstractions
const AxesMLD = PureRNGs.MLDataDevices

struct ZeroBasedVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(array::ZeroBasedVector) = size(array.data)
Base.axes(array::ZeroBasedVector) = (Base.IdentityUnitRange(0:(length(array.data)-1)),)
Base.IndexStyle(::Type{<:ZeroBasedVector}) = IndexLinear()
function Base.getindex(array::ZeroBasedVector, index::Int)
    checkbounds(array, index)
    return @inbounds array.data[index+1]
end
function Base.setindex!(array::ZeroBasedVector, value, index::Int)
    checkbounds(array, index)
    @inbounds array.data[index+1] = value
    return value
end
AxesMLD.get_device(::ZeroBasedVector) = AxesMLD.CPUDevice()
AxesKA.get_backend(::ZeroBasedVector) = AxesKA.CPU()

struct IdentityAxesMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
end

Base.size(array::IdentityAxesMatrix) = size(array.data)
Base.axes(::IdentityAxesMatrix) =
    (Base.IdentityUnitRange(0:1), Base.IdentityUnitRange(-1:1))
Base.IndexStyle(::Type{<:IdentityAxesMatrix}) = IndexCartesian()
function Base.getindex(array::IdentityAxesMatrix, row::Int, column::Int)
    checkbounds(array, row, column)
    return @inbounds array.data[row+1, column+2]
end
function Base.setindex!(array::IdentityAxesMatrix, value, row::Int, column::Int)
    checkbounds(array, row, column)
    @inbounds array.data[row+1, column+2] = value
    return value
end
AxesMLD.get_device(::IdentityAxesMatrix) = AxesMLD.CPUDevice()
AxesKA.get_backend(::IdentityAxesMatrix) = AxesKA.CPU()

@testset "R24 and R26 zero-based CPU fills" begin
    for (fill!, T) in ((rand_next!, UInt32), (randn_next!, Float32))
        rng = Philox4x32(0x5240)
        serial = ZeroBasedVector(Vector{T}(undef, 9))
        threaded = ZeroBasedVector(similar(serial.data))

        fill!(rng, serial; threaded = false)
        fill!(rng, threaded; threaded = true)
        AxesKA.synchronize(AxesKA.CPU())

        @test collect(threaded) == collect(serial)

        serial_matrix = IdentityAxesMatrix(Matrix{T}(undef, 2, 3))
        threaded_matrix = IdentityAxesMatrix(similar(serial_matrix.data))
        fill!(rng, serial_matrix; threaded = false)
        fill!(rng, threaded_matrix; threaded = true)
        AxesKA.synchronize(AxesKA.CPU())

        @test threaded_matrix.data == serial_matrix.data
    end
end
