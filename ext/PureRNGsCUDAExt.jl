module PureRNGsCUDAExt

import CUDA
import PureRNGs
import MLDataDevices
import Random

const IR = PureRNGs
const _NamedCUDADevice = MLDataDevices.CUDADevice{<:CUDA.CuDevice}
const _CUDAFamily = Union{
    IR.Philox2x32{<:_NamedCUDADevice},
    IR.Philox4x32{<:_NamedCUDADevice},
    IR.Philox2x64{<:_NamedCUDADevice},
    IR.Philox4x64{<:_NamedCUDADevice},
    IR.Threefry2x32{<:_NamedCUDADevice},
    IR.Threefry4x32{<:_NamedCUDADevice},
    IR.Threefry2x64{<:_NamedCUDADevice},
    IR.Threefry4x64{<:_NamedCUDADevice},
}

@inline function (device::_NamedCUDADevice)(rng::IR.AbstractPureRNG)
    return IR._rebuild(rng, rng.position, device)
end

@inline function IR._with_device(f, device::_NamedCUDADevice)
    return CUDA.device!(f, device.device)
end

@inline function IR._allocate_array(
    device::_NamedCUDADevice,
    ::Type{T},
    dims::Tuple,
) where {T}
    return IR._with_device(device) do
        CUDA.CuArray{T}(undef, dims)
    end
end

@inline function IR._same_fill_device(
    generator_device::_NamedCUDADevice,
    destination_device::_NamedCUDADevice,
)
    return CUDA.deviceid(generator_device.device) ==
           CUDA.deviceid(destination_device.device)
end

@inline function IR.rand_next(rng::_CUDAFamily, dim1::Integer, dims::Integer...)
    return IR._rand_next_uniform_array(rng, Float64, (dim1, dims...))
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline function Random.rand(
            rng::_CUDAFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._rand_next_uniform_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function IR.rand_next(
            rng::_CUDAFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._rand_next_uniform_array(rng, $T, (dim1, dims...))
        end
    end
end

@inline function IR.randn_next(rng::_CUDAFamily, dim1::Integer, dims::Integer...)
    return IR._randn_next_array(rng, Float64, (dim1, dims...))
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randn(
            rng::_CUDAFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._randn_next_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function IR.randn_next(
            rng::_CUDAFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._randn_next_array(rng, $T, (dim1, dims...))
        end
    end
end

for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
    @eval begin
        @inline function Random.rand(
            rng::_CUDAFamily,
            range::AbstractRange{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._rand_next_range_array(rng, range, (dim1, dims...))
            return destination
        end

        @inline function IR.rand_next(
            rng::_CUDAFamily,
            range::AbstractRange{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._rand_next_range_array(rng, range, (dim1, dims...))
        end
    end
end

end
