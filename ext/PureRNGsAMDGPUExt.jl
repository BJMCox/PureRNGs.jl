module PureRNGsAMDGPUExt

import AMDGPU
import PureRNGs
import Random

const IR = PureRNGs
const _AMDGPUFamily = Union{
    IR.Philox2x32{IR._AMDGPUBackend},
    IR.Philox4x32{IR._AMDGPUBackend},
    IR.Philox2x64{IR._AMDGPUBackend},
    IR.Philox4x64{IR._AMDGPUBackend},
    IR.Threefry2x32{IR._AMDGPUBackend},
    IR.Threefry4x32{IR._AMDGPUBackend},
    IR.Threefry2x64{IR._AMDGPUBackend},
    IR.Threefry4x64{IR._AMDGPUBackend},
}

@inline IR._with_device(f, ::IR._AMDGPUBackend) = f()

@inline function IR._allocate_array(::IR._AMDGPUBackend, ::Type{T}, dims::Tuple) where {T}
    return AMDGPU.ROCArray{T}(undef, dims)
end

@inline function IR.rand_next(rng::_AMDGPUFamily, dim1::Integer, dims::Integer...)
    return IR._rand_next_uniform_array(rng, Float64, (dim1, dims...))
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline function Random.rand(
            rng::_AMDGPUFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._rand_next_uniform_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function IR.rand_next(
            rng::_AMDGPUFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._rand_next_uniform_array(rng, $T, (dim1, dims...))
        end
    end
end

@inline function IR.randn_next(rng::_AMDGPUFamily, dim1::Integer, dims::Integer...)
    return IR._randn_next_array(rng, Float64, (dim1, dims...))
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randn(
            rng::_AMDGPUFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._randn_next_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function IR.randn_next(
            rng::_AMDGPUFamily,
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
            rng::_AMDGPUFamily,
            range::AbstractRange{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._rand_next_range_array(rng, range, (dim1, dims...))
            return destination
        end

        @inline function IR.rand_next(
            rng::_AMDGPUFamily,
            range::AbstractRange{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._rand_next_range_array(rng, range, (dim1, dims...))
        end
    end
end

end
