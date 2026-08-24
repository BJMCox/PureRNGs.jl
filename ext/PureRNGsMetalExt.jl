module PureRNGsMetalExt

import PureRNGs
import Metal
import Random

const IR = PureRNGs
const _Metal32Family = IR._Backend32Family{IR._MetalBackend}
const _MetalFamily = IR._BackendFamily{IR._MetalBackend}

@noinline function _metal_device_error()
    throw(
        ArgumentError(
            "device-executing operation is not supported on Metal for this result type or generator family",
        ),
    )
end

for T in (Bool, UInt32, UInt64, Float32)
    @eval @inline IR._check_serviceability(::_Metal32Family, ::Type{$T}) = nothing
end
@inline IR._check_serviceability(::_MetalFamily, ::Type) = _metal_device_error()
@inline IR._check_sampling_serviceability(::_MetalFamily) = _metal_device_error()

@inline function IR._allocate_array(::IR._MetalBackend, ::Type{T}, dims::Tuple) where {T}
    return Metal.MtlArray{T}(undef, dims)
end

@inline function IR.rand_next(rng::_MetalFamily, dim1::Integer, dims::Integer...)
    return IR._rand_next_uniform_array(rng, Float64, (dim1, dims...))
end

for T in (Bool, UInt32, UInt64, Float32, Float64)
    @eval begin
        @inline function Random.rand(
            rng::_MetalFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._rand_next_uniform_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function IR.rand_next(
            rng::_MetalFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            return IR._rand_next_uniform_array(rng, $T, (dim1, dims...))
        end
    end
end

@inline function IR.randn_next(rng::_MetalFamily, dim1::Integer, dims::Integer...)
    return IR._randn_next_array(rng, Float64, (dim1, dims...))
end

for T in (Float32, Float64)
    @eval begin
        @inline function Random.randn(
            rng::_MetalFamily,
            ::Type{$T},
            dim1::Integer,
            dims::Integer...,
        )
            _, destination = IR._randn_next_array(rng, $T, (dim1, dims...))
            return destination
        end

        @inline function IR.randn_next(
            rng::_MetalFamily,
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
            ::_MetalFamily,
            ::AbstractRange{$T},
            ::Integer,
            ::Integer...,
        )
            return _metal_device_error()
        end

        @inline function IR.rand_next(
            ::_MetalFamily,
            ::AbstractRange{$T},
            ::Integer,
            ::Integer...,
        )
            return _metal_device_error()
        end
    end
end

end
