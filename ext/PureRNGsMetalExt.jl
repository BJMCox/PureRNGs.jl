module PureRNGsMetalExt

import PureRNGs
import Metal

const IR = PureRNGs
const _Metal32Generators = IR._Backend32Generators{IR._MetalBackend}
const _MetalGenerators = IR._BackendGenerators{IR._MetalBackend}

@noinline function _metal_device_error()
    throw(
        ArgumentError(
            "device execution is not supported on Metal for this result type or generator",
        ),
    )
end

for T in (Bool, UInt32, Int32, UInt64, Int64, Float32)
    @eval @inline IR._check_serviceability(::_Metal32Generators, ::Type{$T}) = nothing
end
@inline IR._check_serviceability(::_MetalGenerators, ::Type) = _metal_device_error()
@inline IR._check_serviceability(::_MetalGenerators, ::AbstractRange) =
    _metal_device_error()
@inline IR._check_sampling_serviceability(::_MetalGenerators) = _metal_device_error()

@inline function IR._allocate_array(::IR._MetalBackend, ::Type{T}, dims::Tuple) where {T}
    return Metal.MtlArray{T}(undef, dims)
end

end
