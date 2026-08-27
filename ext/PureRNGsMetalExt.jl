module PureRNGsMetalExt

import PureRNGs
import Metal

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

for T in (Bool, UInt32, Int32, UInt64, Int64, Float32)
    @eval @inline IR._check_serviceability(::_Metal32Family, ::Type{$T}) = nothing
end
@inline IR._check_serviceability(::_MetalFamily, ::Type) = _metal_device_error()
@inline IR._check_serviceability(::_MetalFamily, ::AbstractRange) = _metal_device_error()
@inline IR._check_sampling_serviceability(::_MetalFamily) = _metal_device_error()

@inline function IR._allocate_array(::IR._MetalBackend, ::Type{T}, dims::Tuple) where {T}
    return Metal.MtlArray{T}(undef, dims)
end

end
