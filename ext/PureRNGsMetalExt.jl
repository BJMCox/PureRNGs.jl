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

# Cache host wrappers without creating a device context or compiling a kernel.
let rng_type = IR.Philox4x32{IR._MetalBackend,10},
    array_type = Metal.MtlArray{Float32,1,Metal.PrivateStorage}

    precompile(IR.rand_next, (rng_type, Type{Float32}, Int))
    precompile(IR.rand_next!, (rng_type, array_type))
    precompile(IR.randn_next!, (rng_type, array_type))
end

end
