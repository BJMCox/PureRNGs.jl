module PureRNGsMetalExt

import PureRNGs
import Metal

const IR = PureRNGs
const _MetalGenerators = IR._BackendGenerators{IR._MetalBackend}

# Metal has no Float64 or 128-bit integer arithmetic. Every other draw runs on
# the device; weighted sampling stays off it, since its cumulative sums are
# Float64.
@noinline function _metal_type_error(::Type{T}) where {T}
    throw(
        ArgumentError(
            "$T draws need Float64 or 128-bit integer arithmetic, which Metal lacks; " *
            "move the generator to the CPU",
        ),
    )
end

@noinline function _metal_weighted_error()
    throw(
        ArgumentError(
            "weighted sampling folds Float64 weights, which Metal lacks; move the " *
            "generator to the CPU",
        ),
    )
end

@inline IR._check_serviceability(
    ::_MetalGenerators,
    ::Type{T},
) where {T<:Union{Float64,Complex{Float64},IR._WideInteger}} = _metal_type_error(T)
@inline IR._check_weighted_serviceability(::_MetalGenerators) = _metal_weighted_error()
IR._fold_device_weights(::IR._MetalBackend, weights, ::Bool) = _metal_weighted_error()

@inline function IR._allocate_array(::IR._MetalBackend, ::Type{T}, dims::Tuple) where {T}
    return Metal.MtlArray{T}(undef, dims)
end

@inline IR._materialize_population(::IR._MetalBackend, population) =
    Metal.MtlArray(IR._collect_population(population))

# Cache host wrappers without creating a device context or compiling a kernel.
let rng_type = IR.Philox4x32{IR._MetalBackend,10},
    array_type = Metal.MtlArray{Float32,1,Metal.PrivateStorage}

    precompile(IR.rand_next, (rng_type, Type{Float32}, Int))
    precompile(IR.rand_next!, (rng_type, array_type))
    precompile(IR.randn_next!, (rng_type, array_type))
end

end
