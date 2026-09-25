@noinline function _fill_device_mismatch(generator_device, destination_device)
    throw(
        ArgumentError(
            "destination device differs from the generator device: generator on " *
            "$generator_device, destination on $destination_device",
        ),
    )
end

# The destination device is read once: a destination may count the query.
@inline function _check_fill_device(rng::_ScalarUniformGenerators, destination)
    generator_device = MLDataDevices.get_device_type(rng.device)
    destination_device = MLDataDevices.get_device_type(destination)
    generator_device <: destination_device ||
        _fill_device_mismatch(generator_device, destination_device)
    return nothing
end

@inline _check_serviceability(rng, ::Type) = nothing

@noinline function _host_only_type(::Type{T}) where {T}
    throw(ArgumentError("$T draws run on the CPU only; move the generator to the CPU"))
end

# Device kernels hold no 128-bit integers, so these types stay on the host.
@inline _check_serviceability(
    ::_BackendGenerators{<:Union{_CUDABackend,_AMDGPUBackend}},
    ::Type{T},
) where {T<:_WideInteger} = _host_only_type(T)
@inline _check_serviceability(rng, range::AbstractRange) =
    _check_serviceability(rng, eltype(range))

@noinline function _sampling_device_mismatch(noun)
    throw(ArgumentError("$noun device differs from the generator device"))
end

@inline function _check_sampling_device(rng, object, noun)
    device = MLDataDevices.get_device(object)
    device === nothing && return true
    device isa MLDataDevices.get_device_type(rng.device) || _sampling_device_mismatch(noun)
    return false
end

@inline function _check_population_device(rng, population)
    agnostic = _check_sampling_device(rng, population, "population")
    (population isa AbstractArray || agnostic) || _sampling_device_mismatch("population")
    return agnostic
end

# Weighted sampling folds Float64 weights; a backend without Float64 rejects it.
@inline _check_weighted_serviceability(rng) = nothing

@inline function _check_sampling_fill_device(rng, destination::Array)
    rng.device isa _CPUBackend || _fill_device_mismatch(
        MLDataDevices.get_device_type(rng.device),
        MLDataDevices.CPUDevice,
    )
    return nothing
end

@inline function _check_sampling_fill_device(rng, destination::AbstractArray)
    storage = parent(destination)
    storage === destination && return _check_fill_device(rng, destination)
    return _check_sampling_fill_device(rng, storage)
end
