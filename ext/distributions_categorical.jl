@inline _categorical_probabilities(d::Distributions.Categorical) = Distributions.probs(d)

@inline function _check_categorical_scalar_probabilities(probabilities)
    device = IR.MLDataDevices.get_device(probabilities)
    (device === nothing || device isa IR.MLDataDevices.CPUDevice) ||
        IR._sampling_device_mismatch("probabilities")
    return nothing
end

@inline function _categorical_probability_agnostic(rng, probabilities)
    device = IR.MLDataDevices.get_device(probabilities)
    if rng.device isa IR._CPUBackend
        device === nothing && return true
        device isa IR.MLDataDevices.CPUDevice ||
            IR._sampling_device_mismatch("probabilities")
        return false
    end
    device === nothing && IR._sampling_device_mismatch("probabilities")
    device isa IR.MLDataDevices.get_device_type(rng.device) ||
        IR._sampling_device_mismatch("probabilities")
    return false
end

@inline function _prepare_categorical_scalar(rng, d::Distributions.Categorical)
    probabilities = _categorical_probabilities(d)
    _check_categorical_scalar_probabilities(probabilities)
    cpu_rng = IR.MLDataDevices.CPUDevice()(rng)
    _, total, cumulative = IR._prepare_weight_scan(cpu_rng, probabilities, false)
    return cpu_rng, total, cumulative
end

@inline function _draw_categorical_unchecked(rng, total::Float64, cumulative)
    threshold = IR._weighted_threshold(rng, rng.position, total)
    return IR._weighted_cdf_index(cumulative, threshold)
end

@inline function _draw_categorical(rng, d::Distributions.Categorical)
    cpu_rng, total, cumulative = _prepare_categorical_scalar(rng, d)
    IR._sampling_reservation(rng, 1, IR._WEIGHT_BITS)
    return _draw_categorical_unchecked(cpu_rng, total, cumulative)
end

@inline function _draw_categorical_next(rng, d::Distributions.Categorical)
    cpu_rng, total, cumulative = _prepare_categorical_scalar(rng, d)
    next_rng = IR._sampling_reservation(rng, 1, IR._WEIGHT_BITS)
    return _draw_categorical_unchecked(cpu_rng, total, cumulative), next_rng
end

@inline function Random.rand(rng::IR._ScalarUniformGenerators, d::Distributions.Categorical)
    return _draw_categorical(rng, d)
end

@inline function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
)
    return _draw_categorical_next(rng, d)
end

@inline function IR.randat(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
    index::Integer,
)
    cpu_rng, total, cumulative = _prepare_categorical_scalar(rng, d)
    addressed = IR._addressed_rng(rng, IR._WEIGHT_BITS, index)
    addressed_cpu_rng = IR.MLDataDevices.CPUDevice()(addressed)
    return _draw_categorical_unchecked(addressed_cpu_rng, total, cumulative)
end

@inline function _prepare_categorical(rng, d::Distributions.Categorical)
    probabilities = _categorical_probabilities(d)
    agnostic = _categorical_probability_agnostic(rng, probabilities)
    prepared, total, cumulative = IR._prepare_weight_scan(rng, probabilities, agnostic)
    return probabilities, prepared, total, cumulative
end

@inline function _fill_categorical_prepared!(
    rng,
    probabilities,
    prepared,
    total,
    cumulative,
    destination,
    threaded,
)
    next_rng = IR._sampling_reservation(rng, length(destination), IR._WEIGHT_BITS)
    isempty(destination) && return destination, next_rng
    labels = Base.OneTo(length(probabilities))
    if !threaded && rng.device isa IR._CPUBackend
        IR._fill_weighted_samples_cpu_unchecked!(
            rng,
            rng.position,
            labels,
            total,
            cumulative,
            destination,
            1:length(destination),
        )
        return destination, next_rng
    end
    IR._fill_weighted_samples!(
        IR._fill_backend(destination),
        rng,
        labels,
        prepared,
        total,
        cumulative,
        destination,
    )
    return destination, next_rng
end

@inline function _rand_categorical_next_fill!(rng, d, destination, threaded)
    IR._check_fill_device(rng, destination)
    _check_serviceability(rng, Int)
    probabilities, prepared, total, cumulative = _prepare_categorical(rng, d)
    return _fill_categorical_prepared!(
        rng,
        probabilities,
        prepared,
        total,
        cumulative,
        destination,
        threaded,
    )
end

@inline function _rand_categorical_next_array(rng, d, dims)
    _check_serviceability(rng, Int)
    probabilities, prepared, total, cumulative = _prepare_categorical(rng, d)
    destination = IR._allocate_draw_array(rng.device, Int, dims)
    return _fill_categorical_prepared!(
        rng,
        probabilities,
        prepared,
        total,
        cumulative,
        destination,
        true,
    )
end

@inline function Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
    dim1::Integer,
    dims::Integer...,
)
    destination, _ = _rand_categorical_next_array(rng, d, (dim1, dims...))
    return destination
end

@inline function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
    dim1::Integer,
    dims::Integer...,
)
    return _rand_categorical_next_array(rng, d, (dim1, dims...))
end

@inline function Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
    destination::AbstractArray{Int};
    threaded = true,
)
    result, _ =
        _rand_categorical_next_fill!(rng, d, destination, IR._check_threaded(threaded))
    return result
end

@inline function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
    destination::AbstractArray{Int};
    threaded = true,
)
    return _rand_categorical_next_fill!(rng, d, destination, IR._check_threaded(threaded))
end
