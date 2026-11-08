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

# A second scan avoids allocation for small populations. Larger populations
# measured faster with the stored CDF and binary search.
const _CATEGORICAL_SCAN_LIMIT = 64

@inline function _prepare_categorical_scalar(rng, d::Distributions.Categorical)
    probabilities = _categorical_probabilities(d)
    _check_categorical_scalar_probabilities(probabilities)
    cpu_rng = IR.MLDataDevices.CPUDevice()(rng)
    if length(probabilities) <= _CATEGORICAL_SCAN_LIMIT
        total = IR._weight_total_cpu(probabilities)
        return cpu_rng, total, probabilities, false
    end
    _, total, cumulative = IR._prepare_weight_scan(cpu_rng, probabilities, false)
    return cpu_rng, total, cumulative, true
end

@inline function _draw_categorical_unchecked(rng, total::Float64, prepared, is_cumulative)
    scale = total < IR._WEIGHT_SCALE_BOUND ? IR._WEIGHT_SCALE : 1.0
    threshold = IR._weighted_threshold(rng, rng.position, total * scale)
    is_cumulative && return IR._weighted_cdf_index(prepared, threshold)
    cumulative = 0.0
    # Match the prepared CDF's left fold and tiny-total scaling without storing it.
    for ordinal = 1:length(prepared)
        cumulative += Float64(IR._population_value(prepared, UInt64(ordinal)))
        threshold < cumulative * scale && return ordinal
    end
    return length(prepared)
end

@inline function _draw_categorical_next(rng, d::Distributions.Categorical)
    cpu_rng, total, prepared, is_cumulative = _prepare_categorical_scalar(rng, d)
    next_rng = IR._sampling_reservation(rng, 1, IR._WEIGHT_BITS)
    return _draw_categorical_unchecked(cpu_rng, total, prepared, is_cumulative), next_rng
end

@inline function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
)
    return _draw_categorical_next(rng, d)
end

@inline function IR.rand_at(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
    index::Integer,
)
    _, total, prepared, is_cumulative = _prepare_categorical_scalar(rng, d)
    addressed = IR._addressed_rng(rng, IR._WEIGHT_BITS, index)
    addressed_cpu_rng = IR.MLDataDevices.CPUDevice()(addressed)
    return _draw_categorical_unchecked(addressed_cpu_rng, total, prepared, is_cumulative)
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
        IR._fill_weighted_cpu!(
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
        IR._fill_backend(rng.device, destination),
        rng,
        labels,
        prepared,
        total,
        cumulative,
        destination,
    )
    return destination, next_rng
end

@inline function _rand_categorical_next_fill!(rng, d, destination, threaded::Bool)
    IR._check_fill_device(rng, destination)
    IR._check_serviceability(rng, d)
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

@inline function _rand_categorical_next_array(rng, d, dims, threaded::Bool)
    IR._check_serviceability(rng, d)
    probabilities, prepared, total, cumulative = _prepare_categorical(rng, d)
    destination = IR._allocate_draw_array(rng.device, Int, dims)
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

@inline function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return _rand_categorical_next_array(rng, d, (dim1, dims...), threaded)
end

@inline function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Categorical,
    destination::AbstractArray{Int};
    threaded::Bool = false,
)
    return _rand_categorical_next_fill!(rng, d, destination, threaded)
end
