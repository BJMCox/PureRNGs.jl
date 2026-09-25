const _WEIGHT_BITS = UInt16(53)
const _WEIGHTED_LOOKUP_LANES = 32

@noinline function _invalid_weights()
    throw(
        ArgumentError(
            "weights must be finite and non-negative, with a finite positive total",
        ),
    )
end

@noinline function _invalid_weight_values()
    throw(ArgumentError("weights must be finite and non-negative"))
end

@noinline function _weight_length_error(weight_count, cardinality)
    throw(
        ArgumentError(
            "got $weight_count weights for a population of $cardinality elements",
        ),
    )
end

function _collect_weights(weights)
    converted = Vector{Float64}(undef, length(weights))
    invalid = false
    for ordinal in eachindex(converted)
        weight = Float64(_population_value(weights, UInt64(ordinal)))
        converted[ordinal] = weight
        invalid |= !isfinite(weight) || weight < zero(Float64)
    end
    invalid && _invalid_weight_values()
    return converted
end

@inline function _fold_weights_cpu(weights)
    cumulative = Vector{Float64}(undef, length(weights))
    total = zero(Float64)
    invalid = false
    for ordinal = 1:length(weights)
        weight = Float64(_population_value(weights, UInt64(ordinal)))
        invalid |= !isfinite(weight) || weight < zero(Float64)
        total += weight
        cumulative[ordinal] = total
    end
    invalid |= !isfinite(total) || total <= zero(Float64)
    invalid && _invalid_weights()
    return total, cumulative
end

"""
    WeightTable(weights)

Prepared weights for repeated weighted sampling. Holds the cumulative table the
weighted forms build on every call, so a caller with fixed weights pays that cost
once. Accepted wherever a weight vector is. Draws with a table equal draws with the
weights it was built from. A table lives on the device of its weights and serves
generators on that device.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> pop = [10, 20, 30, 40];

julia> weights = [1.0, 1.0, 1.0, 7.0];

julia> table = WeightTable(weights);

julia> randsample(rng, pop, table, 6)
6-element Vector{Int64}:
 20
 20
 40
 40
 40
 10

julia> randsample(rng, pop, table, 6) == randsample(rng, pop, weights, 6)
true
```
"""
struct WeightTable{T,V<:AbstractVector{Float64}}
    total::T
    cumulative::V
end

# Device weights fold on their device, where the total stays as a one-element
# array for the threshold kernel.
function WeightTable(weights)
    device = MLDataDevices.get_device(weights)
    device isa MLDataDevices.AbstractGPUDevice ||
        return WeightTable(_fold_weights_cpu(weights)...)
    _, total, cumulative = _fold_device_weights(_backend_token(device), weights, false)
    return WeightTable(total, cumulative)
end

function _fold_device_weights end

@inline _check_sampling_device(rng, table::WeightTable, noun) =
    _check_sampling_device(rng, table.cumulative, noun)

@inline _weight_count(weights) = UInt64(length(weights))
@inline _weight_count(table::WeightTable) = UInt64(length(table.cumulative))

function _prepare_weight_scan(rng::_CPUGenerators, weights, _agnostic::Bool)
    total, cumulative = _fold_weights_cpu(weights)
    return nothing, total, cumulative
end

@inline _prepare_weight_scan(rng::_CPUGenerators, table::WeightTable, _agnostic::Bool) =
    (nothing, table.total, table.cumulative)
@inline _prepare_weight_scan(rng, table::WeightTable, _agnostic::Bool) =
    (nothing, table.total, table.cumulative)

# `threshold < cumulative[end]`; odd spans overlap one index between both halves.
Base.@propagate_inbounds function _weighted_cdf_index(cumulative, threshold)
    lower = 1
    span = length(cumulative)
    while span > 1
        half = span >>> 1
        middle = lower + half - 1
        lower = ifelse(!(threshold < cumulative[middle]), middle + 1, lower)
        span -= half
    end
    return lower
end

@inline function _fill_weighted_cpu!(
    rng::_CPUGenerators,
    position,
    population,
    total::Float64,
    cumulative,
    destination,
    ordinals::UnitRange{Int},
)
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    indices = eachindex(destination)
    # One range check covers every store below; per-element checks in these
    # loops cost up to 25% (BenchmarkTools, serial 2^16 weighted fills).
    checkbounds(indices, (firstindex(indices) - 1) .+ ordinals)
    ordinal = first(ordinals)
    last_ordinal = last(ordinals)
    if length(ordinals) >= _WEIGHTED_LOOKUP_LANES
        thresholds = Vector{Float64}(undef, _WEIGHTED_LOOKUP_LANES)
        lower = Vector{Int}(undef, _WEIGHTED_LOOKUP_LANES)
        while ordinal <= last_ordinal - (_WEIGHTED_LOOKUP_LANES - 1)
            for lane = 1:_WEIGHTED_LOOKUP_LANES
                raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(53))
                thresholds[lane] = _weighted_threshold_from_bits(raw, total)
                lower[lane] = 1
            end
            span = length(cumulative)
            # The search keeps every `middle` inside `cumulative` and every lane
            # inside the two scratch vectors; checking them costs 10-14%.
            while span > 1
                half = span >>> 1
                @inbounds @simd for lane = 1:_WEIGHTED_LOOKUP_LANES
                    middle = lower[lane] + half - 1
                    lower[lane] = ifelse(
                        !(thresholds[lane] < cumulative[middle]),
                        middle + 1,
                        lower[lane],
                    )
                end
                span -= half
            end
            @inbounds for lane = 1:_WEIGHTED_LOOKUP_LANES
                population_index = lower[lane]
                index = _destination_index(indices, ordinal + lane - 1)
                destination[index] = _population_value(population, UInt64(population_index))
            end
            ordinal += _WEIGHTED_LOOKUP_LANES
        end
    end
    @inbounds while ordinal <= last_ordinal
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(53))
        threshold = _weighted_threshold_from_bits(raw, total)
        population_index = _weighted_cdf_index(cumulative, threshold)
        index = _destination_index(indices, ordinal)
        destination[index] = _population_value(population, UInt64(population_index))
        ordinal += 1
    end
    return destination
end

@inline function _fill_weighted_samples!(
    ::_CPUBackend,
    rng::_CPUGenerators,
    population,
    _weights,
    total::Float64,
    cumulative,
    destination,
)
    # Lane-aligned chunks keep every workitem on the vectorized lookup path.
    chunk_elements = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_WEIGHT_BITS))
    chunk_elements -= chunk_elements % _WEIGHTED_LOOKUP_LANES
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), _WEIGHT_BITS)
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        _fill_weighted_cpu!(
            rng,
            position,
            population,
            total,
            cumulative,
            destination,
            first:last,
        )
    end
    return destination
end

# The device scan runs a KernelAbstractions kernel, so the extension owns every
# method of this launcher.
function _launch_weighted_scan! end

@inline function _fill_weighted_samples!(
    backend,
    rng,
    population,
    _weights,
    total,
    cumulative,
    destination,
)
    thresholds = _allocate_array(rng.device, Float64, (length(destination),))
    _fill_weighted_thresholds!(backend, rng, total, thresholds)
    return _launch_weighted_scan!(backend, population, cumulative, thresholds, destination)
end

@inline function _weighted_threshold_from_bits(raw::UInt64, total::Float64)
    uniform = _from_bits(Float64, raw)
    return min(uniform * total, prevfloat(total))
end

@inline function _weighted_threshold(rng, position, total::Float64)
    raw = _extract_bits_unchecked(rng, _position_block(position), position.bit, Val(53))
    return _weighted_threshold_from_bits(raw, total)
end

@inline function _fill_weighted_thresholds!(::_CPUBackend, rng, total::Float64, thresholds)
    isempty(thresholds) && return thresholds
    cursor = _dense_cursor(rng, _position_block(rng.position), rng.position.bit)
    for index in eachindex(thresholds)
        raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(53))
        thresholds[index] = _weighted_threshold_from_bits(raw, total)
    end
    return thresholds
end

@noinline function _weighted_unique_count_error(count, positive)
    throw(
        ArgumentError(
            "cannot draw $count elements without replacement: only $positive weights are positive",
        ),
    )
end

@noinline function _weight_table_unique_error()
    throw(
        ArgumentError(
            "replace = false needs the weight vector; a WeightTable holds only cumulative sums",
        ),
    )
end

# A table keeps cumulative sums, whose differences are not the weights, so a
# table sample could not equal the weight-vector sample.
_unique_weights(rng::_CPUGenerators, ::WeightTable, _agnostic::Bool) =
    _weight_table_unique_error()
_unique_weights(rng, ::WeightTable, _agnostic::Bool) = _weight_table_unique_error()
_unique_weights(rng::_CPUGenerators, weights, _agnostic::Bool) = _collect_weights(weights)
function _unique_weights(rng, weights, agnostic::Bool)
    agnostic && return _transfer_array(rng.device, _collect_weights(weights))
    converted = Float64.(weights)
    any(weight -> !isfinite(weight) || weight < zero(Float64), converted) &&
        _invalid_weight_values()
    return converted
end

# The key orders `E / w` for every finite weight, where Float64 division would
# overflow. With `w = s * 2^e` and `s` in [1, 2), `E / s` is a normal Float64
# for every exponential draw, and `-e` moves into the key's 12-bit exponent
# field. Zero weights sort last.
const _RACE_EXPONENT_BIAS = 64

@inline function _race_key(exponential::Float64, weight::Float64)
    iszero(weight) && return typemax(UInt64)
    ratio = reinterpret(UInt64, exponential / significand(weight))
    return ratio + ((_RACE_EXPONENT_BIAS - exponent(weight)) % UInt64) << 52
end

# The leading `count` indices in key order, ties by index. A device sort need
# not be stable, so the device then orders each tied run that reaches the
# leading `count` by index. Zero-weight keys tie, but they sort after `count`.
_race_order(rng, keys::Vector{UInt64}, count::Int) = partialsortperm(keys, 1:count)

function _race_order(rng, keys::AbstractVector{UInt64}, count::Int)
    order = sortperm!(similar(keys, Int), keys)
    _order_device_key_runs!(_fill_backend(rng.device, keys), order, keys, count, nothing)
    return order[1:count]
end

# A weighted sample without replacement orders the population by `E / w`, one
# exponential draw `E` per element (Efraimidis and Spirakis 2006, with
# `E = -log(u)`). The smallest `E / w` belongs to element `i` with probability
# `w[i] / sum(w)`, and the others restart by memorylessness, so the order is
# that of successive draws proportional to the remaining weights. It consumes 52
# bits per population element for every `count`.
function _weighted_unique_sample(
    rng,
    indexed,
    weights,
    agnostic::Bool,
    count::Int,
    threaded::Bool,
)
    converted = _unique_weights(rng, weights, agnostic)
    positive = Base.count(>(zero(Float64)), converted)
    count <= positive || _weighted_unique_count_error(count, positive)
    exponentials, next_rng = randexp_next(rng, Float64, length(converted); threaded)
    keys = _race_key.(exponentials, converted)
    order = _race_order(rng, keys, count)
    return _gather(_fill_backend(rng.device, keys), vec(indexed), order), next_rng
end

function _randsample_next_weighted(
    rng,
    population,
    weights,
    requested_count,
    replace::Bool,
    threaded::Bool,
)
    population_agnostic = _check_population_device(rng, population)
    weights_agnostic = _check_sampling_device(rng, weights, "weights")
    _check_weighted_serviceability(rng)

    count = requested_count === nothing ? nothing : _sampling_count(requested_count)
    _prevalidate_sampling_cardinality(population, count)
    indexed = _prepare_population(rng.device, population, population_agnostic)
    cardinality = _sampling_cardinality(indexed)
    count === nothing && (count = _sampling_count(cardinality, nothing))
    count > 0 && iszero(cardinality) && _empty_sampling_population(count)
    _weight_count(weights) == cardinality ||
        _weight_length_error(_weight_count(weights), cardinality)
    replace || return _weighted_unique_sample(
        rng,
        indexed,
        weights,
        weights_agnostic,
        count,
        threaded,
    )

    converted, total, cumulative = _prepare_weight_scan(rng, weights, weights_agnostic)
    next_rng = _sampling_reservation(rng, count, _WEIGHT_BITS)
    destination = _allocate_sampling_result(rng, indexed, count)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_weighted_cpu!(
            rng,
            rng.position,
            indexed,
            total,
            cumulative,
            destination,
            1:count,
        )
        return destination, next_rng
    end

    backend = _fill_backend(rng.device, destination)
    _fill_weighted_samples!(
        backend,
        rng,
        indexed,
        converted,
        total,
        cumulative,
        destination,
    )
    return destination, next_rng
end

function _randsample_next_weighted!(
    rng,
    population,
    weights,
    destination,
    replace::Bool,
    threaded::Bool,
)
    _check_sampling_fill_device(rng, destination)
    _check_weighted_serviceability(rng)
    population_agnostic = _check_population_device(rng, population)
    weights_agnostic = _check_sampling_device(rng, weights, "weights")
    _check_sampling_population_overlap(destination, population)
    _prevalidate_sampling_cardinality(population, length(destination))
    indexed = _prepare_population(rng.device, population, population_agnostic)
    cardinality = _sampling_cardinality(indexed)
    _weight_count(weights) == cardinality ||
        _weight_length_error(_weight_count(weights), cardinality)
    _check_sampling_destination_eltype(destination, indexed)
    !isempty(destination) &&
        iszero(cardinality) &&
        _empty_sampling_population(length(destination))
    if !replace
        values, next_rng = _weighted_unique_sample(
            rng,
            indexed,
            weights,
            weights_agnostic,
            length(destination),
            threaded,
        )
        copyto!(destination, values)
        return destination, next_rng
    end

    converted, total, cumulative = _prepare_weight_scan(rng, weights, weights_agnostic)
    next_rng = _sampling_reservation(rng, length(destination), _WEIGHT_BITS)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_weighted_cpu!(
            rng,
            rng.position,
            indexed,
            total,
            cumulative,
            destination,
            1:length(destination),
        )
        return destination, next_rng
    end
    _fill_weighted_samples!(
        _fill_backend(rng.device, destination),
        rng,
        indexed,
        converted,
        total,
        cumulative,
        destination,
    )
    return destination, next_rng
end

@inline function randsample(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable};
    replace::Bool = true,
    threaded::Bool = false,
)
    return first(
        _randsample_next_weighted(rng, population, weights, nothing, replace, threaded),
    )
end


@inline function randsample(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
    count::Integer;
    replace::Bool = true,
    threaded::Bool = false,
)
    return first(
        _randsample_next_weighted(rng, population, weights, count, replace, threaded),
    )
end


@inline function randsample_next(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable};
    replace::Bool = true,
    threaded::Bool = false,
)
    return _randsample_next_weighted(rng, population, weights, nothing, replace, threaded)
end


@inline function randsample_next(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
    count::Integer;
    replace::Bool = true,
    threaded::Bool = false,
)
    return _randsample_next_weighted(rng, population, weights, count, replace, threaded)
end

@inline function randsample!(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
    destination::AbstractArray;
    replace::Bool = true,
    threaded::Bool = false,
)
    return first(
        _randsample_next_weighted!(
            rng,
            population,
            weights,
            destination,
            replace,
            threaded,
        ),
    )
end

@inline function randsample_next!(
    rng::AbstractPureRNG,
    population,
    weights::Union{AbstractVector{<:Real},WeightTable},
    destination::AbstractArray;
    replace::Bool = true,
    threaded::Bool = false,
)
    return _randsample_next_weighted!(
        rng,
        population,
        weights,
        destination,
        replace,
        threaded,
    )
end
