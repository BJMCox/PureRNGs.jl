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

@inline _check_sampling_serviceability(rng) = nothing

@inline function _check_sampling_fill_device(rng, destination::Array)
    rng.device isa _CPUBackend || _fill_device_mismatch()
    return nothing
end

@inline function _check_sampling_fill_device(rng, destination::AbstractArray)
    storage = parent(destination)
    storage === destination && return _check_fill_device(rng, destination)
    return _check_sampling_fill_device(rng, storage)
end

@noinline function _sampling_population_overlap()
    throw(ArgumentError("destination may not overlap the population"))
end

@noinline function _sampling_destination_eltype_mismatch()
    throw(ArgumentError("destination eltype differs from the population eltype"))
end

@inline function _check_sampling_population_overlap(destination, population)
    population isa AbstractArray &&
        Base.mightalias(destination, population) &&
        _sampling_population_overlap()
    return nothing
end

@inline function _check_sampling_destination_eltype(destination, population)
    eltype(destination) === eltype(population) || _sampling_destination_eltype_mismatch()
    return nothing
end

@inline _collect_population(population::AbstractArray) =
    [population[index] for index in CartesianIndices(axes(population))]
@inline _collect_population(population) = collect(population)

@inline _materialize_population(::_CPUBackend, population) = _collect_population(population)

@inline _prepare_population(::_CPUBackend, population::AbstractArray, agnostic::Bool) =
    population
@inline _prepare_population(
    ::_CPUBackend,
    population::AbstractRange{T},
    agnostic::Bool,
) where {T<:Integer} = population
@inline _prepare_population(
    device::_BackendToken,
    population::AbstractRange{T},
    agnostic::Bool,
) where {T<:Integer} = population
@inline _prepare_population(
    device::_BackendToken,
    population::AbstractArray,
    agnostic::Bool,
) = agnostic ? _materialize_population(device, population) : population
@inline _prepare_population(device::_BackendToken, population, agnostic::Bool) =
    _materialize_population(device, population)

@noinline function _sampling_cardinality_error()
    throw(ArgumentError("population cardinality exceeds typemax(UInt64)"))
end

@inline function _checked_sampling_cardinality(cardinality::Integer)
    0 <= cardinality <= typemax(UInt64) || _sampling_cardinality_error()
    return UInt64(cardinality)
end

@inline function _sampling_cardinality(population::OrdinalRange{T}) where {T<:_RangeInteger}
    isempty(population) && return zero(UInt64)
    first_value = Int128(first(population))
    last_value = Int128(last(population))
    stride = Int128(step(population))
    cardinality = UInt128(abs(last_value - first_value)) ÷ UInt128(abs(stride)) + 1
    return _checked_sampling_cardinality(cardinality)
end

@inline function _sampling_cardinality(population::OrdinalRange{<:Integer})
    isempty(population) && return zero(UInt64)
    cardinality =
        div(
            abs(big(last(population)) - big(first(population))),
            abs(big(step(population))),
        ) + 1
    return _checked_sampling_cardinality(cardinality)
end


@inline function _prevalidate_sampling_cardinality(population, requested_count)
    iterator_size = Base.IteratorSize(typeof(population))
    (iterator_size isa Base.HasLength || iterator_size isa Base.HasShape) || return nothing
    cardinality = _sampling_cardinality(population)
    requested_count === nothing && _sampling_count(cardinality, nothing)
    requested_count !== nothing &&
        requested_count > 0 &&
        iszero(cardinality) &&
        _empty_sampling_population()
    return nothing
end

@inline function _sampling_cardinality(population)
    cardinality = length(population)
    cardinality isa Integer || _sampling_cardinality_error()
    iszero(cardinality) && !isempty(population) && _sampling_cardinality_error()
    return _checked_sampling_cardinality(cardinality)
end

@noinline function _sampling_count_error()
    throw(ArgumentError("k must satisfy 0 <= k <= typemax(Int)"))
end

@inline function _sampling_count(count::Integer)
    0 <= count <= typemax(Int) || _sampling_count_error()
    return Int(count)
end

@inline function _sampling_count(cardinality::UInt64, ::Nothing)
    cardinality <= UInt64(typemax(Int)) ||
        throw(ArgumentError("a no-k sampling result length exceeds typemax(Int)"))
    return Int(cardinality)
end

@inline function _sampling_reservation(rng, count::Int, width::UInt16)
    bits_lo, bits_hi = _bit_span(UInt64(count), width)
    return _reserve(rng, bits_lo, bits_hi)
end

@inline function _allocate_sampling_result(rng, population, count::Int)
    return _allocate_array(rng.device, eltype(population), (count,))
end

@inline function _sampling_destination_index(indices, ordinal::Int)
    return @inbounds indices[firstindex(indices)+ordinal-1]
end

@inline function _population_value(
    population::AbstractRange{T},
    ordinal::UInt64,
) where {T<:_RangeInteger}
    return _range_value(population, ordinal - UInt64(1))
end

@inline function _population_value(population::AbstractRange, ordinal::UInt64)
    return @inbounds population[ordinal]
end

@inline function _population_value(population::AbstractArray, ordinal::UInt64)
    indices = CartesianIndices(axes(population))
    index = @inbounds indices[firstindex(indices)+Int(ordinal)-1]
    return @inbounds population[index]
end

KernelAbstractions.@kernel function _unweighted_sample_kernel!(
    rng,
    population,
    cardinality::UInt64,
    destination,
    width::UInt16,
)
    draw_ordinal = @index(Global, Linear)
    bits_lo, bits_hi = _bit_span(UInt64(draw_ordinal - 1), width)
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    population_ordinal = _range_offset(rng, position, cardinality) + UInt64(1)
    indices = eachindex(destination)
    index = _sampling_destination_index(indices, draw_ordinal)
    @inbounds destination[index] = _population_value(population, population_ordinal)
end

@inline function _fill_unweighted_cpu_unchecked!(
    rng,
    position,
    population,
    cardinality::UInt64,
    destination,
    width::UInt16,
    ordinals,
)
    isempty(ordinals) && return nothing
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    indices = eachindex(destination)
    if width == UInt16(64)
        @inbounds for ordinal in ordinals
            candidate, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
            population_ordinal = _reduce_range_candidate(candidate, cardinality) + UInt64(1)
            index = _sampling_destination_index(indices, ordinal)
            destination[index] = _population_value(population, population_ordinal)
        end
    else
        @inbounds for ordinal in ordinals
            hi, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
            lo, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
            population_ordinal = _reduce_range_candidate(lo, hi, cardinality) + UInt64(1)
            index = _sampling_destination_index(indices, ordinal)
            destination[index] = _population_value(population, population_ordinal)
        end
    end
    return nothing
end

@inline function _fill_unweighted_grouped_unchecked!(
    rng,
    position,
    population,
    cardinality::UInt64,
    destination,
    first_ordinal::Int,
    ::Val{2},
)
    cursor = _dense_cursor(rng, _position_block(position), position.bit)
    indices = eachindex(destination)
    @inbounds for offset = 0:1
        ordinal = first_ordinal + offset
        ordinal > length(destination) && break
        candidate, cursor = _take_dense_bits_unchecked(rng, cursor, Val(64))
        population_ordinal = _reduce_range_candidate(candidate, cardinality) + UInt64(1)
        index = _sampling_destination_index(indices, ordinal)
        destination[index] = _population_value(population, population_ordinal)
    end
    return nothing
end

KernelAbstractions.@kernel function _unweighted_sample_grouped_kernel!(
    rng,
    population,
    cardinality::UInt64,
    destination,
    group::Val{2},
)
    workitem = @index(Global, Linear)
    first_ordinal = (workitem - 1) * 2 + 1
    bits_lo, bits_hi = _bit_span(UInt64(first_ordinal - 1), UInt16(64))
    position = _advance_position_unchecked(rng, bits_lo, bits_hi)
    _fill_unweighted_grouped_unchecked!(
        rng,
        position,
        population,
        cardinality,
        destination,
        first_ordinal,
        group,
    )
end

@inline function _launch_unweighted_sample!(
    backend,
    rng,
    population,
    cardinality::UInt64,
    destination,
    width::UInt16,
)
    plan = _device_range_fill_plan(backend, rng, cardinality)
    if plan !== nothing
        group = plan[2]
        workitems = cld(length(destination), _fill_group_size(group))
        _unweighted_sample_grouped_kernel!(backend)(
            rng,
            population,
            cardinality,
            destination,
            group;
            ndrange = workitems,
        )
        return destination
    end
    _unweighted_sample_kernel!(backend)(
        rng,
        population,
        cardinality,
        destination,
        width;
        ndrange = length(destination),
    )
    return destination
end

@inline function _launch_unweighted_sample!(
    ::KernelAbstractions.CPU,
    rng,
    population,
    cardinality::UInt64,
    destination,
    width::UInt16,
)
    chunk_elements = Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(width))
    _run_chunks(length(destination), chunk_elements) do first, last
        bits_lo, bits_hi = _bit_span(UInt64(first - 1), width)
        position = _advance_position_unchecked(rng, bits_lo, bits_hi)
        _fill_unweighted_cpu_unchecked!(
            rng,
            position,
            population,
            cardinality,
            destination,
            width,
            first:last,
        )
    end
    return destination
end

function _randsample_next_unweighted!(rng, population, destination, threaded::Bool)
    _check_sampling_fill_device(rng, destination)
    _check_sampling_serviceability(rng)
    agnostic = _check_population_device(rng, population)
    _check_sampling_population_overlap(destination, population)
    _prevalidate_sampling_cardinality(population, length(destination))
    indexed = _prepare_population(rng.device, population, agnostic)
    cardinality = _sampling_cardinality(indexed)
    _check_sampling_destination_eltype(destination, indexed)
    !isempty(destination) && iszero(cardinality) && _empty_sampling_population()

    width = _range_bits(cardinality)
    next_rng = _sampling_reservation(rng, length(destination), width)
    isempty(destination) && return destination, next_rng
    if !threaded && rng.device isa _CPUBackend
        _fill_unweighted_cpu_unchecked!(
            rng,
            rng.position,
            indexed,
            cardinality,
            destination,
            width,
            1:length(destination),
        )
        return destination, next_rng
    end
    _launch_unweighted_sample!(
        _fill_backend(destination),
        rng,
        indexed,
        cardinality,
        destination,
        width,
    )
    return destination, next_rng
end

@noinline function _empty_sampling_population()
    throw(ArgumentError("population must be non-empty when k is positive"))
end

function _randsample_next_unweighted(rng, population, requested_count)
    agnostic = _check_population_device(rng, population)
    _check_sampling_serviceability(rng)
    count = requested_count === nothing ? nothing : _sampling_count(requested_count)
    _prevalidate_sampling_cardinality(population, count)
    indexed = _prepare_population(rng.device, population, agnostic)
    cardinality = _sampling_cardinality(indexed)
    count === nothing && (count = _sampling_count(cardinality, nothing))
    count > 0 && iszero(cardinality) && _empty_sampling_population()

    width = _range_bits(cardinality)
    next_rng = _sampling_reservation(rng, count, width)
    destination = _allocate_sampling_result(rng, indexed, count)
    isempty(destination) && return destination, next_rng
    backend = _fill_backend(destination)
    _launch_unweighted_sample!(backend, rng, indexed, cardinality, destination, width)
    return destination, next_rng
end

"""
    randsample(rng, population[, count])
    randsample(rng, population, weights[, count])

Sample with replacement from `population`. Without `count`, return as many
draws as the population has elements. With `weights`, use non-negative finite
weights proportional to the desired probabilities.

The no-count form returns `length(pop)` samples, unlike `StatsBase.sample(rng, a)`,
which returns one element. `randsample(rng, pop, 1)` returns a one-element vector.

The result is a vector on the generator's device. This convenience form does
not return the advanced generator; use [`randsample_next`](@ref) when subsequent
draws must continue after the sample.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> pop = [10, 20, 30, 40];

julia> randsample(rng, pop)
4-element Vector{Int64}:
 10
 40
 30
 10

julia> randsample(rng, pop, 1)
1-element Vector{Int64}:
 10
```
"""
@inline function randsample(rng::AbstractPureRNG, population)
    return first(_randsample_next_unweighted(rng, population, nothing))
end

@inline function randsample(rng::AbstractPureRNG, population, count::Integer)
    return first(_randsample_next_unweighted(rng, population, count))
end

"""
    randsample_next(rng, population[, count]) -> (values, next_rng)
    randsample_next(rng, population, weights[, count]) -> (values, next_rng)

Sample with replacement from `population` and return the advanced immutable
generator with the result. Without `count`, return as many draws as the
population has elements. With `weights`, use non-negative finite weights
proportional to the desired probabilities.

The result is a vector on the generator's device. The input generator never
changes.
"""
@inline function randsample_next(rng::AbstractPureRNG, population)
    return _randsample_next_unweighted(rng, population, nothing)
end

@inline function randsample_next(rng::AbstractPureRNG, population, count::Integer)
    return _randsample_next_unweighted(rng, population, count)
end

"""
    randsample!(rng, population[, weights], destination; threaded=true) -> destination

Sample with replacement into `destination`. Its length determines the number of
draws, and it must have exactly the prepared population element type. Return the
identical destination. `threaded=false` keeps an unweighted CPU fill on the
calling task.

Inputs and the complete random span are validated before writing. A destination
that might alias `population` is rejected; it may alias `weights` after the
weights have been privately prepared. Empty destinations still validate inputs
and consume no bits. Weighted fills may allocate preparation scratch space.

# Examples

```jldoctest
julia> rng = Philox4x32(20250918);

julia> destination = zeros(Int, 5);

julia> randsample!(rng, [10, 20, 30, 40], destination)
5-element Vector{Int64}:
 10
 40
 30
 10
 10
```
"""
@inline function randsample!(
    rng::AbstractPureRNG,
    population,
    destination::AbstractArray;
    threaded::Bool = true,
)
    return first(_randsample_next_unweighted!(rng, population, destination, threaded))
end

"""
    randsample_next!(rng, population[, weights], destination; threaded=true) -> (destination, next_rng)

Sample with replacement into `destination` and return the advanced immutable
generator. The fill and validation rules are the same as for [`randsample!`](@ref).
The input generator is not changed.
"""
@inline function randsample_next!(
    rng::AbstractPureRNG,
    population,
    destination::AbstractArray;
    threaded::Bool = true,
)
    return _randsample_next_unweighted!(rng, population, destination, threaded)
end
