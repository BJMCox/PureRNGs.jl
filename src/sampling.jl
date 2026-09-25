@noinline function _sampling_population_overlap()
    throw(
        ArgumentError(
            "the destination shares memory with the population; sample into a separate array",
        ),
    )
end

@noinline function _sampling_destination_eltype_mismatch(destination_type, population_type)
    throw(
        ArgumentError(
            "destination element type $destination_type differs from the population element type $population_type",
        ),
    )
end

@inline function _check_sampling_population_overlap(destination, population)
    population isa AbstractArray &&
        Base.mightalias(destination, population) &&
        _sampling_population_overlap()
    return nothing
end

@inline function _check_sampling_destination_eltype(destination, population)
    eltype(destination) === eltype(population) ||
        _sampling_destination_eltype_mismatch(eltype(destination), eltype(population))
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
    throw(
        ArgumentError(
            "population length must be an integer no larger than typemax(UInt64)",
        ),
    )
end

@inline function _checked_sampling_cardinality(cardinality::Integer)
    0 <= cardinality <= typemax(UInt64) || _sampling_cardinality_error()
    return UInt64(cardinality)
end

@inline function _sampling_cardinality(
    population::OrdinalRange{T},
) where {T<:_RangeInteger64}
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
        _empty_sampling_population(requested_count)
    return nothing
end

@inline function _sampling_cardinality(population)
    cardinality = length(population)
    cardinality isa Integer || _sampling_cardinality_error()
    iszero(cardinality) && !isempty(population) && _sampling_cardinality_error()
    return _checked_sampling_cardinality(cardinality)
end

@noinline function _sampling_count_error(count)
    throw(ArgumentError("sample count must be in 0:typemax(Int), got $count"))
end

@inline function _sampling_count(count::Integer)
    0 <= count <= typemax(Int) || _sampling_count_error(count)
    return Int(count)
end

@noinline function _uncounted_sample_error(cardinality)
    throw(
        ArgumentError(
            "a sample without a count has one element per population element, and $cardinality exceeds typemax(Int); pass a count",
        ),
    )
end

@inline function _sampling_count(cardinality::UInt64, ::Nothing)
    cardinality <= UInt64(typemax(Int)) || _uncounted_sample_error(cardinality)
    return Int(cardinality)
end

@inline function _sampling_reservation(rng, count::Int, width::UInt16)
    bits_lo, bits_hi = _bit_span(UInt64(count), width)
    return _reserve(rng, bits_lo, bits_hi)
end

@inline function _allocate_sampling_result(rng, population, count::Int)
    return _allocate_array(rng.device, eltype(population), (count,))
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
    return @inbounds population[_destination_index(
        CartesianIndices(axes(population)),
        Int(ordinal),
    )]
end

# The population codec reduces a candidate over the cardinality and gathers the
# population element at that ordinal, so the transformed scaffold serves
# unweighted sampling unchanged.
struct _PopulationCodec{P}
    population::P
    cardinality::UInt64
end

@inline _fill_width(codec::_PopulationCodec, ::Type) = _range_bits(codec.cardinality)

@inline _fill_chunk_elements(codec::_PopulationCodec, ::Type) =
    Int(_CPU_FILL_CHUNK_BITS ÷ UInt64(_range_bits(codec.cardinality)))

@inline function _codec_take(codec::_PopulationCodec, rng, cursor, ::Type)
    offset, cursor = _take_range_offset(rng, cursor, codec.cardinality)
    return _population_value(codec.population, offset + UInt64(1)), cursor
end

@inline function _transformed_draw_unchecked(codec::_PopulationCodec, rng, position, ::Type)
    ordinal = _range_offset(rng, position, codec.cardinality) + UInt64(1)
    return _population_value(codec.population, ordinal)
end

function _randsample_next_unweighted!(
    rng,
    population,
    destination,
    replace::Bool,
    threaded::Bool,
)
    _check_sampling_fill_device(rng, destination)
    agnostic = _check_population_device(rng, population)
    _check_sampling_population_overlap(destination, population)
    _prevalidate_sampling_cardinality(population, length(destination))
    indexed = _prepare_population(rng.device, population, agnostic)
    cardinality = _sampling_cardinality(indexed)
    _check_sampling_destination_eltype(destination, indexed)
    !isempty(destination) &&
        iszero(cardinality) &&
        _empty_sampling_population(length(destination))

    if !replace
        values, next_rng =
            _unique_sample(rng, indexed, cardinality, length(destination), threaded)
        copyto!(destination, values)
        return destination, next_rng
    end
    codec = _PopulationCodec(indexed, cardinality)
    return _fill_prevalidated!(rng, destination, threaded, codec)
end

@noinline function _unique_count_error(count, cardinality)
    throw(
        ArgumentError(
            "cannot draw $count elements without replacement from a population of $cardinality",
        ),
    )
end

# A sample without replacement is the leading `count` elements of the shuffled
# population, so it consumes 64 bits per population element for every `count`.
function _unique_sample(rng, indexed, cardinality::UInt64, count::Int, threaded::Bool)
    count <= cardinality || _unique_count_error(count, cardinality)
    order, next_rng = _randperm_next(rng, Int(cardinality), threaded)
    return _gather(_fill_backend(rng.device, order), vec(indexed), order[1:count]), next_rng
end

@noinline function _empty_sampling_population(count)
    throw(ArgumentError("cannot draw $count elements from an empty population"))
end

function _randsample_next_unweighted(
    rng,
    population,
    requested_count,
    replace::Bool,
    threaded::Bool,
)
    agnostic = _check_population_device(rng, population)
    count = requested_count === nothing ? nothing : _sampling_count(requested_count)
    _prevalidate_sampling_cardinality(population, count)
    indexed = _prepare_population(rng.device, population, agnostic)
    cardinality = _sampling_cardinality(indexed)
    count === nothing && (count = _sampling_count(cardinality, nothing))
    count > 0 && iszero(cardinality) && _empty_sampling_population(count)
    replace || return _unique_sample(rng, indexed, cardinality, count, threaded)

    destination = _allocate_sampling_result(rng, indexed, count)
    codec = _PopulationCodec(indexed, cardinality)
    return _fill_prevalidated!(rng, destination, threaded, codec)
end

"""
    randsample(rng, population[, count]; replace=true, threaded=false)
    randsample(rng, population, weights[, count]; replace=true, threaded=false)

Sample from `population`. Without `count`, return as many draws as the
population has elements. With `weights`, use non-negative finite weights
proportional to the desired probabilities.

Sampling is with replacement by default. With `replace=false`, the sample is the
first `count` elements of `shuffle_next(rng, collect(population))`: it consumes
64 bits per population element for every `count`, and `count` may not exceed
the population.

Weighted sampling with `replace=false` draws `E = randexp_next(rng, Float64, n)`
and returns the population in increasing order of `E[i] / weights[i]`, ties by
index, truncated to `count`. This is successive sampling proportional to the
remaining weights (Efraimidis and Spirakis 2006). It consumes 52 bits per
population element for every `count`, and `count` may not exceed the number of
positive weights. It does not accept a `WeightTable`. Device exponentials can
differ from CPU exponentials in the last ulp, so a device sample can differ
from the CPU sample when two ratios are that close.

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
@inline function randsample(
    rng::AbstractPureRNG,
    population;
    replace::Bool = true,
    threaded::Bool = false,
)
    return first(_randsample_next_unweighted(rng, population, nothing, replace, threaded))
end

@inline function randsample(
    rng::AbstractPureRNG,
    population,
    count::Integer;
    replace::Bool = true,
    threaded::Bool = false,
)
    return first(_randsample_next_unweighted(rng, population, count, replace, threaded))
end

"""
    randsample_next(rng, population[, count]; replace=true, threaded=false) -> (values, next_rng)
    randsample_next(rng, population, weights[, count]; replace=true, threaded=false) -> (values, next_rng)

Sample from `population` and return the advanced immutable generator with the
result. Without `count`, return as many draws as the population has elements.
With `weights`, use non-negative finite weights proportional to the desired
probabilities. `replace` is as for [`randsample`](@ref).

The result is a vector on the generator's device. The input generator never
changes.
"""
@inline function randsample_next(
    rng::AbstractPureRNG,
    population;
    replace::Bool = true,
    threaded::Bool = false,
)
    return _randsample_next_unweighted(rng, population, nothing, replace, threaded)
end

@inline function randsample_next(
    rng::AbstractPureRNG,
    population,
    count::Integer;
    replace::Bool = true,
    threaded::Bool = false,
)
    return _randsample_next_unweighted(rng, population, count, replace, threaded)
end

"""
    randsample!(rng, population[, weights], destination; replace=true, threaded=false) -> destination

Sample into `destination`, with replacement unless `replace=false` (see [`randsample`](@ref)). Its length determines the number of
draws, and it must have exactly the prepared population element type. Return the
identical destination. Fills run on the calling task by default; `threaded=true`
splits a CPU fill across threads without changing the values.

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
    replace::Bool = true,
    threaded::Bool = false,
)
    return first(
        _randsample_next_unweighted!(rng, population, destination, replace, threaded),
    )
end

"""
    randsample_next!(rng, population[, weights], destination; replace=true, threaded=false) -> (destination, next_rng)

Sample into `destination` and return the advanced immutable
generator. The fill and validation rules are the same as for [`randsample!`](@ref).
The input generator is not changed.
"""
@inline function randsample_next!(
    rng::AbstractPureRNG,
    population,
    destination::AbstractArray;
    replace::Bool = true,
    threaded::Bool = false,
)
    return _randsample_next_unweighted!(rng, population, destination, replace, threaded)
end
