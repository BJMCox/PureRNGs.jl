# A pick is one unweighted sample: one range candidate reduced over the
# cardinality selects the element at that ordinal, so `rand_next(rng, pop)`
# consumes and returns what `randsample_next(rng, pop, 1)` does.
const _PickPopulation = Union{AbstractArray,Tuple,AbstractString,AbstractDict,AbstractSet}

@inline _population_value(population::Tuple, ordinal::UInt64) = population[Int(ordinal)]

# Strings, dicts, and sets have no ordinal index, so a pick walks to it: O(n).
@inline _population_value(population, ordinal::UInt64) =
    first(Iterators.drop(population, Int(ordinal) - 1))

@noinline _empty_collection_error() = throw(ArgumentError("collection must be non-empty"))

@inline function _pick_cardinality(population)
    cardinality = _sampling_cardinality(population)
    iszero(cardinality) && _empty_collection_error()
    return cardinality
end

@inline function _pick_unchecked(rng, position, population, cardinality::UInt64)
    offset = _range_offset(rng, position, cardinality)
    return _population_value(population, offset + one(UInt64))
end

@inline function _rand_next_pick(rng::_ScalarUniformGenerators, population)
    cardinality = _pick_cardinality(population)
    next_rng = _reserve_scalar(rng, _range_bits(cardinality))
    return _pick_unchecked(rng, rng.position, population, cardinality), next_rng
end

@inline function _rand_next_pick_array(
    rng::_ScalarUniformGenerators,
    population,
    dims::Tuple,
    threaded::Bool,
)
    agnostic = _check_population_device(rng, population)
    indexed = _prepare_population(rng.device, population, agnostic)
    destination = _allocate_draw_array(rng.device, eltype(indexed), dims)
    return _randsample_next_unweighted!(rng, indexed, destination, true, threaded)
end

@inline Random.rand(rng::_ScalarUniformGenerators, population::_PickPopulation) =
    first(_rand_next_pick(rng, population))
@inline rand_next(rng::_ScalarUniformGenerators, population::_PickPopulation) =
    _rand_next_pick(rng, population)

@inline function rand_at(
    rng::_ScalarUniformGenerators,
    population::_PickPopulation,
    i::Integer,
)
    cardinality = _pick_cardinality(population)
    addressed = _addressed_rng(rng, _range_bits(cardinality), i)
    return _pick_unchecked(addressed, addressed.position, population, cardinality)
end

@inline function Random.rand(
    rng::_ScalarUniformGenerators,
    population::_PickPopulation,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return first(_rand_next_pick_array(rng, population, (dim1, dims...), threaded))
end
@inline Random.rand(
    rng::_ScalarUniformGenerators,
    population::_PickPopulation,
    dims::Dims;
    threaded::Bool = false,
) = first(_rand_next_pick_array(rng, population, dims, threaded))

@inline function rand_next(
    rng::_ScalarUniformGenerators,
    population::_PickPopulation,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return _rand_next_pick_array(rng, population, (dim1, dims...), threaded)
end
@inline rand_next(
    rng::_ScalarUniformGenerators,
    population::_PickPopulation,
    dims::Dims;
    threaded::Bool = false,
) = _rand_next_pick_array(rng, population, dims, threaded)

@inline function Random.rand!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray,
    population::_PickPopulation;
    threaded::Bool = false,
)
    return first(_randsample_next_unweighted!(rng, population, destination, true, threaded))
end

@inline function rand_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray,
    population::_PickPopulation;
    threaded::Bool = false,
)
    return _randsample_next_unweighted!(rng, population, destination, true, threaded)
end

# Characters are uniform over the Unicode scalar values, as in `Random`: an
# offset over the 1,112,064 values steps over the 2,048 surrogates.
struct _CharCodec <: _MappedFillCodec end

const _CHAR_SPAN = UInt64(0x110000 - 0x800)

@inline function _unicode_scalar(offset::UInt64)
    code = offset % UInt32
    return Char(code < 0xd800 ? code : code + UInt32(0x800))
end

@inline _fill_width(::_CharCodec, ::Type{Char}) = _range_bits(_CHAR_SPAN)
@inline _cooperative_value(::_CharCodec, ::Type{Char}, raw) =
    _unicode_scalar(_reduce_range_candidate(raw % UInt64, _CHAR_SPAN))

@inline Random.rand(rng::_ScalarUniformGenerators, ::Type{Char}) =
    first(_draw_next(rng, _CharCodec(), Char))
@inline rand_next(rng::_ScalarUniformGenerators, ::Type{Char}) =
    _draw_next(rng, _CharCodec(), Char)
@inline rand_at(rng::_ScalarUniformGenerators, ::Type{Char}, i::Integer) =
    _draw_at(rng, _CharCodec(), Char, i)
@inline rand_at(
    rng::_ScalarUniformGenerators,
    ::Type{Char},
    indices::AbstractUnitRange{<:Integer};
    threaded::Bool = false,
) = _addressed_array(rng, Char, indices, _range_bits(_CHAR_SPAN), rand_next, threaded)

@inline function Random.rand(
    rng::_ScalarUniformGenerators,
    ::Type{Char},
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return first(
        _rand_transformed_next_array(rng, Char, (dim1, dims...), _CharCodec(), threaded),
    )
end
@inline Random.rand(
    rng::_ScalarUniformGenerators,
    ::Type{Char},
    dims::Dims;
    threaded::Bool = false,
) = first(_rand_transformed_next_array(rng, Char, dims, _CharCodec(), threaded))

@inline function rand_next(
    rng::_ScalarUniformGenerators,
    ::Type{Char},
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return _rand_transformed_next_array(rng, Char, (dim1, dims...), _CharCodec(), threaded)
end
@inline rand_next(
    rng::_ScalarUniformGenerators,
    ::Type{Char},
    dims::Dims;
    threaded::Bool = false,
) = _rand_transformed_next_array(rng, Char, dims, _CharCodec(), threaded)

@inline Random.rand!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{Char};
    threaded::Bool = false,
) = first(_rand_transformed_next_fill!(rng, destination, threaded, _CharCodec()))
@inline rand_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{Char};
    threaded::Bool = false,
) = _rand_transformed_next_fill!(rng, destination, threaded, _CharCodec())
