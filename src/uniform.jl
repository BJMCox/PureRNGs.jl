# Uniform draws run the transformed scaffold under `Val(:uniform)`, whose
# `_fill_width` and `_cooperative_value` methods live in `fill_hooks.jl`.
@inline function Random.rand!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_UniformResult}
    result, _ = _rand_transformed_next_fill!(rng, destination, threaded, Val(:uniform))
    return result
end

@inline function rand_next!(
    rng::_ScalarUniformGenerators,
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_UniformResult}
    return _rand_transformed_next_fill!(rng, destination, threaded, Val(:uniform))
end

@doc """
    rand_next!(rng, destination; threaded=false) -> (destination, next_rng)
    rand_next!(rng, destination, range; threaded=false) -> (destination, next_rng)

Fill `destination` from `rng` and return the advanced immutable generator with
the same destination. The destination element type must be a result type
[`rand_next`](@ref) supports, or with `range` the integer element type of that
range, and its device must match the generator.

Fills run serially by default. Set `threaded=true` to split a CPU fill across
threads; the keyword never changes the generated stream. The input generator
never changes.
""" rand_next!

@inline function rand_next(
    rng::_ScalarUniformGenerators,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return _rand_transformed_next_array(
        rng,
        Float64,
        (dim1, dims...),
        Val(:uniform),
        threaded,
    )
end
@inline rand_next(rng::_ScalarUniformGenerators, dims::Dims; threaded::Bool = false) =
    _rand_transformed_next_array(rng, Float64, dims, Val(:uniform), threaded)

@inline function Random.rand(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) where {T<:_UniformResult}
    destination, _ =
        _rand_transformed_next_array(rng, T, (dim1, dims...), Val(:uniform), threaded)
    return destination
end
@inline Random.rand(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:_UniformResult} =
    first(_rand_transformed_next_array(rng, T, dims, Val(:uniform), threaded))

@inline function rand_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) where {T<:_UniformResult}
    return _rand_transformed_next_array(rng, T, (dim1, dims...), Val(:uniform), threaded)
end
@inline rand_next(
    rng::_ScalarUniformGenerators,
    ::Type{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:_UniformResult} =
    _rand_transformed_next_array(rng, T, dims, Val(:uniform), threaded)
