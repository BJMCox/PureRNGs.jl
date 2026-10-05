"""
    randgen!!(rng, from_what, dims...) -> (value, next_rng)
    randgen!!(rng, from_what, dims::Dims) -> (values, next_rng)

Draw from a type, population, or supported distribution with either a pure or
mutable RNG. Arguments, keywords, sample shapes, and values follow the native
`rand_next` or `Random.rand` method. Nonempty tuple dimensions are splatted, so
`(n,)` and `n` select the same sample layout. The source is not modified. Population
elements may be borrowed, not copied.

Always use both returned values. A mutable RNG advances in place; a pure RNG
returns its continuation without changing the input. The `!!` contract permits
mutation but does not invalidate immutable snapshots. It does not add fixed-work,
device, or failure-atomicity guarantees to the native method.
"""
function randgen!! end

# Reactant's traced carrier is not an AbstractPureRNG, but has the same API.
for (generate, draw, next) in (
    (:randgen!!, :rand, :rand_next),
    (:randngen!!, :randn, :randn_next),
    (:randexpgen!!, :randexp, :randexp_next),
)
    @eval begin
        @inline $generate(rng::Union{AbstractPureRNG,_ReactantRNG}, args...; kwargs...) =
            $next(rng, args...; kwargs...)
        @inline $generate(rng::Random.AbstractRNG, args...; kwargs...) =
            (Random.$draw(rng, args...; kwargs...), rng)
    end
end

# In Distributions, the native `(n,)` and `n` forms can use different layouts.
# The adapter gives both dimension spellings the vararg layout.
for R in (Union{AbstractPureRNG,_ReactantRNG}, Random.AbstractRNG)
    @eval @inline randgen!!(rng::$R, source, dims::Tuple{Int,Vararg{Int}}; kwargs...) =
        randgen!!(rng, source, dims...; kwargs...)
end

"""
    randugen!!(rng[, T], dims...) -> (value, next_rng)
    randugen!!(rng[, T], dims::Dims) -> (values, next_rng)

Draw uniform values, with `Float64` as the default result type. A tuple here is
always a shape. Use [`randgen!!`](@ref) to sample a population or distribution.
"""
@inline randugen!!(rng, dims::Integer...; kwargs...) =
    randgen!!(rng, Float64, dims...; kwargs...)
@inline randugen!!(rng, dims::Dims; kwargs...) = randgen!!(rng, Float64, dims; kwargs...)
@inline randugen!!(rng, ::Type{T}, dims...; kwargs...) where {T} =
    randgen!!(rng, T, dims...; kwargs...)

@doc """
    randngen!!(rng[, T], dims...) -> (value, next_rng)
    randngen!!(rng[, T], dims::Dims) -> (values, next_rng)

Draw standard normal values through `randn_next` or `Random.randn`.
The result type defaults to `Float64`. See [`randgen!!`](@ref) for ownership.
""" randngen!!

@doc """
    randexpgen!!(rng[, T], dims...) -> (value, next_rng)
    randexpgen!!(rng[, T], dims::Dims) -> (values, next_rng)

Draw standard exponential values through `randexp_next` or `Random.randexp`.
The result type defaults to `Float64`. See [`randgen!!`](@ref) for ownership.
""" randexpgen!!

# Select replacement storage before any draws. Immutable array extensions only
# need to supply storage and reconstruct the value after the native fill.
@inline _randset_eltype(source, destination) = eltype(destination)
@inline _randset_eltype(::Type{T}, destination) where {T} = T
@inline _randset_storage(destination::AbstractArray, ::Type{T}) where {T} =
    eltype(destination) === T ? destination : similar(destination, T)
@inline _randset_storage(destination::AbstractRange, ::Type{T}) where {T} =
    similar(destination, T)
@inline _randset_finish(destination, values) = values

# Unknown sources can hide references to destination storage. Known sources
# specialize this constant-time check instead of walking an object graph.
@inline _randset_aliases(destination, source) = !isbitstype(typeof(source))
@inline _randset_aliases(destination, ::Type) = false
@inline _randset_aliases(destination, source::AbstractArray) =
    Base.mightalias(destination, source)

@inline _randset_fill!!(
    rng::Union{AbstractPureRNG,_ReactantRNG},
    source,
    destination;
    kwargs...,
) = rand_next!(rng, destination, source; kwargs...)
@inline _randset_fill!!(rng::Random.AbstractRNG, source, destination; kwargs...) =
    (Random.rand!(rng, destination, source; kwargs...), rng)
@inline _randset_fill!!(
    rng::Union{AbstractPureRNG,_ReactantRNG},
    ::Type,
    destination;
    kwargs...,
) = randuset!!(rng, destination; kwargs...)
@inline _randset_fill!!(rng::Random.AbstractRNG, ::Type, destination; kwargs...) =
    randuset!!(rng, destination; kwargs...)

"""
    randset!!(rng, from_what, old_value) -> (value, next_rng)

Replace a scalar with one draw, or reuse a writable array with a native bulk fill.
The source precedes the destination, including for distributions. A scalar type
source selects the result element type; a different destination element type causes
replacement. Array shapes otherwise follow the native fill, including a vector
for one multivariate draw or a matrix with one draw per column.

Ranges and immutable StaticArrays are replaced. Arrays of immutable samples use
their native fill methods. A static array type with a static template denotes
one whole sample. Arbitrary tuples and structs are single values, not
recursively filled containers. Extensions may define further replacement rules.

The source is not modified. A destination that aliases a supported source is
replaced before sampling. Custom source methods must preserve the same contract.
Population elements can still be borrowed. Use both returned values and do not
reuse the old RNG or destination in generic code. A failure can leave a mutable
RNG advanced or a destination partly written, as with the native fill. No failed
draw is retried. See [`randgen!!`](@ref).
"""
@inline randset!!(rng, source, old_value; kwargs...) = randgen!!(rng, source; kwargs...)

@inline function randset!!(rng, source, destination::AbstractArray; kwargs...)
    target = _randset_storage(destination, _randset_eltype(source, destination))
    if _randset_aliases(target, source)
        target = similar(target)
    end
    values, next_rng = _randset_fill!!(rng, source, target; kwargs...)
    return _randset_finish(destination, values), next_rng
end

for (set, generate, fill!, next!) in (
    (:randuset!!, :randugen!!, :rand!, :rand_next!),
    (:randnset!!, :randngen!!, :randn!, :randn_next!),
    (:randexpset!!, :randexpgen!!, :randexp!, :randexp_next!),
)
    @eval begin
        @inline $set(rng, old_value; kwargs...) =
            $generate(rng, typeof(old_value); kwargs...)
        @inline function $set(rng, destination::AbstractArray; kwargs...)
            target = _randset_storage(destination, eltype(destination))
            values, next_rng = _primitive_set!!($set, rng, target; kwargs...)
            return _randset_finish(destination, values), next_rng
        end
        @inline _primitive_set!!(
            ::typeof($set),
            rng::Union{AbstractPureRNG,_ReactantRNG},
            destination;
            kwargs...,
        ) = $next!(rng, destination; kwargs...)
        @inline _primitive_set!!(
            ::typeof($set),
            rng::Random.AbstractRNG,
            destination;
            kwargs...,
        ) = (Random.$fill!(rng, destination; kwargs...), rng)
    end
end

@doc """
    randuset!!(rng, old_value) -> (value, next_rng)

Replace a scalar or fill an array with uniform values of its existing element
type. See [`randset!!`](@ref) for replacement, ownership, and failure rules.
""" randuset!!

@doc """
    randnset!!(rng, old_value) -> (value, next_rng)

Replace a scalar or fill an array with standard normal values of its existing
element type. See [`randset!!`](@ref) for replacement, ownership, and failure rules.
""" randnset!!

@doc """
    randexpset!!(rng, old_value) -> (value, next_rng)

Replace a scalar or fill an array with standard exponential values of its existing
element type. See [`randset!!`](@ref) for replacement, ownership, and failure rules.
""" randexpset!!
