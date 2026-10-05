# Distribution fills put their source before the destination, unlike Random's
# type and population fills. Keep that distinction out of the public adapter.
@inline IR._randset_fill!!(
    rng::Union{IR.AbstractPureRNG,IR._ReactantRNG},
    source::Distributions.Sampleable,
    destination;
    kwargs...,
) = IR.rand_next!(rng, source, destination; kwargs...)
@inline IR._randset_fill!!(
    rng::Random.AbstractRNG,
    source::Distributions.Sampleable,
    destination;
    kwargs...,
) = (Random.rand!(rng, source, destination; kwargs...), rng)

# Replace inner samples instead of mutating borrowed or immutable values.
# The native allocation flag preserves the outer buffer without an alias scan.
@inline IR._randset_fill!!(
    rng::Random.AbstractRNG,
    source::Distributions.Sampleable,
    destination::AbstractArray{<:AbstractArray};
    kwargs...,
) = (Random.rand!(rng, source, destination, true; kwargs...), rng)

# Inspect parameter storage, never materialize a covariance or copy a source.
@inline IR._randset_aliases(
    destination,
    source::Union{
        Distributions.Categorical,
        Distributions.DiscreteNonParametric,
        Distributions.Dirichlet,
        Distributions.MvNormal,
    },
) = any(
    parameter -> IR._randset_aliases(destination, parameter),
    Distributions.params(source),
)
@inline IR._randset_aliases(destination, source::Distributions.PDMats.PDMat) =
    Base.mightalias(destination, source.mat) ||
    Base.mightalias(destination, source.chol.factors)
@inline IR._randset_aliases(destination, source::Distributions.PDMats.PDiagMat) =
    Base.mightalias(destination, source.diag)
