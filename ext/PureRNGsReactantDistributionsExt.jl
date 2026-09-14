module PureRNGsReactantDistributionsExt

import Distributions
import PureRNGs
import Random
import Reactant

const IR = PureRNGs
const _ReactantRNG = IR._ReactantRNG

include("distributions_common.jl")

struct _ReactantDistributionOps end

@inline _distribution_muladd(::_ReactantDistributionOps, a, b, c) = muladd(a, b, c)
@inline _distribution_product(::_ReactantDistributionOps, a, b, half) =
    IR._rounded_product(a, b, b - half)

@inline _primitive(rng, ::Distributions.Normal{T}) where {T<:_FloatType} =
    Random.randn(rng, T)
@inline _primitive(rng, ::Distributions.Uniform{T}) where {T<:_FloatType} =
    Random.rand(rng, T)
@inline _primitive(rng, ::Distributions.Exponential{T}) where {T<:_FloatType} =
    Random.randexp(rng, T)
@inline _primitive(rng, ::Distributions.LogNormal{T}) where {T<:_FloatType} =
    Random.randn(rng, T)
@inline _primitive(rng, ::Distributions.Weibull{T}) where {T<:_FloatType} =
    Random.randexp(rng, T)
@inline _primitive(rng, ::Distributions.Rayleigh{T}) where {T<:_FloatType} =
    Random.randexp(rng, T)
@inline function _primitive(rng, ::Distributions.Laplace{T}) where {T<:_FloatType}
    magnitude, after_magnitude = IR.randexp_next(rng, T)
    return magnitude, Random.rand(after_magnitude, Bool)
end
@inline _primitive(rng, ::Distributions.Bernoulli{T}) where {T<:_FloatType} =
    Random.rand(rng, T)
@inline _primitive(rng, d::Distributions.DiscreteUniform) = Random.rand(rng, d.a:d.b)

@inline _primitive_next(rng, ::Distributions.Normal{T}) where {T<:_FloatType} =
    IR.randn_next(rng, T)
@inline _primitive_next(rng, ::Distributions.Uniform{T}) where {T<:_FloatType} =
    IR.rand_next(rng, T)
@inline _primitive_next(rng, ::Distributions.Exponential{T}) where {T<:_FloatType} =
    IR.randexp_next(rng, T)
@inline _primitive_next(rng, ::Distributions.LogNormal{T}) where {T<:_FloatType} =
    IR.randn_next(rng, T)
@inline _primitive_next(rng, ::Distributions.Weibull{T}) where {T<:_FloatType} =
    IR.randexp_next(rng, T)
@inline _primitive_next(rng, ::Distributions.Rayleigh{T}) where {T<:_FloatType} =
    IR.randexp_next(rng, T)
@inline function _primitive_next(rng, ::Distributions.Laplace{T}) where {T<:_FloatType}
    magnitude, after_magnitude = IR.randexp_next(rng, T)
    positive, next_rng = IR.rand_next(after_magnitude, Bool)
    return (magnitude, positive), next_rng
end
@inline _primitive_next(rng, ::Distributions.Bernoulli{T}) where {T<:_FloatType} =
    IR.rand_next(rng, T)
@inline _primitive_next(rng, d::Distributions.DiscreteUniform) = IR.rand_next(rng, d.a:d.b)

@inline _primitive_at(rng, ::Distributions.Normal{T}, index) where {T<:_FloatType} =
    IR.randnat(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Uniform{T}, index) where {T<:_FloatType} =
    IR.randat(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Exponential{T}, index) where {T<:_FloatType} =
    IR.randexpat(rng, T, index)
@inline _primitive_at(rng, ::Distributions.LogNormal{T}, index) where {T<:_FloatType} =
    IR.randnat(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Weibull{T}, index) where {T<:_FloatType} =
    IR.randexpat(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Rayleigh{T}, index) where {T<:_FloatType} =
    IR.randexpat(rng, T, index)
@inline function _primitive_at(
    rng,
    d::Distributions.Laplace{T},
    index,
) where {T<:_FloatType}
    return _primitive(IR._addressed_rng(rng, _distribution_span(d), index), d)
end
@inline _primitive_at(rng, ::Distributions.Bernoulli{T}, index) where {T<:_FloatType} =
    IR.randat(rng, T, index)
@inline function _primitive_at(rng, d::Distributions.DiscreteUniform, index)
    range = d.a:d.b
    addressed = IR._addressed_rng(rng, _distribution_span(d), index)
    return IR._range_value(addressed, range)
end

@inline _map_primitive(d::Distributions.Normal, value) =
    _map_distribution(_ReactantDistributionOps(), d, value)
@inline _map_primitive(d::Distributions.Uniform, value) =
    _map_distribution(_ReactantDistributionOps(), d, value)
@inline _map_primitive(d::Distributions.Exponential, value) = _map_distribution(d, value)
@inline _map_primitive(d::Distributions.LogNormal, value) =
    _map_distribution(_ReactantDistributionOps(), d, value)
@inline _map_primitive(d::Distributions.Weibull, value) = _map_distribution(d, value)
@inline _map_primitive(d::Distributions.Rayleigh, value) = _map_distribution(d, value)
@inline _map_primitive(d::Distributions.Laplace, payload) =
    _map_distribution(_ReactantDistributionOps(), d, payload...)
@inline _map_primitive(d::Distributions.Bernoulli, value) = _map_distribution(d, value)
@inline _map_primitive(::Distributions.DiscreteUniform, value) = value

@inline function Random.rand(rng::_ReactantRNG, d::_FixedDistribution)
    _validate_distribution(d)
    return _map_primitive(d, _primitive(rng, d))
end

@inline function IR.rand_next(rng::_ReactantRNG, d::_FixedDistribution)
    _validate_distribution(d)
    value, next_rng = _primitive_next(rng, d)
    return _map_primitive(d, value), next_rng
end

@inline function IR.randat(rng::_ReactantRNG, d::_FixedDistribution, index::Integer)
    _validate_distribution(d)
    return _map_primitive(d, _primitive_at(rng, d, index))
end

end
