module PureRNGsReactantDistributionsExt

import Distributions
import PureRNGs
import Random
import Reactant

const IR = PureRNGs
const _ReactantRNG = IR._ReactantRNG

include("distributions_common.jl")
include("distributions_gamma.jl")

# The Reactant extension declares the same ops for the core generators. Julia
# extensions cannot share a name, so the singleton is declared again here.
struct _ReactantTransformOps end

@inline IR._transform_muladd(::_ReactantTransformOps, a, b, c) = muladd(a, b, c)
@inline IR._transform_product(::_ReactantTransformOps, a, b, half) =
    IR._rounded_product(a, b, b - half)

@inline _primitive(rng, ::Distributions.Normal{T}) where {T<:_FloatType} =
    Random.randn(rng, T)
@inline _primitive(rng, ::Distributions.Uniform{T}) where {T<:_FloatType} =
    Random.rand(rng, T)
@inline _primitive(rng, ::Distributions.Exponential{T}) where {T<:_FloatType} =
    Random.randexp(rng, T)
@inline _primitive(rng, ::Distributions.LogNormal{T}) where {T<:_FloatType} =
    Random.randn(rng, T)
@inline _primitive(
    rng,
    ::Union{
        Distributions.Logistic{T},
        Distributions.Gumbel{T},
        Distributions.Frechet{T},
        Distributions.Cauchy{T},
    },
) where {T<:_FloatType} = IR._midpoint_value(rng, T)
@inline _primitive(rng, ::Distributions.Weibull{T}) where {T<:_FloatType} =
    Random.randexp(rng, T)
@inline _primitive(rng, ::Distributions.Rayleigh{T}) where {T<:_FloatType} =
    Random.randexp(rng, T)
@inline _primitive(rng, ::Distributions.Pareto{T}) where {T<:_FloatType} =
    Random.randexp(rng, T)
@inline _primitive(rng, ::Distributions.TriangularDist{T}) where {T<:_FloatType} =
    Random.rand(rng, T)
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
@inline function _primitive_next(
    rng,
    ::Union{
        Distributions.Logistic{T},
        Distributions.Gumbel{T},
        Distributions.Frechet{T},
        Distributions.Cauchy{T},
    },
) where {T<:_FloatType}
    return IR._midpoint_value(rng, T), IR._addressed_rng(rng, IR._normal_bits(T), 2)
end
@inline _primitive_next(rng, ::Distributions.Weibull{T}) where {T<:_FloatType} =
    IR.randexp_next(rng, T)
@inline _primitive_next(rng, ::Distributions.Rayleigh{T}) where {T<:_FloatType} =
    IR.randexp_next(rng, T)
@inline _primitive_next(rng, ::Distributions.Pareto{T}) where {T<:_FloatType} =
    IR.randexp_next(rng, T)
@inline _primitive_next(rng, ::Distributions.TriangularDist{T}) where {T<:_FloatType} =
    IR.rand_next(rng, T)
@inline function _primitive_next(rng, ::Distributions.Laplace{T}) where {T<:_FloatType}
    magnitude, after_magnitude = IR.randexp_next(rng, T)
    positive, next_rng = IR.rand_next(after_magnitude, Bool)
    return (magnitude, positive), next_rng
end
@inline _primitive_next(rng, ::Distributions.Bernoulli{T}) where {T<:_FloatType} =
    IR.rand_next(rng, T)
@inline _primitive_next(rng, d::Distributions.DiscreteUniform) = IR.rand_next(rng, d.a:d.b)

@inline _primitive_at(rng, ::Distributions.Normal{T}, index) where {T<:_FloatType} =
    IR.randn_at(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Uniform{T}, index) where {T<:_FloatType} =
    IR.rand_at(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Exponential{T}, index) where {T<:_FloatType} =
    IR.randexp_at(rng, T, index)
@inline _primitive_at(rng, ::Distributions.LogNormal{T}, index) where {T<:_FloatType} =
    IR.randn_at(rng, T, index)
@inline function _primitive_at(
    rng,
    d::Union{
        Distributions.Logistic{T},
        Distributions.Gumbel{T},
        Distributions.Frechet{T},
        Distributions.Cauchy{T},
    },
    index,
) where {T<:_FloatType}
    return _primitive(IR._addressed_rng(rng, _distribution_span(d), index), d)
end
@inline _primitive_at(rng, ::Distributions.Weibull{T}, index) where {T<:_FloatType} =
    IR.randexp_at(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Rayleigh{T}, index) where {T<:_FloatType} =
    IR.randexp_at(rng, T, index)
@inline _primitive_at(rng, ::Distributions.Pareto{T}, index) where {T<:_FloatType} =
    IR.randexp_at(rng, T, index)
@inline _primitive_at(rng, ::Distributions.TriangularDist{T}, index) where {T<:_FloatType} =
    IR.rand_at(rng, T, index)
@inline function _primitive_at(
    rng,
    d::Distributions.Laplace{T},
    index,
) where {T<:_FloatType}
    return _primitive(IR._addressed_rng(rng, _distribution_span(d), index), d)
end
@inline _primitive_at(rng, ::Distributions.Bernoulli{T}, index) where {T<:_FloatType} =
    IR.rand_at(rng, T, index)
@inline function _primitive_at(rng, d::Distributions.DiscreteUniform, index)
    range = d.a:d.b
    addressed = IR._addressed_rng(rng, _distribution_span(d), index)
    return IR._range_value(addressed, range)
end

@inline _map_primitive(d::Distributions.Normal, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.Uniform, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.Exponential, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.LogNormal, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.Logistic, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.Gumbel{T}, value) where {T} =
    _map_distribution(_ReactantTransformOps(), d, -Base.log(one(T) - value))
@inline _map_primitive(d::Distributions.Pareto, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.Frechet{T}, value) where {T} =
    _map_distribution(_ReactantTransformOps(), d, -Base.log(one(T) - value))
@inline _map_primitive(d::Distributions.Cauchy, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.TriangularDist, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.Weibull, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.Rayleigh, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
@inline _map_primitive(d::Distributions.Laplace, payload) =
    _map_distribution(_ReactantTransformOps(), d, payload...)
@inline _map_primitive(d::Distributions.Bernoulli, value) =
    _map_distribution(_ReactantTransformOps(), d, value)
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

@inline function IR.rand_at(rng::_ReactantRNG, d::_FixedDistribution, index::Integer)
    _validate_distribution(d)
    return _map_primitive(d, _primitive_at(rng, d, index))
end


# The Gamma family maps traced standard Gamma draws in the spans the eager draw
# reads, so compiled and eager draws consume the same bits.
@inline _traced_gamma(rng, shape) =
    IR._traced_gamma(rng, shape, Val(false), IR._GAMMA_CANDIDATES)
@inline _traced_log_gamma(rng, shape) =
    IR._traced_gamma(rng, shape, Val(true), IR._GAMMA_CANDIDATES)

@inline _gamma_family_value(rng, d::Distributions.Gamma) = d.θ * _traced_gamma(rng, d.α)
@inline _gamma_family_value(rng, d::Distributions.Chisq{T}) where {T} =
    T(2) * _traced_gamma(rng, d.ν / 2)
@inline _gamma_family_value(rng, d::Distributions.InverseGamma) =
    d.θ / _traced_gamma(rng, d.invd.α)
@inline function _gamma_family_value(rng, d::Distributions.Beta{T}) where {T}
    width = IR._gamma_span(T, IR._GAMMA_CANDIDATES)
    log_x = _traced_log_gamma(rng, d.α)
    log_y = _traced_log_gamma(IR._addressed_rng(rng, width, 2), d.β)
    return inv(one(T) + exp(log_y - log_x))
end
@inline function _gamma_family_value(rng, d::Distributions.TDist{T}) where {T}
    z = Random.randn(rng, T)
    g = _traced_gamma(IR._addressed_rng(rng, IR._normal_bits(T), 2), d.ν / 2)
    return z * sqrt(d.ν / (2 * g))
end

@inline function Random.rand(rng::_ReactantRNG, d::_GammaFamilyDistribution)
    _validate_distribution(d)
    return _gamma_family_value(rng, d)
end

@inline function IR.rand_next(rng::_ReactantRNG, d::_GammaFamilyDistribution)
    _validate_distribution(d)
    next_rng = IR._addressed_rng(rng, _distribution_span(d), 2)
    return _gamma_family_value(rng, d), next_rng
end

@inline function IR.rand_at(rng::_ReactantRNG, d::_GammaFamilyDistribution, index::Integer)
    _validate_distribution(d)
    return _gamma_family_value(IR._addressed_rng(rng, _distribution_span(d), index), d)
end

end
