const _FloatType = Union{Float32,Float64}
const _MappedDistribution = Union{
    Distributions.Normal{Float32},
    Distributions.Normal{Float64},
    Distributions.Uniform{Float32},
    Distributions.Uniform{Float64},
    Distributions.Exponential{Float32},
    Distributions.Exponential{Float64},
    Distributions.LogNormal{Float32},
    Distributions.LogNormal{Float64},
    Distributions.Weibull{Float32},
    Distributions.Weibull{Float64},
    Distributions.Rayleigh{Float32},
    Distributions.Rayleigh{Float64},
    Distributions.Laplace{Float32},
    Distributions.Laplace{Float64},
    Distributions.Logistic{Float32},
    Distributions.Logistic{Float64},
    Distributions.Gumbel{Float32},
    Distributions.Gumbel{Float64},
    Distributions.Pareto{Float32},
    Distributions.Pareto{Float64},
    Distributions.Frechet{Float32},
    Distributions.Frechet{Float64},
    Distributions.Cauchy{Float32},
    Distributions.Cauchy{Float64},
    Distributions.TriangularDist{Float32},
    Distributions.TriangularDist{Float64},
    Distributions.Bernoulli{Float32},
    Distributions.Bernoulli{Float64},
}
const _FixedDistribution = Union{_MappedDistribution,Distributions.DiscreteUniform}

@inline _result_type(::Distributions.Normal{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Uniform{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Exponential{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.LogNormal{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Weibull{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Rayleigh{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Laplace{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Logistic{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Gumbel{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Pareto{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Frechet{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Cauchy{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.TriangularDist{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Bernoulli{T}) where {T<:_FloatType} = Bool
@inline _result_type(::Distributions.DiscreteUniform) = Int

@noinline _invalid_parameters(::Distributions.Normal) =
    throw(ArgumentError("invalid Normal parameters"))
@noinline _invalid_parameters(::Distributions.Uniform) =
    throw(ArgumentError("invalid Uniform parameters"))
@noinline _invalid_parameters(::Distributions.Exponential) =
    throw(ArgumentError("invalid Exponential parameters"))
@noinline _invalid_parameters(::Distributions.LogNormal) =
    throw(ArgumentError("invalid LogNormal parameters"))
@noinline _invalid_parameters(::Distributions.Weibull) =
    throw(ArgumentError("invalid Weibull parameters"))
@noinline _invalid_parameters(::Distributions.Rayleigh) =
    throw(ArgumentError("invalid Rayleigh parameters"))
@noinline _invalid_parameters(::Distributions.Laplace) =
    throw(ArgumentError("invalid Laplace parameters"))
@noinline _invalid_parameters(::Distributions.Logistic) =
    throw(ArgumentError("invalid Logistic parameters"))
@noinline _invalid_parameters(::Distributions.Gumbel) =
    throw(ArgumentError("invalid Gumbel parameters"))
@noinline _invalid_parameters(::Distributions.Pareto) =
    throw(ArgumentError("invalid Pareto parameters"))
@noinline _invalid_parameters(::Distributions.Frechet) =
    throw(ArgumentError("invalid Frechet parameters"))
@noinline _invalid_parameters(::Distributions.Cauchy) =
    throw(ArgumentError("invalid Cauchy parameters"))
@noinline _invalid_parameters(::Distributions.TriangularDist) =
    throw(ArgumentError("invalid TriangularDist parameters"))
@noinline _invalid_parameters(::Distributions.Bernoulli) =
    throw(ArgumentError("invalid Bernoulli parameters"))
@noinline _invalid_parameters(::Distributions.DiscreteUniform) =
    throw(ArgumentError("invalid DiscreteUniform parameters"))

@inline function _validate_distribution(d::Distributions.Normal{T}) where {T<:_FloatType}
    isfinite(d.μ) || _invalid_parameters(d)
    isfinite(d.σ) || _invalid_parameters(d)
    d.σ >= zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(d::Distributions.Uniform{T}) where {T<:_FloatType}
    isfinite(d.a) || _invalid_parameters(d)
    isfinite(d.b) || _invalid_parameters(d)
    d.a < d.b || _invalid_parameters(d)
    isfinite(d.b - d.a) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(
    d::Distributions.Exponential{T},
) where {T<:_FloatType}
    isfinite(d.θ) || _invalid_parameters(d)
    d.θ > zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(d::Distributions.LogNormal{T}) where {T<:_FloatType}
    isfinite(d.μ) || _invalid_parameters(d)
    isfinite(d.σ) || _invalid_parameters(d)
    d.σ >= zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(d::Distributions.Weibull{T}) where {T<:_FloatType}
    isfinite(d.α) || _invalid_parameters(d)
    isfinite(d.θ) || _invalid_parameters(d)
    d.α > zero(T) || _invalid_parameters(d)
    d.θ > zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(d::Distributions.Rayleigh{T}) where {T<:_FloatType}
    isfinite(d.σ) || _invalid_parameters(d)
    d.σ > zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(d::Distributions.Laplace{T}) where {T<:_FloatType}
    isfinite(d.μ) || _invalid_parameters(d)
    isfinite(d.θ) || _invalid_parameters(d)
    d.θ > zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(
    d::Union{Distributions.Logistic{T},Distributions.Gumbel{T}},
) where {T<:_FloatType}
    isfinite(d.μ) || _invalid_parameters(d)
    isfinite(d.θ) || _invalid_parameters(d)
    d.θ > zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(
    d::Union{Distributions.Pareto{T},Distributions.Frechet{T}},
) where {T<:_FloatType}
    isfinite(d.α) || _invalid_parameters(d)
    isfinite(d.θ) || _invalid_parameters(d)
    d.α > zero(T) || _invalid_parameters(d)
    d.θ > zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(d::Distributions.Cauchy{T}) where {T<:_FloatType}
    isfinite(d.μ) || _invalid_parameters(d)
    isfinite(d.σ) || _invalid_parameters(d)
    d.σ > zero(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(
    d::Distributions.TriangularDist{T},
) where {T<:_FloatType}
    isfinite(d.a) || _invalid_parameters(d)
    isfinite(d.b) || _invalid_parameters(d)
    isfinite(d.c) || _invalid_parameters(d)
    d.a <= d.c <= d.b || _invalid_parameters(d)
    isfinite(d.b - d.a) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(d::Distributions.Bernoulli{T}) where {T<:_FloatType}
    isfinite(d.p) || _invalid_parameters(d)
    d.p >= zero(T) || _invalid_parameters(d)
    d.p <= one(T) || _invalid_parameters(d)
    return nothing
end

@inline function _validate_distribution(d::Distributions.DiscreteUniform)
    d.a <= d.b || _invalid_parameters(d)
    return nothing
end

@inline _distribution_span(::Distributions.Normal{T}) where {T<:_FloatType} =
    IR._normal_bits(T)
@inline _distribution_span(::Distributions.Uniform{T}) where {T<:_FloatType} =
    IR._draw_bits(T)
@inline _distribution_span(::Distributions.Exponential{T}) where {T<:_FloatType} =
    IR._exponential_bits(T)
@inline _distribution_span(::Distributions.LogNormal{T}) where {T<:_FloatType} =
    IR._normal_bits(T)
@inline _distribution_span(::Distributions.Weibull{T}) where {T<:_FloatType} =
    IR._exponential_bits(T)
@inline _distribution_span(::Distributions.Rayleigh{T}) where {T<:_FloatType} =
    IR._exponential_bits(T)
@inline _distribution_span(::Distributions.Laplace{T}) where {T<:_FloatType} =
    IR._exponential_bits(T) + UInt16(1)
@inline _distribution_span(
    ::Union{
        Distributions.Logistic{T},
        Distributions.Gumbel{T},
        Distributions.Frechet{T},
        Distributions.Cauchy{T},
    },
) where {T<:_FloatType} = IR._normal_bits(T)
@inline _distribution_span(::Distributions.Pareto{T}) where {T<:_FloatType} =
    IR._exponential_bits(T)
@inline _distribution_span(::Distributions.TriangularDist{T}) where {T<:_FloatType} =
    IR._draw_bits(T)
@inline _distribution_span(::Distributions.Bernoulli{T}) where {T<:_FloatType} =
    IR._draw_bits(T)

@inline function _discrete_span(d::Distributions.DiscreteUniform)
    # UInt64 wrap maps cardinality 2^64 to the R55 zero sentinel.
    return (d.b % UInt64 - d.a % UInt64) + UInt64(1)
end

@inline _distribution_span(d::Distributions.DiscreteUniform) =
    IR._range_bits(_discrete_span(d))

# The ops-first `_map_distribution` methods take the ops object first and a
# distribution second. Leaving the first argument untyped makes every one of
# them ambiguous with `_map_distribution(d::Laplace, x, positive)`, which has
# the same arity. The shared supertype separates the two argument orders.
abstract type _DistributionOps end

struct _NativeDistributionOps <: _DistributionOps end

function _distribution_muladd end
function _distribution_product end

@inline _distribution_muladd(::_NativeDistributionOps, a, b, c) = fma(a, b, c)
@inline _distribution_product(::_NativeDistributionOps, a, b, half) = a * b
@inline _map_distribution(ops::_DistributionOps, d::Distributions.Normal, z) =
    _distribution_muladd(ops, d.σ, z, d.μ)
@inline _map_distribution(d::Distributions.Normal, z) =
    _map_distribution(_NativeDistributionOps(), d, z)
@inline function _map_distribution(
    ops::_DistributionOps,
    d::Distributions.Uniform{T},
    u,
) where {T}
    width = d.b - d.a
    scaled = _distribution_product(ops, width, u, T(0.5))
    return d.a + scaled
end
@inline _map_distribution(d::Distributions.Uniform, u) =
    _map_distribution(_NativeDistributionOps(), d, u)
@inline _map_distribution(d::Distributions.Exponential, x) = d.θ * x
@inline function _map_distribution(ops::_DistributionOps, d::Distributions.LogNormal, z)
    return exp(_distribution_muladd(ops, d.σ, z, d.μ))
end
@inline _map_distribution(d::Distributions.LogNormal, z) =
    _map_distribution(_NativeDistributionOps(), d, z)
@inline _map_distribution(d::Distributions.Weibull, x) = d.θ * x^inv(d.α)
@inline _map_distribution(d::Distributions.Rayleigh{T}, x) where {T} = d.σ * sqrt(T(2) * x)
@inline function _map_distribution(
    ops::_DistributionOps,
    d::Distributions.Laplace,
    x,
    positive,
)
    return _distribution_muladd(ops, ifelse(positive, d.θ, -d.θ), x, d.μ)
end
@inline _map_distribution(d::Distributions.Laplace, x, positive) =
    _map_distribution(_NativeDistributionOps(), d, x, positive)
@inline function _map_distribution(ops::_DistributionOps, d::Distributions.Logistic, u)
    return _distribution_muladd(ops, d.θ, log(u) - log1p(-u), d.μ)
end
@inline _map_distribution(d::Distributions.Logistic, u) =
    _map_distribution(_NativeDistributionOps(), d, u)
@inline function _map_distribution(ops::_DistributionOps, d::Distributions.Gumbel, e)
    return _distribution_muladd(ops, -d.θ, log(e), d.μ)
end
@inline _map_distribution(d::Distributions.Gumbel, e) =
    _map_distribution(_NativeDistributionOps(), d, e)
@inline _map_distribution(d::Distributions.Pareto, x) = d.θ * exp(x / d.α)
@inline _map_distribution(d::Distributions.Frechet, e) = d.θ * e^(-inv(d.α))
@inline function _map_distribution(
    ops::_DistributionOps,
    d::Distributions.Cauchy{T},
    u,
) where {T}
    return _distribution_muladd(ops, d.σ, tanpi(u - T(0.5)), d.μ)
end
@inline _map_distribution(d::Distributions.Cauchy, u) =
    _map_distribution(_NativeDistributionOps(), d, u)
@inline function _map_distribution(d::Distributions.TriangularDist{T}, v) where {T}
    d.a == d.b && return d.a
    p = (d.c - d.a) / (d.b - d.a)
    iszero(v) && return d.a
    if v <= p
        return fma(d.c - d.a, sqrt(v / p), d.a)
    end
    return fma(d.c - d.b, sqrt((one(T) - v) / (one(T) - p)), d.b)
end
@inline function _map_distribution(
    ops::_DistributionOps,
    d::Distributions.TriangularDist{T},
    v,
) where {T}
    d.a == d.b && return d.a
    p = (d.c - d.a) / (d.b - d.a)
    lower = _distribution_muladd(ops, d.c - d.a, sqrt(v / p), d.a)
    upper = _distribution_muladd(ops, d.c - d.b, sqrt((one(T) - v) / (one(T) - p)), d.b)
    return ifelse(iszero(v), d.a, ifelse(v <= p, lower, upper))
end
@inline _map_distribution(d::Distributions.Bernoulli, u) = u < d.p
