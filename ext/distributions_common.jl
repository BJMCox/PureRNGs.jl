const _FloatType = Union{Float32,Float64}
# The parameter ties a distribution to its destination element type, so a
# mismatched destination finds no method rather than a validation error.
const _FloatMapped{T} = Union{
    Distributions.Normal{T},
    Distributions.Uniform{T},
    Distributions.Exponential{T},
    Distributions.LogNormal{T},
    Distributions.Weibull{T},
    Distributions.Rayleigh{T},
    Distributions.Laplace{T},
    Distributions.Logistic{T},
    Distributions.Gumbel{T},
    Distributions.Pareto{T},
    Distributions.Frechet{T},
    Distributions.Cauchy{T},
    Distributions.TriangularDist{T},
}
const _MappedDistribution{T} = Union{_FloatMapped{T},Distributions.Bernoulli{T}}
const _FixedDistribution =
    Union{_MappedDistribution{<:_FloatType},Distributions.DiscreteUniform}

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
@inline _result_type(::Distributions.Categorical) = Int

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
    # UInt64 wrap maps cardinality 2^64 to zero, the range reduction's full-width case.
    return (d.b % UInt64 - d.a % UInt64) + UInt64(1)
end

@inline _distribution_span(d::Distributions.DiscreteUniform) =
    IR._range_bits(_discrete_span(d))

# Every mapping takes the core transform ops first, so a backend that must
# avoid `fma` supplies its own ops object instead of a second protocol.
@inline _map_distribution(ops, d::Distributions.Normal, z) =
    IR._transform_muladd(ops, d.σ, z, d.μ)
@inline function _map_distribution(ops, d::Distributions.Uniform{T}, u) where {T}
    width = d.b - d.a
    scaled = IR._transform_product(ops, width, u, T(0.5))
    return d.a + scaled
end
@inline _map_distribution(ops, d::Distributions.Exponential, x) = d.θ * x
@inline _map_distribution(ops, d::Distributions.LogNormal, z) =
    exp(IR._transform_muladd(ops, d.σ, z, d.μ))
@inline _map_distribution(ops, d::Distributions.Weibull, x) = d.θ * x^inv(d.α)
@inline _map_distribution(ops, d::Distributions.Rayleigh{T}, x) where {T} =
    d.σ * sqrt(T(2) * x)
@inline _map_distribution(ops, d::Distributions.Laplace, x, positive) =
    IR._transform_muladd(ops, ifelse(positive, d.θ, -d.θ), x, d.μ)
@inline _map_distribution(ops, d::Distributions.Logistic, u) =
    IR._transform_muladd(ops, d.θ, log(u) - log1p(-u), d.μ)
@inline _map_distribution(ops, d::Distributions.Gumbel, e) =
    IR._transform_muladd(ops, -d.θ, log(e), d.μ)
@inline _map_distribution(ops, d::Distributions.Pareto, x) = d.θ * exp(x / d.α)
@inline _map_distribution(ops, d::Distributions.Frechet, e) = d.θ * e^(-inv(d.α))
@inline _map_distribution(ops, d::Distributions.Cauchy{T}, u) where {T} =
    IR._transform_muladd(ops, d.σ, tanpi(u - T(0.5)), d.μ)
@inline function _map_distribution(
    ::IR._NativeTransformOps,
    d::Distributions.TriangularDist{T},
    v,
) where {T}
    d.a == d.b && return d.a
    p = (d.c - d.a) / (d.b - d.a)
    iszero(v) && return d.a
    if v <= p
        return fma(d.c - d.a, sqrt(v / p), d.a)
    end
    return fma(d.c - d.b, sqrt((one(T) - v) / (one(T) - p)), d.b)
end
# A traced value cannot steer a branch, so the general form evaluates both arms
# and selects. The arm it discards can be non-finite.
@inline function _map_distribution(ops, d::Distributions.TriangularDist{T}, v) where {T}
    d.a == d.b && return d.a
    p = (d.c - d.a) / (d.b - d.a)
    lower = IR._transform_muladd(ops, d.c - d.a, sqrt(v / p), d.a)
    upper = IR._transform_muladd(ops, d.c - d.b, sqrt((one(T) - v) / (one(T) - p)), d.b)
    return ifelse(iszero(v), d.a, ifelse(v <= p, lower, upper))
end
@inline _map_distribution(ops, d::Distributions.Bernoulli, u) = u < d.p
