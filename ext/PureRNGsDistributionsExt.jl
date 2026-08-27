module PureRNGsDistributionsExt

import Distributions
import PureRNGs
import Random

const IR = PureRNGs
const _FloatType = Union{Float32,Float64}
const _MappedDistribution = Union{
    Distributions.Normal{Float32},
    Distributions.Normal{Float64},
    Distributions.Uniform{Float32},
    Distributions.Uniform{Float64},
    Distributions.Exponential{Float32},
    Distributions.Exponential{Float64},
    Distributions.Bernoulli{Float32},
    Distributions.Bernoulli{Float64},
}
const _FixedDistribution = Union{_MappedDistribution,Distributions.DiscreteUniform}

@inline _result_type(::Distributions.Normal{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Uniform{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Exponential{T}) where {T<:_FloatType} = T
@inline _result_type(::Distributions.Bernoulli{T}) where {T<:_FloatType} = Bool
@inline _result_type(::Distributions.DiscreteUniform) = Int

@noinline function _invalid_parameters(name)
    throw(ArgumentError("invalid $(name) parameters"))
end

@inline function _validate_distribution(d::Distributions.Normal{T}) where {T<:_FloatType}
    isfinite(d.μ) || _invalid_parameters("Normal")
    isfinite(d.σ) || _invalid_parameters("Normal")
    d.σ >= zero(T) || _invalid_parameters("Normal")
    return nothing
end

@inline function _validate_distribution(d::Distributions.Uniform{T}) where {T<:_FloatType}
    isfinite(d.a) || _invalid_parameters("Uniform")
    isfinite(d.b) || _invalid_parameters("Uniform")
    d.a < d.b || _invalid_parameters("Uniform")
    isfinite(d.b - d.a) || _invalid_parameters("Uniform")
    return nothing
end

@inline function _validate_distribution(
    d::Distributions.Exponential{T},
) where {T<:_FloatType}
    isfinite(d.θ) || _invalid_parameters("Exponential")
    d.θ > zero(T) || _invalid_parameters("Exponential")
    return nothing
end

@inline function _validate_distribution(d::Distributions.Bernoulli{T}) where {T<:_FloatType}
    isfinite(d.p) || _invalid_parameters("Bernoulli")
    d.p >= zero(T) || _invalid_parameters("Bernoulli")
    d.p <= one(T) || _invalid_parameters("Bernoulli")
    return nothing
end

@inline function _validate_distribution(d::Distributions.DiscreteUniform)
    d.a <= d.b || _invalid_parameters("DiscreteUniform")
    return nothing
end

@inline _distribution_span(::Distributions.Normal{T}) where {T<:_FloatType} =
    IR._normal_bits(T)
@inline _distribution_span(::Distributions.Uniform{T}) where {T<:_FloatType} =
    IR._draw_bits(T)
@inline _distribution_span(::Distributions.Exponential{T}) where {T<:_FloatType} =
    IR._exponential_bits(T)
@inline _distribution_span(::Distributions.Bernoulli{T}) where {T<:_FloatType} =
    IR._draw_bits(T)

@inline function _discrete_span(d::Distributions.DiscreteUniform)
    # UInt64 wrap maps cardinality 2^64 to the R55 zero sentinel.
    return (d.b % UInt64 - d.a % UInt64) + UInt64(1)
end

@inline _distribution_span(d::Distributions.DiscreteUniform) =
    IR._range_bits(_discrete_span(d))

@inline _map_distribution(d::Distributions.Normal, z) = fma(d.σ, z, d.μ)
@inline function _map_distribution(d::Distributions.Uniform, u)
    width = d.b - d.a
    scaled = width * u
    return d.a + scaled
end
@inline _map_distribution(d::Distributions.Exponential, x) = d.θ * x
@inline _map_distribution(d::Distributions.Bernoulli, u) = u < d.p

struct _DistributionCodec{D<:_MappedDistribution,B<:IR._BackendToken} <: IR._MappedFillCodec
    distribution::D
    device::B
end

@inline function _draw_distribution_unchecked(rng, position, d::_MappedDistribution)
    codec = _DistributionCodec(d, rng.device)
    return IR._transformed_draw_unchecked(codec, rng, position, _result_type(d))
end

@inline function _draw_distribution_unchecked(
    rng,
    position,
    d::Distributions.DiscreteUniform,
)
    range = d.a:d.b
    return IR._draw_range_unchecked(rng, position, range, _discrete_span(d))
end

@inline function _draw_distribution(rng, d::_FixedDistribution)
    _validate_distribution(d)
    width = _distribution_span(d)
    IR._reserve(rng, UInt64(width), UInt64(0))
    return _draw_distribution_unchecked(rng, rng.position, d)
end

@inline function _draw_distribution_next(rng, d::_FixedDistribution)
    _validate_distribution(d)
    width = _distribution_span(d)
    next_rng = IR._reserve(rng, UInt64(width), UInt64(0))
    return next_rng, _draw_distribution_unchecked(rng, rng.position, d)
end

@inline function Random.rand(rng::IR._ScalarUniformFamily, d::_FixedDistribution)
    return _draw_distribution(rng, d)
end

@inline function IR.rand_next(rng::IR._ScalarUniformFamily, d::_FixedDistribution)
    return _draw_distribution_next(rng, d)
end

@inline function IR.randat(
    rng::IR._ScalarUniformFamily,
    d::_FixedDistribution,
    index::Integer,
)
    _validate_distribution(d)
    addressed = IR._addressed_rng(rng, _distribution_span(d), index)
    return _draw_distribution_unchecked(addressed, addressed.position, d)
end

@inline IR._fill_width(codec::_DistributionCodec, ::Type) =
    _distribution_span(codec.distribution)

@inline IR._fill_family(::_DistributionCodec{<:Distributions.Normal}) = IR.FAMILY_NORMAL
@inline IR._fill_family(::_DistributionCodec{<:Distributions.Uniform}) = IR.FAMILY_BITS
@inline IR._fill_family(::_DistributionCodec{<:Distributions.Exponential}) = IR.FAMILY_EXP
@inline IR._fill_family(::_DistributionCodec{<:Distributions.Bernoulli}) = IR.FAMILY_BITS

@inline function IR._cooperative_value(
    codec::_DistributionCodec{Distributions.Normal{T}},
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(codec.distribution, IR._normal_from_bits(T, raw))
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{Distributions.Uniform{T}},
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(codec.distribution, IR._from_bits(T, raw))
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{Distributions.Exponential{T}},
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(
        codec.distribution,
        IR._exponential_from_bits(codec.device, T, raw),
    )
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{Distributions.Bernoulli{T}},
    ::Type{Bool},
    raw,
) where {T<:_FloatType}
    return _map_distribution(codec.distribution, IR._from_bits(T, raw))
end

@inline _scalar_store_plan(plan) = plan
@inline function _scalar_store_plan(
    plan::Tuple{Val{:cooperative},Val{O},Val{L},Val{P}},
) where {O,L,P}
    return plan[1], plan[2], plan[3]
end

@inline function IR._transformed_fill_plan(
    ::_DistributionCodec{Distributions.Normal{T}},
    backend,
    rng,
    ::Type,
) where {T<:_FloatType}
    return IR._device_normal_fill_plan(backend, rng, T)
end

@inline function IR._transformed_fill_plan(
    ::_DistributionCodec{Distributions.Uniform{T}},
    backend,
    rng,
    ::Type,
) where {T<:_FloatType}
    return _scalar_store_plan(IR._device_uniform_fill_plan(backend, rng, T))
end

@inline IR._transformed_fill_plan(
    ::_DistributionCodec{<:Distributions.Exponential},
    backend,
    rng,
    ::Type,
) = nothing

@inline function IR._transformed_fill_plan(
    ::_DistributionCodec{Distributions.Bernoulli{T}},
    backend,
    rng,
    ::Type{Bool},
) where {T<:_FloatType}
    return _scalar_store_plan(IR._device_uniform_fill_plan(backend, rng, T))
end

@noinline function _metal_distribution_error()
    throw(ArgumentError("fixed-distribution draws are not supported on Metal"))
end

@inline function _check_serviceability(rng, result_type)
    IR._check_serviceability(rng, result_type)
    rng.device isa IR._MetalBackend && _metal_distribution_error()
    return nothing
end

@inline function _fill_distribution_prevalidated!(rng, d, destination, threaded)
    codec = _DistributionCodec(d, rng.device)
    return IR._fill_transformed_prevalidated!(rng, destination, threaded, codec)
end

@inline function _fill_distribution_prevalidated!(
    rng,
    d::Distributions.DiscreteUniform,
    destination,
    threaded,
)
    range = d.a:d.b
    span = _discrete_span(d)
    width = IR._range_bits(span)
    bits_lo, bits_hi = IR._bit_span(UInt64(length(destination)), width)
    next_rng = IR._reserve(rng, bits_lo, bits_hi)
    isempty(destination) && return next_rng, destination
    if !threaded && rng.device isa IR._CPUBackend
        IR._fill_range_cpu_unchecked!(
            rng,
            rng.position,
            destination,
            range,
            span,
            eachindex(destination),
        )
        return next_rng, destination
    end
    backend = IR._fill_backend(destination)
    IR._launch_range!(backend, rng, destination, range, span)
    return next_rng, destination
end

@inline function _rand_distribution_next_fill!(rng, d, destination, threaded)
    IR._check_fill_device(rng, destination)
    _check_serviceability(rng, eltype(destination))
    _validate_distribution(d)
    return _fill_distribution_prevalidated!(rng, d, destination, threaded)
end

@inline function _rand_distribution_next_array(rng, d, dims)
    result_type = _result_type(d)
    _check_serviceability(rng, result_type)
    _validate_distribution(d)
    destination = IR._allocate_array(rng.device, result_type, dims)
    return _fill_distribution_prevalidated!(rng, d, destination, true)
end

@inline function Random.rand(
    rng::IR._ScalarUniformFamily,
    d::_FixedDistribution,
    dim1::Integer,
    dims::Integer...,
)
    _, destination = _rand_distribution_next_array(rng, d, (dim1, dims...))
    return destination
end

@inline function IR.rand_next(
    rng::IR._ScalarUniformFamily,
    d::_FixedDistribution,
    dim1::Integer,
    dims::Integer...,
)
    return _rand_distribution_next_array(rng, d, (dim1, dims...))
end

for (distribution_type, result_type) in (
    (Distributions.Normal{Float32}, Float32),
    (Distributions.Normal{Float64}, Float64),
    (Distributions.Uniform{Float32}, Float32),
    (Distributions.Uniform{Float64}, Float64),
    (Distributions.Exponential{Float32}, Float32),
    (Distributions.Exponential{Float64}, Float64),
    (Distributions.Bernoulli{Float32}, Bool),
    (Distributions.Bernoulli{Float64}, Bool),
    (Distributions.DiscreteUniform, Int),
)
    @eval begin
        @inline function Random.rand!(
            rng::IR._ScalarUniformFamily,
            d::$distribution_type,
            destination::AbstractArray{$result_type};
            threaded::Bool = true,
        )
            _, result = _rand_distribution_next_fill!(rng, d, destination, threaded)
            return result
        end

        @inline function IR.rand_next!(
            rng::IR._ScalarUniformFamily,
            d::$distribution_type,
            destination::AbstractArray{$result_type};
            threaded::Bool = true,
        )
            return _rand_distribution_next_fill!(rng, d, destination, threaded)
        end
    end
end

end
