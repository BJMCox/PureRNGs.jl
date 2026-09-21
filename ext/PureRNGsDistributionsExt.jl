module PureRNGsDistributionsExt

import Distributions
import PureRNGs
import Random
using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload

const IR = PureRNGs
include("distributions_common.jl")

struct _DistributionCodec{D<:_MappedDistribution{<:_FloatType},B<:IR._BackendToken} <:
       IR._MappedFillCodec
    distribution::D
    device::B
end

@inline function _draw_distribution_unchecked(
    rng,
    position,
    d::_MappedDistribution{<:_FloatType},
)
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
    return _draw_distribution_unchecked(rng, rng.position, d), next_rng
end

@inline function Random.rand(rng::IR._ScalarUniformGenerators, d::_FixedDistribution)
    return _draw_distribution(rng, d)
end

@inline function IR.rand_next(rng::IR._ScalarUniformGenerators, d::_FixedDistribution)
    return _draw_distribution_next(rng, d)
end

@inline function IR.rand_at(
    rng::IR._ScalarUniformGenerators,
    d::_FixedDistribution,
    index::Integer,
)
    _validate_distribution(d)
    addressed = IR._addressed_rng(rng, _distribution_span(d), index)
    return _draw_distribution_unchecked(addressed, addressed.position, d)
end

@inline IR._fill_width(codec::_DistributionCodec, ::Type) =
    _distribution_span(codec.distribution)

@inline function IR._cooperative_value(
    codec::_DistributionCodec{<:Union{Distributions.Normal{T},Distributions.LogNormal{T}}},
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(
        IR._NativeTransformOps(),
        codec.distribution,
        IR._normal_from_bits(codec.device, T, raw),
    )
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{<:Union{Distributions.Logistic{T},Distributions.Cauchy{T}}},
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(
        IR._NativeTransformOps(),
        codec.distribution,
        IR._open_midpoint(T, raw),
    )
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{<:Union{Distributions.Gumbel{T},Distributions.Frechet{T}}},
    ::Type,
    raw,
) where {T<:_FloatType}
    u = IR._open_midpoint(T, raw)
    e = IR._exponential_transform(codec.device, T, one(T) - u)
    return _map_distribution(IR._NativeTransformOps(), codec.distribution, e)
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{Distributions.Uniform{T}},
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(
        IR._NativeTransformOps(),
        codec.distribution,
        IR._from_bits(T, raw),
    )
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{
        <:Union{
            Distributions.Exponential{T},
            Distributions.Weibull{T},
            Distributions.Rayleigh{T},
            Distributions.Pareto{T},
        },
    },
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(
        IR._NativeTransformOps(),
        codec.distribution,
        IR._exponential_from_bits(codec.device, T, raw),
    )
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{Distributions.TriangularDist{T}},
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(
        IR._NativeTransformOps(),
        codec.distribution,
        IR._from_bits(T, raw),
    )
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{Distributions.Laplace{T}},
    ::Type,
    raw,
) where {T<:_FloatType}
    return _map_distribution(
        IR._NativeTransformOps(),
        codec.distribution,
        IR._exponential_from_bits(codec.device, T, raw >> 1),
        isodd(raw),
    )
end

@inline function IR._cooperative_value(
    codec::_DistributionCodec{Distributions.Bernoulli{T}},
    ::Type{Bool},
    raw,
) where {T<:_FloatType}
    return _map_distribution(
        IR._NativeTransformOps(),
        codec.distribution,
        IR._from_bits(T, raw),
    )
end

@inline _scalar_store_plan(plan) = plan
@inline function _scalar_store_plan(
    plan::Tuple{Val{:cooperative},Val{O},Val{L},Val{P}},
) where {O,L,P}
    return plan[1], plan[2], plan[3]
end

@inline function IR._device_fill_plan(
    backend,
    rng,
    ::_DistributionCodec{
        <:Union{
            Distributions.Normal{T},
            Distributions.LogNormal{T},
            Distributions.Logistic{T},
            Distributions.Gumbel{T},
            Distributions.Frechet{T},
            Distributions.Cauchy{T},
        },
    },
    ::Type,
) where {T<:_FloatType}
    return _scalar_store_plan(
        IR._device_fill_plan(backend, rng, IR._NormalCodec(rng.device), T),
    )
end

@inline function IR._device_fill_plan(
    backend,
    rng,
    ::_DistributionCodec{Distributions.Uniform{T}},
    ::Type,
) where {T<:_FloatType}
    return IR._device_fill_plan(backend, rng, Val(:uniform), T)
end

@inline IR._device_fill_plan(
    backend,
    rng,
    codec::_DistributionCodec{
        <:Union{
            Distributions.Exponential,
            Distributions.Weibull,
            Distributions.Rayleigh,
            Distributions.Pareto,
        },
    },
    ::Type{T},
) where {T<:_FloatType} =
    IR._device_fill_plan(backend, rng, IR._ExponentialCodec(codec.device), T)

@inline IR._device_fill_plan(
    backend,
    rng,
    ::_DistributionCodec{<:Distributions.Laplace},
    ::Type,
) = nothing

@inline function IR._device_fill_plan(
    backend,
    rng,
    ::_DistributionCodec{Distributions.Bernoulli{T}},
    ::Type{Bool},
) where {T<:_FloatType}
    return _scalar_store_plan(IR._device_fill_plan(backend, rng, Val(:uniform), T))
end

@inline function IR._device_fill_plan(
    backend,
    rng,
    ::_DistributionCodec{Distributions.TriangularDist{T}},
    ::Type,
) where {T<:_FloatType}
    return IR._device_fill_plan(backend, rng, Val(:uniform), T)
end

@noinline function _metal_distribution_error()
    throw(ArgumentError("fixed-distribution draws are not supported on Metal"))
end

@inline IR._check_serviceability(
    rng,
    d::Union{_FixedDistribution,Distributions.Categorical},
) = IR._check_serviceability(rng, _result_type(d))

# [R41] Metal serves no distribution draw. The result type is checked first so a
# type Metal does not serve keeps reporting the device error.
@inline function IR._check_serviceability(
    rng::IR._BackendGenerators{IR._MetalBackend},
    d::Union{_FixedDistribution,Distributions.Categorical},
)
    IR._check_serviceability(rng, _result_type(d))
    return _metal_distribution_error()
end

@inline function _fill_distribution_prevalidated!(rng, d, destination, threaded)
    codec = _DistributionCodec(d, rng.device)
    return IR._fill_prevalidated!(rng, destination, threaded, codec)
end

@inline function _fill_distribution_prevalidated!(
    rng,
    d::Distributions.DiscreteUniform,
    destination,
    threaded,
)
    codec = IR._RangeCodec(d.a:d.b, _discrete_span(d))
    return IR._fill_prevalidated!(rng, destination, threaded, codec)
end

@inline function _rand_distribution_next_fill!(rng, d, destination, threaded)
    IR._check_fill_device(rng, destination)
    IR._check_serviceability(rng, d)
    _validate_distribution(d)
    return _fill_distribution_prevalidated!(rng, d, destination, threaded)
end

@inline function _rand_distribution_next_array(rng, d, dims)
    result_type = _result_type(d)
    IR._check_serviceability(rng, d)
    _validate_distribution(d)
    destination = IR._allocate_draw_array(rng.device, result_type, dims)
    return _fill_distribution_prevalidated!(rng, d, destination, true)
end

@inline function Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_FixedDistribution,
    dim1::Integer,
    dims::Integer...,
)
    destination, _ = _rand_distribution_next_array(rng, d, (dim1, dims...))
    return destination
end

@inline function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_FixedDistribution,
    dim1::Integer,
    dims::Integer...,
)
    return _rand_distribution_next_array(rng, d, (dim1, dims...))
end

@inline function Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::_FloatMapped{T},
    destination::AbstractArray{T};
    threaded = true,
) where {T<:_FloatType}
    result, _ =
        _rand_distribution_next_fill!(rng, d, destination, IR._check_threaded(threaded))
    return result
end

@inline function Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Bernoulli{<:_FloatType},
    destination::AbstractArray{Bool};
    threaded = true,
)
    result, _ =
        _rand_distribution_next_fill!(rng, d, destination, IR._check_threaded(threaded))
    return result
end

@inline function Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.DiscreteUniform,
    destination::AbstractArray{Int};
    threaded = true,
)
    result, _ =
        _rand_distribution_next_fill!(rng, d, destination, IR._check_threaded(threaded))
    return result
end

@inline function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::_FloatMapped{T},
    destination::AbstractArray{T};
    threaded = true,
) where {T<:_FloatType}
    return _rand_distribution_next_fill!(rng, d, destination, IR._check_threaded(threaded))
end

@inline function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Bernoulli{<:_FloatType},
    destination::AbstractArray{Bool};
    threaded = true,
)
    return _rand_distribution_next_fill!(rng, d, destination, IR._check_threaded(threaded))
end

@inline function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.DiscreteUniform,
    destination::AbstractArray{Int};
    threaded = true,
)
    return _rand_distribution_next_fill!(rng, d, destination, IR._check_threaded(threaded))
end

include("distributions_categorical.jl")

@setup_workload begin
    draws = 128
    distributions = (
        Distributions.Normal(),
        Distributions.Uniform(),
        Distributions.Exponential(),
        Distributions.LogNormal(),
        Distributions.Weibull(),
        Distributions.Rayleigh(),
        Distributions.Laplace(),
        Distributions.Logistic(),
        Distributions.Gumbel(),
        Distributions.Pareto(),
        Distributions.Frechet(),
        Distributions.Cauchy(),
        Distributions.TriangularDist(0.0, 1.0, 0.5),
        Distributions.Bernoulli(),
        Distributions.DiscreteUniform(1, 6),
        Distributions.Categorical([0.2, 0.3, 0.5]),
    )

    @compile_workload begin
        rng = IR.Philox4x32(20250918)
        for d in distributions
            Random.rand(rng, d)
            value, _ = IR.rand_next(rng, d)
            Random.rand(rng, d, draws)
            values, _ = IR.rand_next(rng, d, draws)
            destination = Vector{_result_type(d)}(undef, draws)
            Random.rand!(rng, d, destination)
            IR.rand_next!(rng, d, destination)
        end
    end
end

end
