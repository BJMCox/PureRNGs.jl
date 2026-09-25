module PureRNGsDistributionsExt

import Distributions
import PureRNGs
import Random
using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload

const IR = PureRNGs
include("distributions_common.jl")
include("distributions_gamma.jl")

# Reactant shares the common file but not Gamma, whose draw can loop.
const _NativeDistribution = Union{_FixedDistribution,_GammaDistribution}

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

@inline function _draw_distribution(rng, d::_NativeDistribution)
    _validate_distribution(d)
    width = _distribution_span(d)
    IR._reserve(rng, UInt64(width), UInt64(0))
    return _draw_distribution_unchecked(rng, rng.position, d)
end

@inline function _draw_distribution_next(rng, d::_NativeDistribution)
    _validate_distribution(d)
    width = _distribution_span(d)
    next_rng = IR._reserve(rng, UInt64(width), UInt64(0))
    return _draw_distribution_unchecked(rng, rng.position, d), next_rng
end

@inline function Random.rand(rng::IR._ScalarUniformGenerators, d::_NativeDistribution)
    return _draw_distribution(rng, d)
end

@inline function IR.rand_next(rng::IR._ScalarUniformGenerators, d::_NativeDistribution)
    return _draw_distribution_next(rng, d)
end

@inline function IR.rand_at(
    rng::IR._ScalarUniformGenerators,
    d::_NativeDistribution,
    index::Integer,
)
    _validate_distribution(d)
    addressed = IR._addressed_rng(rng, _distribution_span(d), index)
    return _draw_distribution_unchecked(addressed, addressed.position, d)
end

@inline IR._fill_width(codec::_DistributionCodec, ::Type) =
    _distribution_span(codec.distribution)

@inline IR._cooperative_value(codec::_DistributionCodec, ::Type, raw) = _map_distribution(
    IR._NativeTransformOps(),
    codec.distribution,
    _base_variates(codec.distribution, codec.device, raw)...,
)

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
    d::Union{_NativeDistribution,Distributions.Categorical},
) = IR._check_serviceability(rng, _result_type(d))

# Metal serves no distribution draw. The result type is checked first so a
# type Metal does not serve keeps reporting the device error.
@inline function IR._check_serviceability(
    rng::IR._BackendGenerators{IR._MetalBackend},
    d::Union{_NativeDistribution,Distributions.Categorical},
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

@inline function _rand_distribution_next_fill!(rng, d, destination, threaded::Bool)
    IR._check_fill_device(rng, destination)
    IR._check_serviceability(rng, d)
    _validate_distribution(d)
    return _fill_distribution_prevalidated!(rng, d, destination, threaded)
end

@inline function _rand_distribution_next_array(rng, d, dims, threaded::Bool)
    result_type = _result_type(d)
    IR._check_serviceability(rng, d)
    _validate_distribution(d)
    destination = IR._allocate_draw_array(rng.device, result_type, dims)
    return _fill_distribution_prevalidated!(rng, d, destination, threaded)
end

@inline function Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_NativeDistribution,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    destination, _ = _rand_distribution_next_array(rng, d, (dim1, dims...), threaded)
    return destination
end

@inline function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_NativeDistribution,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
)
    return _rand_distribution_next_array(rng, d, (dim1, dims...), threaded)
end

@inline function Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Union{_FloatMapped{T},Distributions.Gamma{T}},
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_FloatType}
    result, _ = _rand_distribution_next_fill!(rng, d, destination, threaded)
    return result
end

@inline function Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Bernoulli{<:_FloatType},
    destination::AbstractArray{Bool};
    threaded::Bool = false,
)
    result, _ = _rand_distribution_next_fill!(rng, d, destination, threaded)
    return result
end

@inline function Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.DiscreteUniform,
    destination::AbstractArray{Int};
    threaded::Bool = false,
)
    result, _ = _rand_distribution_next_fill!(rng, d, destination, threaded)
    return result
end

@inline function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Union{_FloatMapped{T},Distributions.Gamma{T}},
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_FloatType}
    return _rand_distribution_next_fill!(rng, d, destination, threaded)
end

@inline function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Bernoulli{<:_FloatType},
    destination::AbstractArray{Bool};
    threaded::Bool = false,
)
    return _rand_distribution_next_fill!(rng, d, destination, threaded)
end

@inline function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.DiscreteUniform,
    destination::AbstractArray{Int};
    threaded::Bool = false,
)
    return _rand_distribution_next_fill!(rng, d, destination, threaded)
end

include("distributions_categorical.jl")

# A multivariate normal draw is Distributions' own map, `μ + L z`, applied to
# `length(d)` standard normal draws at the held position, where `L` is the
# covariance's lower factor. `n` draws are one fill, a column per draw. The
# covariance lives in host memory, so the draws run on the CPU.
const _FloatMvNormal = Distributions.MvNormal{<:_FloatType}

@inline function _whitened_to_mvnormal!(d, values)
    Distributions.PDMats.unwhiten!(d.Σ, values)
    values .+= d.μ
    return values
end

function IR.rand_next(rng::IR._CPUGenerators, d::_FloatMvNormal)
    values, next_rng = IR.randn_next(rng, eltype(d), length(d))
    return _whitened_to_mvnormal!(d, values), next_rng
end
Random.rand(rng::IR._CPUGenerators, d::_FloatMvNormal) = first(IR.rand_next(rng, d))
function IR.rand_at(rng::IR._CPUGenerators, d::_FloatMvNormal, index::Integer)
    index < 1 && IR._invalid_address_index()
    start = Base.Checked.checked_mul(index - one(index), length(d)) + 1
    values = IR.randn_at(rng, eltype(d), start:(start+length(d)-1))
    return _whitened_to_mvnormal!(d, values)
end

function IR.rand_next(
    rng::IR._CPUGenerators,
    d::_FloatMvNormal,
    n::Integer;
    threaded::Bool = false,
)
    values, next_rng = IR.randn_next(rng, eltype(d), length(d), n; threaded)
    return _whitened_to_mvnormal!(d, values), next_rng
end
Random.rand(rng::IR._CPUGenerators, d::_FloatMvNormal, n::Integer; threaded::Bool = false) =
    first(IR.rand_next(rng, d, n; threaded))

function IR.rand_next!(
    rng::IR._CPUGenerators,
    d::Distributions.MvNormal{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType}
    size(destination, 1) == length(d) || throw(
        DimensionMismatch(
            "destination has $(size(destination, 1)) rows for a $(length(d))-dimensional MvNormal",
        ),
    )
    _, next_rng = IR.randn_next!(rng, destination; threaded)
    return _whitened_to_mvnormal!(d, destination), next_rng
end
Random.rand!(
    rng::IR._CPUGenerators,
    d::Distributions.MvNormal{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType} = first(IR.rand_next!(rng, d, destination; threaded))

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
