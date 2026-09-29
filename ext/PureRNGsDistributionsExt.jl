module PureRNGsDistributionsExt

import Distributions
import PureRNGs
import Random
using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload

const IR = PureRNGs
include("distributions_common.jl")
include("distributions_gamma.jl")
include("distributions_dirichlet.jl")

# Reactant shares the common file but not the Gamma family, whose draws can loop.
const _NativeDistribution = Union{_FixedDistribution,_GammaFamilyDistribution}

struct _DistributionCodec{D<:_MappedDistribution{<:_FloatType},B<:IR._BackendToken} <:
       IR._MappedFillCodec
    distribution::D
    device::B
end

# The codec a distribution's draws and fills run through.
@inline _distribution_codec(d::_MappedDistribution{<:_FloatType}, backend) =
    _DistributionCodec(d, backend)
@inline _distribution_codec(d::Distributions.DiscreteUniform, backend) =
    IR._RangeCodec(d.a:d.b, _discrete_span(d))

@inline function IR._engine_rand_next(rng, d::_NativeDistribution)
    _validate_distribution(d)
    codec = _distribution_codec(d, IR._engine_backend(rng))
    return IR._engine_draw_next(rng, codec, _result_type(d))
end

@inline function IR._engine_rand_at(rng, d::_NativeDistribution, index::Integer)
    _validate_distribution(d)
    codec = _distribution_codec(d, IR._engine_backend(rng))
    return IR._engine_draw_at(rng, codec, _result_type(d), index)
end

@inline Random.rand(rng::IR._ScalarUniformGenerators, d::_NativeDistribution) =
    first(IR._engine_rand_next(rng, d))
@inline IR.rand_next(rng::IR._ScalarUniformGenerators, d::_NativeDistribution) =
    IR._engine_rand_next(rng, d)
@inline IR.rand_at(
    rng::IR._ScalarUniformGenerators,
    d::_NativeDistribution,
    index::Integer,
) = IR._engine_rand_at(rng, d, index)

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

@inline IR._check_serviceability(rng, d::_NativeDistribution) =
    IR._check_serviceability(rng, _result_type(d))

# Categorical labels come from a Float64 cumulative table, as weighted samples do.
@inline function IR._check_serviceability(rng, d::Distributions.Categorical)
    IR._check_weighted_serviceability(rng)
    return IR._check_serviceability(rng, _result_type(d))
end

@inline function _rand_distribution_next_fill!(rng, d, destination, threaded::Bool)
    IR._check_fill_device(rng, destination)
    IR._check_serviceability(rng, d)
    _validate_distribution(d)
    codec = _distribution_codec(d, IR._engine_backend(rng))
    return IR._engine_fill!(rng, destination, threaded, codec)
end

@inline function _rand_distribution_next_array(rng, d, dims, threaded::Bool)
    IR._check_serviceability(rng, d)
    _validate_distribution(d)
    backend = IR._engine_backend(rng)
    destination = IR._allocate_draw_array(backend, _result_type(d), dims)
    return IR._engine_fill!(rng, destination, threaded, _distribution_codec(d, backend))
end

@inline IR._engine_rand_next(
    rng,
    d::_NativeDistribution,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) = _rand_distribution_next_array(rng, d, (dim1, dims...), threaded)

@inline IR._engine_rand_next!(
    rng,
    d::Union{_FloatMapped{T},_GammaFamily{T}},
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_FloatType} = _rand_distribution_next_fill!(rng, d, destination, threaded)
@inline IR._engine_rand_next!(
    rng,
    d::Distributions.Bernoulli{<:_FloatType},
    destination::AbstractArray{Bool};
    threaded::Bool = false,
) = _rand_distribution_next_fill!(rng, d, destination, threaded)
@inline IR._engine_rand_next!(
    rng,
    d::Distributions.DiscreteUniform,
    destination::AbstractArray{Int};
    threaded::Bool = false,
) = _rand_distribution_next_fill!(rng, d, destination, threaded)

@inline Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_NativeDistribution,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) = first(IR._engine_rand_next(rng, d, dim1, dims...; threaded))
@inline IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_NativeDistribution,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) = IR._engine_rand_next(rng, d, dim1, dims...; threaded)

@inline Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Union{_FloatMapped{T},_GammaFamily{T}},
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_FloatType} = first(IR._engine_rand_next!(rng, d, destination; threaded))
@inline Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Bernoulli{<:_FloatType},
    destination::AbstractArray{Bool};
    threaded::Bool = false,
) = first(IR._engine_rand_next!(rng, d, destination; threaded))
@inline Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.DiscreteUniform,
    destination::AbstractArray{Int};
    threaded::Bool = false,
) = first(IR._engine_rand_next!(rng, d, destination; threaded))

@inline IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Union{_FloatMapped{T},_GammaFamily{T}},
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:_FloatType} = IR._engine_rand_next!(rng, d, destination; threaded)
@inline IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.Bernoulli{<:_FloatType},
    destination::AbstractArray{Bool};
    threaded::Bool = false,
) = IR._engine_rand_next!(rng, d, destination; threaded)
@inline IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.DiscreteUniform,
    destination::AbstractArray{Int};
    threaded::Bool = false,
) = IR._engine_rand_next!(rng, d, destination; threaded)

include("distributions_categorical.jl")

# A multivariate normal draw is Distributions' own map, `μ + L z`, applied to
# `length(d)` standard normal draws at the held position, where `L` is the
# covariance's lower factor. `n` draws are one fill, a column per draw. PDMats
# keeps the factor in host memory, so a device draw moves it once per call and
# applies it with the device's array operations.
const _FloatMvNormal = Distributions.MvNormal{<:_FloatType}
const _PDMats = Distributions.PDMats
const _LinearAlgebra = Distributions.LinearAlgebra

_covariance_storage(Σ::_PDMats.PDMat) = _LinearAlgebra.cholesky(Σ).factors
_covariance_storage(Σ::_PDMats.PDiagMat) = Σ.diag
_covariance_storage(Σ::_PDMats.ScalMat) = nothing

_device_factor(device, Σ::_PDMats.PDMat) =
    IR._transfer_array(device, Matrix(_PDMats.chol_lower(_LinearAlgebra.cholesky(Σ))))
_device_unwhiten!(device, Σ::_PDMats.PDMat, values) =
    _LinearAlgebra.lmul!(_LinearAlgebra.LowerTriangular(_device_factor(device, Σ)), values)
# Metal's in-place triangular product gave a 3 x 2049 draw different values on
# repeated calls, up to 0.23 from the host product. The dense product of the
# zero-filled factor writes a new array and matches the host.
_device_unwhiten!(device::IR._MetalBackend, Σ::_PDMats.PDMat, values) =
    copyto!(values, _device_factor(device, Σ) * values)
_device_unwhiten!(device, Σ::_PDMats.PDiagMat, values) =
    values .*= sqrt.(IR._transfer_array(device, collect(Σ.diag)))
_device_unwhiten!(device, Σ::_PDMats.ScalMat, values) = _PDMats.unwhiten!(Σ, values)

@inline function _whitened_to_mvnormal!(::IR._CPUBackend, d, values)
    _PDMats.unwhiten!(d.Σ, values)
    values .+= d.μ
    return values
end

@inline function _whitened_to_mvnormal!(device, d, values)
    _device_unwhiten!(device, d.Σ, values)
    values .+= IR._transfer_array(device, collect(d.μ))
    return values
end

function IR._engine_rand_next(rng, d::_FloatMvNormal)
    values, next_rng = IR._engine_randn_next(rng, eltype(d), length(d))
    return _whitened_to_mvnormal!(IR._engine_backend(rng), d, values), next_rng
end
function IR._engine_rand_at(rng, d::_FloatMvNormal, index::Integer)
    index < 1 && IR._invalid_address_index()
    start = Base.Checked.checked_mul(index - one(index), length(d)) + 1
    values = IR._engine_randn_at(rng, eltype(d), start:(start+length(d)-1))
    return _whitened_to_mvnormal!(IR._engine_backend(rng), d, values)
end
function IR._engine_rand_next(rng, d::_FloatMvNormal, n::Integer; threaded::Bool = false)
    values, next_rng = IR._engine_randn_next(rng, eltype(d), length(d), n; threaded)
    return _whitened_to_mvnormal!(IR._engine_backend(rng), d, values), next_rng
end
function IR._engine_rand_next!(
    rng,
    d::Distributions.MvNormal{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType}
    size(destination, 1) == length(d) || throw(
        DimensionMismatch(
            "destination has $(size(destination, 1)) rows for a $(length(d))-dimensional MvNormal",
        ),
    )
    IR._check_parameter_overlap(destination, d.μ, "the mean")
    IR._check_parameter_overlap(
        destination,
        _covariance_storage(d.Σ),
        "the covariance factor",
    )
    _, next_rng = IR._engine_randn_next!(rng, destination; threaded)
    return _whitened_to_mvnormal!(IR._engine_backend(rng), d, destination), next_rng
end

IR.rand_next(rng::IR._ScalarUniformGenerators, d::_FloatMvNormal) =
    IR._engine_rand_next(rng, d)
Random.rand(rng::IR._ScalarUniformGenerators, d::_FloatMvNormal) =
    first(IR._engine_rand_next(rng, d))
IR.rand_at(rng::IR._ScalarUniformGenerators, d::_FloatMvNormal, index::Integer) =
    IR._engine_rand_at(rng, d, index)
IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_FloatMvNormal,
    n::Integer;
    threaded::Bool = false,
) = IR._engine_rand_next(rng, d, n; threaded)
Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_FloatMvNormal,
    n::Integer;
    threaded::Bool = false,
) = first(IR._engine_rand_next(rng, d, n; threaded))
IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.MvNormal{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType} = IR._engine_rand_next!(rng, d, destination; threaded)
Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::Distributions.MvNormal{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType} = first(IR._engine_rand_next!(rng, d, destination; threaded))

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
