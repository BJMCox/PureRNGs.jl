module PureRNGsDistributionsForwardDiffExt

# A fixed distribution with dual parameters decodes the base variates of its
# primal distribution and maps them with the dual parameters, so its value is
# the primal draw and its partials are the pathwise derivatives.

import Distributions
import ForwardDiff
import PureRNGs
import Random

const IR = PureRNGs
include("distributions_common.jl")
include("distributions_gamma.jl")

const _DualMapped = _FloatMapped{<:ForwardDiff.Dual}
const _DualGammaFamily = _GammaFamily{<:ForwardDiff.Dual}
const _DualDistribution = Union{_DualMapped,_DualGammaFamily}

IR._primal_float(::Type{<:ForwardDiff.Dual{<:Any,V}}) where {V} = IR._primal_float(V)

# A dual shape draws the primal Gamma and carries the implicit shape derivative
# into the partials; the family maps carry every other parameter. Nested duals
# find no method rather than a wrong second derivative.
for (draw, slope) in (
    (:_gamma_value, :_gamma_shape_derivative),
    (:_gamma_log_value, :_gamma_log_shape_derivative),
)
    @eval function IR.$draw(
        shape::ForwardDiff.Dual{Tag,V},
        codec::IR._GammaCodec,
        rng,
        position,
        cursor,
    ) where {Tag,V<:AbstractFloat}
        s = ForwardDiff.value(shape)
        primal = IR._GammaCodec(
            s,
            ForwardDiff.value(codec.scale),
            codec.device,
            codec.candidates,
        )
        value = IR.$draw(s, primal, rng, position, cursor)
        return ForwardDiff.Dual{Tag}(
            value,
            IR.$slope(s, value) * ForwardDiff.partials(shape),
        )
    end
end

# The shared mapping shifts `u` in the parameter type, and ForwardDiff has no
# dual `tanpi`; the shift carries no parameter, so it stays in `u`'s type.
@inline _map_distribution(ops, d::Distributions.Cauchy{<:ForwardDiff.Dual}, u) =
    IR._transform_muladd(ops, d.σ, tanpi(u - oftype(u, 0.5)), d.μ)

_primal_value(x::ForwardDiff.Dual) = _primal_value(ForwardDiff.value(x))
_primal_value(x::Real) = x
_primal(d) =
    Base.typename(typeof(d)).wrapper(map(_primal_value, Distributions.params(d))...)

struct _DualCodec{D,P,B<:IR._BackendToken} <: IR._MappedFillCodec
    distribution::D
    primal::P
    device::B
end

@inline IR._fill_width(codec::_DualCodec, ::Type) = _distribution_span(codec.primal)
@inline IR._cooperative_value(codec::_DualCodec, ::Type, raw) = _map_distribution(
    IR._NativeTransformOps(),
    codec.distribution,
    _base_variates(codec.primal, codec.device, raw)...,
)

function _dual_codec(rng, d::_DualMapped)
    primal = _primal(d)
    _validate_distribution(primal)
    return _DualCodec(d, primal, rng.device)
end
function _dual_codec(rng, d::_DualGammaFamily)
    _validate_distribution(_primal(d))
    return _family_codec(d, rng.device)
end

function IR.rand_next(rng::IR._ScalarUniformGenerators, d::_DualDistribution)
    codec = _dual_codec(rng, d)
    width = IR._fill_width(codec, Distributions.partype(d))
    next_rng = IR._reserve(rng, UInt64(width), UInt64(0))
    value =
        IR._transformed_draw_unchecked(codec, rng, rng.position, Distributions.partype(d))
    return value, next_rng
end
Random.rand(rng::IR._ScalarUniformGenerators, d::_DualDistribution) =
    first(IR.rand_next(rng, d))
function IR.rand_at(rng::IR._ScalarUniformGenerators, d::_DualDistribution, index::Integer)
    codec = _dual_codec(rng, d)
    width = IR._fill_width(codec, Distributions.partype(d))
    addressed = IR._addressed_rng(rng, width, index)
    return IR._transformed_draw_unchecked(
        codec,
        addressed,
        addressed.position,
        Distributions.partype(d),
    )
end

# A dual is a plain bits type, so a device fill runs the generic kernel.
function IR.rand_next!(
    rng::IR._ScalarUniformGenerators,
    d::_DualDistribution,
    destination::AbstractArray{<:ForwardDiff.Dual};
    threaded::Bool = false,
)
    IR._check_fill_device(rng, destination)
    return IR._fill_prevalidated!(rng, destination, threaded, _dual_codec(rng, d))
end
Random.rand!(
    rng::IR._ScalarUniformGenerators,
    d::_DualDistribution,
    destination::AbstractArray{<:ForwardDiff.Dual};
    threaded::Bool = false,
) = first(IR.rand_next!(rng, d, destination; threaded))

function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_DualDistribution,
    dims::Dims;
    threaded::Bool = false,
)
    destination = IR._allocate_draw_array(rng.device, Distributions.partype(d), dims)
    return IR.rand_next!(rng, d, destination; threaded)
end
IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_DualDistribution,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) = IR.rand_next(rng, d, (dim1, dims...); threaded)
Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_DualDistribution,
    dims::Dims;
    threaded::Bool = false,
) = first(IR.rand_next(rng, d, dims; threaded))
Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_DualDistribution,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) = first(IR.rand_next(rng, d, (dim1, dims...); threaded))

# A dual MvNormal whitens standard normal draws of the primal element type. Its
# factor comes from a generic Cholesky of dual numbers, so its values match the
# primal draw to rounding rather than exactly.
const _DualMvNormal = Distributions.MvNormal{<:ForwardDiff.Dual}

_primal_type(::Type{<:ForwardDiff.Dual{<:Any,V}}) where {V} = _primal_type(V)
_primal_type(::Type{T}) where {T} = T

const _PDMats = Distributions.PDMats
const _LinearAlgebra = Distributions.LinearAlgebra

# A device draw moves the dual factor to the device once per call, as a dense
# matrix, since device arrays have no triangular product for dual elements.
_device_unwhitened(device, Σ::_PDMats.PDMat, values) =
    IR._transfer_array(device, Matrix(_PDMats.chol_lower(_LinearAlgebra.cholesky(Σ)))) *
    values
_device_unwhitened(device, Σ::_PDMats.PDiagMat, values) =
    sqrt.(IR._transfer_array(device, collect(Σ.diag))) .* values
_device_unwhitened(device, Σ::_PDMats.ScalMat, values) = sqrt(Σ.value) .* values

_whitened_to_mvnormal(::IR._CPUBackend, d, values) = d.μ .+ _PDMats.unwhiten(d.Σ, values)
_whitened_to_mvnormal(device, d, values) =
    IR._transfer_array(device, collect(d.μ)) .+ _device_unwhitened(device, d.Σ, values)

function IR.rand_next(rng::IR._ScalarUniformGenerators, d::_DualMvNormal)
    values, next_rng = IR.randn_next(rng, _primal_type(eltype(d)), length(d))
    return _whitened_to_mvnormal(rng.device, d, values), next_rng
end
Random.rand(rng::IR._ScalarUniformGenerators, d::_DualMvNormal) =
    first(IR.rand_next(rng, d))
function IR.rand_next(
    rng::IR._ScalarUniformGenerators,
    d::_DualMvNormal,
    n::Integer;
    threaded::Bool = false,
)
    values, next_rng = IR.randn_next(rng, _primal_type(eltype(d)), length(d), n; threaded)
    return _whitened_to_mvnormal(rng.device, d, values), next_rng
end
Random.rand(
    rng::IR._ScalarUniformGenerators,
    d::_DualMvNormal,
    n::Integer;
    threaded::Bool = false,
) = first(IR.rand_next(rng, d, n; threaded))

end
