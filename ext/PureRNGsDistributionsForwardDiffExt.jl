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

const _DualMapped = _FloatMapped{<:ForwardDiff.Dual}

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

function _dual_codec(rng, d)
    primal = _primal(d)
    _validate_distribution(primal)
    return _DualCodec(d, primal, rng.device)
end

function IR.rand_next(rng::IR._ScalarUniformGenerators, d::_DualMapped)
    codec = _dual_codec(rng, d)
    next_rng = IR._reserve(rng, UInt64(_distribution_span(codec.primal)), UInt64(0))
    value =
        IR._transformed_draw_unchecked(codec, rng, rng.position, Distributions.partype(d))
    return value, next_rng
end
Random.rand(rng::IR._ScalarUniformGenerators, d::_DualMapped) = first(IR.rand_next(rng, d))
function IR.rand_at(rng::IR._ScalarUniformGenerators, d::_DualMapped, index::Integer)
    codec = _dual_codec(rng, d)
    addressed = IR._addressed_rng(rng, _distribution_span(codec.primal), index)
    return IR._transformed_draw_unchecked(
        codec,
        addressed,
        addressed.position,
        Distributions.partype(d),
    )
end

# Fills run on the CPU, where the dual element type is an ordinary array element.
function IR.rand_next!(
    rng::IR._CPUGenerators,
    d::_DualMapped,
    destination::AbstractArray{<:ForwardDiff.Dual};
    threaded::Bool = false,
)
    IR._check_fill_device(rng, destination)
    return IR._fill_prevalidated!(rng, destination, threaded, _dual_codec(rng, d))
end
Random.rand!(
    rng::IR._CPUGenerators,
    d::_DualMapped,
    destination::AbstractArray{<:ForwardDiff.Dual};
    threaded::Bool = false,
) = first(IR.rand_next!(rng, d, destination; threaded))

IR.rand_next(rng::IR._CPUGenerators, d::_DualMapped, dims::Dims; threaded::Bool = false) =
    IR.rand_next!(rng, d, Array{Distributions.partype(d)}(undef, dims); threaded)
IR.rand_next(
    rng::IR._CPUGenerators,
    d::_DualMapped,
    dim1::Integer,
    dims::Integer...;
    threaded::Bool = false,
) = IR.rand_next(rng, d, (dim1, dims...); threaded)
Random.rand(rng::IR._CPUGenerators, d::_DualMapped, dims::Dims; threaded::Bool = false) =
    first(IR.rand_next(rng, d, dims; threaded))
Random.rand(
    rng::IR._CPUGenerators,
    d::_DualMapped,
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

_whitened_to_mvnormal(d, values) = d.μ .+ Distributions.PDMats.unwhiten(d.Σ, values)

function IR.rand_next(rng::IR._CPUGenerators, d::_DualMvNormal)
    values, next_rng = IR.randn_next(rng, _primal_type(eltype(d)), length(d))
    return _whitened_to_mvnormal(d, values), next_rng
end
Random.rand(rng::IR._CPUGenerators, d::_DualMvNormal) = first(IR.rand_next(rng, d))
function IR.rand_next(
    rng::IR._CPUGenerators,
    d::_DualMvNormal,
    n::Integer;
    threaded::Bool = false,
)
    values, next_rng = IR.randn_next(rng, _primal_type(eltype(d)), length(d), n; threaded)
    return _whitened_to_mvnormal(d, values), next_rng
end
Random.rand(rng::IR._CPUGenerators, d::_DualMvNormal, n::Integer; threaded::Bool = false) =
    first(IR.rand_next(rng, d, n; threaded))

end
