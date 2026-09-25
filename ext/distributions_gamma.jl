# The Distributions types the Gamma sampler in src/gamma.jl serves, as codecs.
# The ForwardDiff extension includes this file too, for dual parameters.

const _GammaFamily{T} = Union{
    Distributions.Gamma{T},
    Distributions.Chisq{T},
    Distributions.InverseGamma{T},
    Distributions.Beta{T},
    Distributions.TDist{T},
}
const _GammaFamilyDistribution = _GammaFamily{<:_FloatType}

_family_codec(d::Distributions.Gamma{T}, device) where {T} =
    IR._GammaCodec(d.α, d.θ, device, IR._GAMMA_CANDIDATES)
_family_codec(d::Distributions.Chisq{T}, device) where {T} =
    IR._GammaCodec(d.ν / 2, T(2), device, IR._GAMMA_CANDIDATES)
_family_codec(d::Distributions.InverseGamma{T}, device) where {T} =
    IR._InverseGammaCodec(IR._GammaCodec(d.invd.α, d.θ, device, IR._GAMMA_CANDIDATES))
_family_codec(d::Distributions.Beta{T}, device) where {T} = IR._BetaCodec(
    IR._GammaCodec(d.α, one(T), device, IR._GAMMA_CANDIDATES),
    IR._GammaCodec(d.β, one(T), device, IR._GAMMA_CANDIDATES),
)
_family_codec(d::Distributions.TDist{T}, device) where {T} =
    IR._TDistCodec(IR._GammaCodec(d.ν / 2, one(T), device, IR._GAMMA_CANDIDATES), d.ν)

# Constant messages keep the check compilable inside a GPU kernel.
@noinline _invalid_parameters(::Distributions.Gamma) =
    throw(ArgumentError("invalid Gamma parameters"))
@noinline _invalid_parameters(::Distributions.Chisq) =
    throw(ArgumentError("invalid Chisq parameters"))
@noinline _invalid_parameters(::Distributions.InverseGamma) =
    throw(ArgumentError("invalid InverseGamma parameters"))
@noinline _invalid_parameters(::Distributions.Beta) =
    throw(ArgumentError("invalid Beta parameters"))
@noinline _invalid_parameters(::Distributions.TDist) =
    throw(ArgumentError("invalid TDist parameters"))

_positive_finite(x) = isfinite(x) && x > zero(x)
_shapes(d::Distributions.Gamma) = (d.α, d.θ)
_shapes(d::Distributions.Chisq) = (d.ν,)
_shapes(d::Distributions.InverseGamma) = (d.invd.α, d.θ)
_shapes(d::Distributions.Beta) = (d.α, d.β)
_shapes(d::Distributions.TDist) = (d.ν,)

@inline function _validate_distribution(d::_GammaFamilyDistribution)
    all(_positive_finite, _shapes(d)) || _invalid_parameters(d)
    return nothing
end

@inline _result_type(::_GammaFamily{T}) where {T} = T
@inline _distribution_span(d::_GammaFamilyDistribution) =
    IR._fill_width(_family_codec(d, IR._CPU_BACKEND), _result_type(d))

@inline _draw_distribution_unchecked(rng, position, d::_GammaFamilyDistribution) =
    IR._transformed_draw_unchecked(
        _family_codec(d, rng.device),
        rng,
        position,
        _result_type(d),
    )

@inline _fill_distribution_prevalidated!(
    rng,
    d::_GammaFamilyDistribution,
    destination,
    threaded,
) = IR._fill_prevalidated!(rng, destination, threaded, _family_codec(d, rng.device))
