"""
    StatefulRNG(rng)

Wrap an immutable generator for host-only consumers of `Random.AbstractRNG`.
Construction preserves the key and position and rebinds the generator to the CPU.
`parent(bridge)` returns the exact immutable generator currently held by the
bridge without allocating, mutating, or advancing it.

Package-owned `Array` and `BitArray` fills preflight their full counter span. A
foreign `Random` fill has no chained-scalar consumption-order guarantee. It may
leave its destination partially written on counter exhaustion because each
scalar bridge draw preflights only its own span. The held generator remains
valid at the position after the last successful draw.
"""
mutable struct StatefulRNG{R<:AbstractPureRNG} <: Random.AbstractRNG
    rng::R

    StatefulRNG{R}(::_ConstructionToken, rng::R) where {R} = new{R}(rng)
end

@inline _stateful_rng(rng::R) where {R<:AbstractPureRNG} =
    StatefulRNG{R}(_CONSTRUCTION_TOKEN, rng)

@inline StatefulRNG(rng::AbstractPureRNG) =
    _stateful_rng(MLDataDevices.CPUDevice()(rng))

@inline Base.parent(mutable_rng::StatefulRNG) = mutable_rng.rng

@inline function _commit_bridge!(
    mutable_rng::StatefulRNG{R},
    result::Tuple{R,T},
) where {R,T}
    mutable_rng.rng = first(result)
    return last(result)
end

@inline Random.rand(mutable_rng::StatefulRNG, ::Random.SamplerType{Bool}) =
    _commit_bridge!(mutable_rng, rand_next(mutable_rng.rng, Bool))
@inline Random.rand(mutable_rng::StatefulRNG, ::Random.SamplerType{UInt32}) =
    _commit_bridge!(mutable_rng, rand_next(mutable_rng.rng, UInt32))
@inline Random.rand(mutable_rng::StatefulRNG, ::Random.SamplerType{UInt64}) =
    _commit_bridge!(mutable_rng, rand_next(mutable_rng.rng, UInt64))
@inline Random.rand(mutable_rng::StatefulRNG, ::Random.SamplerType{Int32}) =
    _commit_bridge!(mutable_rng, rand_next(mutable_rng.rng, Int32))
@inline Random.rand(mutable_rng::StatefulRNG, ::Random.SamplerType{Int64}) =
    _commit_bridge!(mutable_rng, rand_next(mutable_rng.rng, Int64))
@inline Random.rand(
    mutable_rng::StatefulRNG,
    ::Random.SamplerTrivial{Random.CloseOpen01{Float32}},
) = _commit_bridge!(mutable_rng, rand_next(mutable_rng.rng, Float32))
@inline Random.rand(
    mutable_rng::StatefulRNG,
    ::Random.SamplerTrivial{Random.CloseOpen01{Float64}},
) = _commit_bridge!(mutable_rng, rand_next(mutable_rng.rng, Float64))

@inline Random.randn(mutable_rng::StatefulRNG) =
    _commit_bridge!(mutable_rng, randn_next(mutable_rng.rng, Float64))
@inline Random.randn(mutable_rng::StatefulRNG, ::Type{Float32}) =
    _commit_bridge!(mutable_rng, randn_next(mutable_rng.rng, Float32))
@inline Random.randn(mutable_rng::StatefulRNG, ::Type{Float64}) =
    _commit_bridge!(mutable_rng, randn_next(mutable_rng.rng, Float64))

@inline Random.randexp(mutable_rng::StatefulRNG) =
    _commit_bridge!(mutable_rng, randexp_next(mutable_rng.rng, Float64))
@inline Random.randexp(mutable_rng::StatefulRNG, ::Type{Float32}) =
    _commit_bridge!(mutable_rng, randexp_next(mutable_rng.rng, Float32))
@inline Random.randexp(mutable_rng::StatefulRNG, ::Type{Float64}) =
    _commit_bridge!(mutable_rng, randexp_next(mutable_rng.rng, Float64))

struct _StatefulRangeSampler{T,R<:AbstractRange{T}} <: Random.Sampler{T}
    range::R
end

@inline Random.Sampler(
    ::Type{<:StatefulRNG},
    range::AbstractRange{T},
    ::Random.Repetition,
) where {T<:_RangeInteger} = _StatefulRangeSampler(range)

@inline Random.Sampler(
    ::Type{<:StatefulRNG},
    range::AbstractUnitRange{T},
    ::Random.Repetition,
) where {T<:_RangeInteger} = _StatefulRangeSampler(range)

@inline Random.rand(mutable_rng::StatefulRNG, sampler::_StatefulRangeSampler) =
    _commit_bridge!(mutable_rng, rand_next(mutable_rng.rng, sampler.range))

const _StatefulUniform = Union{Bool,UInt32,Int32,UInt64,Int64,Float32,Float64}

@inline Random.rand!(
    mutable_rng::StatefulRNG,
    destination::Array{T},
) where {T<:_StatefulUniform} =
    _commit_bridge!(mutable_rng, rand_next!(mutable_rng.rng, destination; threaded = false))

@inline Random.rand!(mutable_rng::StatefulRNG, destination::BitArray) =
    _commit_bridge!(mutable_rng, rand_next!(mutable_rng.rng, destination; threaded = false))

@inline Random.randn!(
    mutable_rng::StatefulRNG,
    destination::Array{T},
) where {T<:Union{Float32,Float64}} = _commit_bridge!(
    mutable_rng,
    randn_next!(mutable_rng.rng, destination; threaded = false),
)

@inline Random.randexp!(
    mutable_rng::StatefulRNG,
    destination::Array{T},
) where {T<:Union{Float32,Float64}} = _commit_bridge!(
    mutable_rng,
    randexp_next!(mutable_rng.rng, destination; threaded = false),
)

for F in _FAMILY_SYMBOLS
    @eval @inline _fresh_bridge_rng(::$F, seed::Integer) = $F(seed)
end

@inline function Random.seed!(mutable_rng::StatefulRNG, seed::Integer)
    fresh = _fresh_bridge_rng(mutable_rng.rng, seed)
    mutable_rng.rng = fresh
    return mutable_rng
end

@inline Base.copy(mutable_rng::StatefulRNG) = _stateful_rng(mutable_rng.rng)
