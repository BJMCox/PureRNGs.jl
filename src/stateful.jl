"""
    StatefulRNG(rng)

Wrap an immutable generator for host-only consumers of `Random.AbstractRNG`.
Construction preserves the key and position and rebinds the generator to the CPU.
`parent(bridge)` returns the exact immutable generator currently held by the
bridge without allocating, mutating, or advancing it.

Package-owned one-argument `Array` and `BitArray` fills preflight their full
counter span. The integer-range `Array` fill preserves Julia's scalar bridge
behavior: counter exhaustion may leave its maximal valid prefix written, with
the held generator after that prefix. A foreign `Random` fill has no
chained-scalar consumption-order guarantee and may also leave its destination
partially written. The held generator remains valid at the position after the
last successful draw.
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

@inline function Random.rand!(
    mutable_rng::StatefulRNG,
    destination::Array{T},
    range::AbstractRange{T},
) where {T<:_RangeInteger}
    isempty(destination) && return destination
    span = _range_span(range)
    width = _range_bits(span)
    count = UInt64(length(destination))
    bits_lo, bits_hi = _bit_span(count, width)
    position, complete = _try_advance(mutable_rng.rng, bits_lo, bits_hi)
    fitting_count = count
    if !complete
        fitting_count = UInt64(0)
        failing_count = count
        while fitting_count + UInt64(1) < failing_count
            midpoint = fitting_count + ((failing_count - fitting_count) >> 1)
            bits_lo, bits_hi = _bit_span(midpoint, width)
            _, fits = _try_advance(mutable_rng.rng, bits_lo, bits_hi)
            if fits
                fitting_count = midpoint
            else
                failing_count = midpoint
            end
        end
        if !iszero(fitting_count)
            bits_lo, bits_hi = _bit_span(fitting_count, width)
            position, _ = _try_advance(mutable_rng.rng, bits_lo, bits_hi)
        end
    end
    if !iszero(fitting_count)
        _fill_range_cpu_unchecked!(
            mutable_rng.rng,
            mutable_rng.rng.position,
            destination,
            range,
            span,
            1:Int(fitting_count),
        )
        mutable_rng.rng = _rebuild(mutable_rng.rng, position, mutable_rng.rng.device)
    end
    complete && return destination
    rand_next(mutable_rng.rng, range)
    return destination
end

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
