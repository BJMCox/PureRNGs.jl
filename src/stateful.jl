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

# Examples

```jldoctest
julia> using Random

julia> bridge = StatefulRNG(Philox4x32(20250918));

julia> rand(bridge, UInt32)
0x23b42aea

julia> rand(bridge, UInt32)
0x467098dd

julia> rand(parent(bridge), UInt32)
0xc25ecc0b
```
"""
mutable struct StatefulRNG{R<:AbstractPureRNG} <: Random.AbstractRNG
    rng::R

    StatefulRNG{R}(::_ConstructionToken, rng::R) where {R} = new{R}(rng)
end

@inline _stateful_rng(rng::R) where {R<:AbstractPureRNG} =
    StatefulRNG{R}(_CONSTRUCTION_TOKEN, rng)

@inline StatefulRNG(rng::AbstractPureRNG) = _stateful_rng(MLDataDevices.CPUDevice()(rng))

@inline Base.parent(mutable_rng::StatefulRNG) = mutable_rng.rng

# Reading the held generator whole copies it through a stack blob that the
# next draw's stores then feed back through memory. Reading it word by word
# keeps the scalar bridge path in registers between the load and the store.
@inline function _held(mutable_rng::StatefulRNG)
    rng = mutable_rng.rng
    return typeof(rng)(
        _CONSTRUCTION_TOKEN,
        _by_element(rng.key),
        rng.position,
        rng.device,
        _by_element(rng.block_words),
    )
end

@inline function _commit_bridge!(
    mutable_rng::StatefulRNG{R},
    result::Tuple{T,R},
) where {R,T}
    mutable_rng.rng = last(result)
    return first(result)
end

@inline Random.rand(
    mutable_rng::StatefulRNG,
    ::Random.SamplerType{T},
) where {T<:Union{Bool,_UniformInteger}} =
    _commit_bridge!(mutable_rng, rand_next(_held(mutable_rng), T))

# Random's own float methods name Float32 and Float64 concretely in the second
# slot, so a `T<:_UniformFloat` bound here would be ambiguous with them.
@inline Random.rand(
    mutable_rng::StatefulRNG,
    ::Random.SamplerTrivial{Random.CloseOpen01{Float32}},
) = _commit_bridge!(mutable_rng, rand_next(_held(mutable_rng), Float32))
@inline Random.rand(
    mutable_rng::StatefulRNG,
    ::Random.SamplerTrivial{Random.CloseOpen01{Float64}},
) = _commit_bridge!(mutable_rng, rand_next(_held(mutable_rng), Float64))
@inline Random.randn(mutable_rng::StatefulRNG, ::Type{Float32}) =
    _commit_bridge!(mutable_rng, randn_next(_held(mutable_rng), Float32))
@inline Random.randn(mutable_rng::StatefulRNG, ::Type{Float64}) =
    _commit_bridge!(mutable_rng, randn_next(_held(mutable_rng), Float64))
@inline Random.randexp(mutable_rng::StatefulRNG, ::Type{Float32}) =
    _commit_bridge!(mutable_rng, randexp_next(_held(mutable_rng), Float32))
@inline Random.randexp(mutable_rng::StatefulRNG, ::Type{Float64}) =
    _commit_bridge!(mutable_rng, randexp_next(_held(mutable_rng), Float64))

@inline Random.randn(mutable_rng::StatefulRNG) = Random.randn(mutable_rng, Float64)
@inline Random.randexp(mutable_rng::StatefulRNG) = Random.randexp(mutable_rng, Float64)

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
    _commit_bridge!(mutable_rng, rand_next(_held(mutable_rng), sampler.range))

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
    # Only the prefix the stream still covers is filled, so the bridge enters the
    # CPU body directly instead of going through a whole-destination fill.
    if !iszero(fitting_count)
        _fill_transformed_cpu!(
            mutable_rng.rng,
            mutable_rng.rng.position,
            destination,
            T,
            1:Int(fitting_count),
            _RangeCodec(range, span),
        )
        mutable_rng.rng = _rebuild(mutable_rng.rng, position, mutable_rng.rng.device)
    end
    complete && return destination
    rand_next(mutable_rng.rng, range)
    return destination
end

@inline Random.rand!(
    mutable_rng::StatefulRNG,
    destination::Array{T},
) where {T<:_UniformResult} =
    _commit_bridge!(mutable_rng, rand_next!(mutable_rng.rng, destination; threaded = false))

# Without this hook `rand(m, T, n)` falls to Random's scalar loop instead of the
# package fill.
@inline Random.rand!(
    mutable_rng::StatefulRNG,
    destination::Array{T},
    ::Random.SamplerTrivial{Random.CloseOpen01{T}},
) where {T<:Union{Float32,Float64}} =
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

for F in _GENERATOR_SYMBOLS
    @eval @inline _fresh_bridge_rng(::$F{_CPUBackend,R}, seed::Integer) where {R} =
        $F{_CPUBackend,R}(
            _CONSTRUCTION_TOKEN,
            _family_key($F, seed),
            _zero_position($F),
            _CPU_BACKEND,
        )
end

@inline function Random.seed!(mutable_rng::StatefulRNG, seed::Integer)
    fresh = _fresh_bridge_rng(mutable_rng.rng, seed)
    mutable_rng.rng = fresh
    return mutable_rng
end

@inline Base.copy(mutable_rng::StatefulRNG) = _stateful_rng(mutable_rng.rng)
