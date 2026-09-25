# Gamma draws by Marsaglia and Tsang (2000) with a fixed stream span. A draw
# reserves one boost uniform and K candidates, each a normal and an open uniform
# on the normal's lattice: (2K + 1) n bits for n normal bits, whatever the shape.
# The first accepted candidate is the value; if all K reject, the draw continues
# the same test on a child stream keyed by its position, so the law stays exactly
# Gamma and the parent stream still advances by the fixed span. Shapes below one
# draw Gamma(shape + 1) and multiply by u^(1/shape), with the power taken in log
# space. The boost comes first, so it shares its block with the first candidate.

const _GAMMA_CANDIDATES = 8
const _GammaDistribution = Distributions.Gamma{<:_FloatType}

@inline _gamma_span(::Type{T}, candidates::Int) where {T} =
    UInt16((2candidates + 1) * IR._normal_bits(T))

# The Marsaglia-Tsang constants depend only on the shape, so the codec holds
# them. `candidates` is `_GAMMA_CANDIDATES` in every public draw.
struct _GammaCodec{T,B<:IR._BackendToken}
    shape::T
    scale::T
    d::T
    c::T
    device::B
    candidates::Int
end

function _GammaCodec(shape::T, scale::T, device, candidates::Int) where {T}
    d = (shape < one(T) ? shape + one(T) : shape) - one(T) / T(3)
    return _GammaCodec(shape, scale, d, inv(sqrt(T(9) * d)), device, candidates)
end
_GammaCodec(d::Distributions.Gamma{T}, device) where {T} =
    _GammaCodec(d.α, d.θ, device, _GAMMA_CANDIDATES)

@inline function _gamma_candidate(x::T, u::T, d::T, c::T) where {T}
    v = one(T) + c * x
    v <= zero(T) && return false, zero(T)
    v = v * v * v
    squared = x * x
    # The squeeze accepts inside the exact region and skips the logarithm.
    accepted =
        u < one(T) - T(0.0331) * squared * squared ||
        log(u) < squared / 2 + d * (one(T) - v + log(v))
    return accepted, d * v
end

const _GAMMA_TAG = 0x67616d6d61636869

@inline _position_index(rng, p::IR._Position64) = (p.block << IR._block_shift(rng)) + p.bit
@inline _position_index(rng, p::IR._Position128) = (p.lo << IR._block_shift(rng)) + p.bit

@noinline function _gamma_child(codec::_GammaCodec, rng, position, d::T, c::T) where {T}
    child = IR.subrng(rng, xor(_position_index(rng, position), _GAMMA_TAG))
    n = Int(IR._normal_bits(T))
    while true
        x, child = IR.randn_next(child, T)
        raw, child = IR.rand_next(child, UInt64)
        accepted, g = _gamma_candidate(x, IR._open_midpoint(T, raw >> (64 - n)), d, c)
        accepted && return g
    end
end

@inline function _gamma_standard(
    codec::_GammaCodec,
    rng,
    position,
    cursor,
    ::Type{T},
) where {T}
    n = Int(IR._normal_bits(T))
    d, c = codec.d, codec.c
    boost_raw, cursor = IR._take_dense_bits_unchecked(rng, cursor, Val(n))
    accepted, g = false, zero(T)
    for _ = 1:codec.candidates
        normal_raw, cursor = IR._take_dense_bits_unchecked(rng, cursor, Val(n))
        uniform_raw, cursor = IR._take_dense_bits_unchecked(rng, cursor, Val(n))
        x = IR._normal_from_bits(codec.device, T, normal_raw)
        accepted, g = _gamma_candidate(x, IR._open_midpoint(T, uniform_raw), d, c)
        accepted && break
    end
    accepted || (g = _gamma_child(codec, rng, position, d, c))
    codec.shape < one(T) || return g
    return g * exp(log(IR._open_midpoint(T, boost_raw)) / codec.shape)
end

@inline IR._fill_width(codec::_GammaCodec, ::Type{T}) where {T} =
    _gamma_span(T, codec.candidates)

@inline function IR._transformed_draw_unchecked(
    codec::_GammaCodec,
    rng,
    position,
    ::Type{T},
) where {T}
    cursor = IR._dense_cursor(rng, IR._position_block(position), position.bit)
    return codec.scale * _gamma_standard(codec, rng, position, cursor, T)
end

@inline _block_start(block::UInt64) = IR._Position64(block, UInt16(0))
@inline _block_start(block::Tuple{UInt64,UInt64}) =
    IR._Position128(block[1], block[2], UInt16(0))

# The cursor already sits at the draw, so the draw reads from it and the next
# draw restarts a cursor past the span, skipping the candidates left unread.
@inline function IR._codec_take(codec::_GammaCodec, rng, cursor, ::Type{T}) where {T}
    shift = IR._block_shift(rng)
    offset = UInt64(cursor.lane) * UInt64(64) + UInt64(cursor.bit)
    position =
        IR._advance_position_unchecked(_block_start(cursor.block), offset, UInt64(0), shift)
    value = codec.scale * _gamma_standard(codec, rng, position, cursor, T)
    span = UInt64(IR._fill_width(codec, T))
    next = IR._advance_position_unchecked(position, span, UInt64(0), shift)
    return value, IR._dense_cursor(rng, IR._position_block(next), next.bit)
end

@noinline _invalid_parameters(::Distributions.Gamma) =
    throw(ArgumentError("invalid Gamma parameters"))

@inline function _validate_distribution(d::_GammaDistribution)
    isfinite(d.α) && d.α > zero(d.α) || _invalid_parameters(d)
    isfinite(d.θ) && d.θ > zero(d.θ) || _invalid_parameters(d)
    return nothing
end

@inline _result_type(::Distributions.Gamma{T}) where {T} = T
@inline _distribution_span(::Distributions.Gamma{T}) where {T<:_FloatType} =
    _gamma_span(T, _GAMMA_CANDIDATES)

@inline function _draw_distribution_unchecked(rng, position, d::_GammaDistribution)
    codec = _GammaCodec(d, rng.device)
    return IR._transformed_draw_unchecked(codec, rng, position, _result_type(d))
end

@inline _fill_distribution_prevalidated!(
    rng,
    d::_GammaDistribution,
    destination,
    threaded,
) = IR._fill_prevalidated!(rng, destination, threaded, _GammaCodec(d, rng.device))
