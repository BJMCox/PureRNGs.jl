# Gamma draws by Marsaglia and Tsang (2000) with a fixed stream span. A draw
# reserves one boost uniform and K candidates, each a normal and an open uniform
# on the normal's lattice: (2K + 1) n bits for n normal bits, whatever the shape.
# The first accepted candidate is the value; if all K reject, the draw continues
# the same test on a child stream keyed by its position, so the law stays exactly
# Gamma and the parent stream still advances by the fixed span. Shapes below one
# draw Gamma(shape + 1) and multiply by u^(1/shape), with the power taken in log
# space. The boost comes first, so it shares its block with the first candidate.

const _GAMMA_CANDIDATES = 8

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

# The first accepted candidate's Gamma(shape) value, or Gamma(shape + 1) for a
# shape below one, and the boost uniform's raw bits.
@inline function _gamma_base(codec::_GammaCodec, rng, position, cursor, ::Type{T}) where {T}
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
    return g, boost_raw
end

@inline function _gamma_standard(
    codec::_GammaCodec,
    rng,
    position,
    cursor,
    ::Type{T},
) where {T}
    g, boost_raw = _gamma_base(codec, rng, position, cursor, T)
    codec.shape < one(T) || return g
    return g * exp(log(IR._open_midpoint(T, boost_raw)) / codec.shape)
end

# The logarithm stays finite where a small shape's value underflows, which
# Beta and Dirichlet need to normalize their draws.
@inline function _gamma_log_standard(
    codec::_GammaCodec,
    rng,
    position,
    cursor,
    ::Type{T},
) where {T}
    g, boost_raw = _gamma_base(codec, rng, position, cursor, T)
    codec.shape < one(T) || return log(g)
    return log(g) + log(IR._open_midpoint(T, boost_raw)) / codec.shape
end

@inline _gamma_cursor(rng, position) =
    IR._dense_cursor(rng, IR._position_block(position), position.bit)
@inline _gamma_offset(rng, position, bits) =
    IR._advance_position_unchecked(position, UInt64(bits), UInt64(0), IR._block_shift(rng))

# InverseGamma(shape, scale) is scale over a standard Gamma(shape) draw.
struct _InverseGammaCodec{G<:_GammaCodec}
    gamma::G
end

# Beta(a, b) is X / (X + Y) for X ~ Gamma(a) and Y ~ Gamma(b) in consecutive
# spans, taken from their logarithms so that small shapes never give 0 / 0.
struct _BetaCodec{G<:_GammaCodec}
    a::G
    b::G
end

# TDist(nu) is a normal over sqrt(chi2 / nu), with chi2 = 2 Gamma(nu / 2) in
# the span after the normal.
struct _TDistCodec{G<:_GammaCodec,T}
    gamma::G
    ν::T
end

const _GammaFamilyCodec = Union{_GammaCodec,_InverseGammaCodec,_BetaCodec,_TDistCodec}

@inline IR._fill_width(codec::_GammaCodec, ::Type{T}) where {T} =
    _gamma_span(T, codec.candidates)
@inline IR._fill_width(codec::_InverseGammaCodec, ::Type{T}) where {T} =
    IR._fill_width(codec.gamma, T)
@inline IR._fill_width(codec::_BetaCodec, ::Type{T}) where {T} =
    UInt16(2 * IR._fill_width(codec.a, T))
@inline IR._fill_width(codec::_TDistCodec, ::Type{T}) where {T} =
    UInt16(IR._normal_bits(T) + IR._fill_width(codec.gamma, T))

@inline _family_value(codec::_GammaCodec, rng, position, cursor, ::Type{T}) where {T} =
    codec.scale * _gamma_standard(codec, rng, position, cursor, T)
@inline _family_value(
    codec::_InverseGammaCodec,
    rng,
    position,
    cursor,
    ::Type{T},
) where {T} = codec.gamma.scale / _gamma_standard(codec.gamma, rng, position, cursor, T)
@inline function _family_value(
    codec::_BetaCodec,
    rng,
    position,
    cursor,
    ::Type{T},
) where {T}
    log_x = _gamma_log_standard(codec.a, rng, position, cursor, T)
    second = _gamma_offset(rng, position, IR._fill_width(codec.a, T))
    log_y = _gamma_log_standard(codec.b, rng, second, _gamma_cursor(rng, second), T)
    return inv(one(T) + exp(log_y - log_x))
end
@inline function _family_value(
    codec::_TDistCodec,
    rng,
    position,
    cursor,
    ::Type{T},
) where {T}
    n = Int(IR._normal_bits(T))
    normal_raw, cursor = IR._take_dense_bits_unchecked(rng, cursor, Val(n))
    z = IR._normal_from_bits(codec.gamma.device, T, normal_raw)
    second = _gamma_offset(rng, position, n)
    g = _gamma_standard(codec.gamma, rng, second, cursor, T)
    return z * sqrt(codec.ν / (T(2) * g))
end

@inline IR._transformed_draw_unchecked(
    codec::_GammaFamilyCodec,
    rng,
    position,
    ::Type{T},
) where {T} = _family_value(codec, rng, position, _gamma_cursor(rng, position), T)

@inline _block_start(block::UInt64) = IR._Position64(block, UInt16(0))
@inline _block_start(block::Tuple{UInt64,UInt64}) =
    IR._Position128(block[1], block[2], UInt16(0))

# The cursor already sits at the draw, so the draw reads from it and the next
# draw restarts a cursor past the span, skipping the candidates left unread.
@inline function IR._codec_take(codec::_GammaFamilyCodec, rng, cursor, ::Type{T}) where {T}
    offset = UInt64(cursor.lane) * UInt64(64) + UInt64(cursor.bit)
    position = _gamma_offset(rng, _block_start(cursor.block), offset)
    value = _family_value(codec, rng, position, cursor, T)
    next = _gamma_offset(rng, position, IR._fill_width(codec, T))
    return value, _gamma_cursor(rng, next)
end

const _GammaFamily{T} = Union{
    Distributions.Gamma{T},
    Distributions.Chisq{T},
    Distributions.InverseGamma{T},
    Distributions.Beta{T},
    Distributions.TDist{T},
}
const _GammaFamilyDistribution = _GammaFamily{<:_FloatType}

_family_codec(d::Distributions.Gamma{T}, device) where {T} =
    _GammaCodec(d.α, d.θ, device, _GAMMA_CANDIDATES)
_family_codec(d::Distributions.Chisq{T}, device) where {T} =
    _GammaCodec(d.ν / 2, T(2), device, _GAMMA_CANDIDATES)
_family_codec(d::Distributions.InverseGamma{T}, device) where {T} =
    _InverseGammaCodec(_GammaCodec(d.invd.α, d.θ, device, _GAMMA_CANDIDATES))
_family_codec(d::Distributions.Beta{T}, device) where {T} = _BetaCodec(
    _GammaCodec(d.α, one(T), device, _GAMMA_CANDIDATES),
    _GammaCodec(d.β, one(T), device, _GAMMA_CANDIDATES),
)
_family_codec(d::Distributions.TDist{T}, device) where {T} =
    _TDistCodec(_GammaCodec(d.ν / 2, one(T), device, _GAMMA_CANDIDATES), d.ν)

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

# A Dirichlet draw normalizes length(alpha) log-gamma draws in consecutive spans
# by log-sum-exp, so small shapes still sum to one. The vector result runs on the
# CPU, as MvNormal draws do.
const _FloatDirichlet = Distributions.Dirichlet{<:_FloatType}

@inline _dirichlet_spans(d, draws) = UInt128(draws) * UInt128(length(d.alpha))

function _validate_dirichlet(d)
    all(_positive_finite, d.alpha) || throw(ArgumentError("invalid Dirichlet parameters"))
    return nothing
end

# Draw `ordinal` (one-based) past `rng`'s position into `destination`.
function _dirichlet_draw!(destination, rng, d, ordinal::Integer)
    T = eltype(destination)
    span = _gamma_span(T, _GAMMA_CANDIDATES)
    first_span = UInt64(ordinal - 1) * UInt64(length(d.alpha))
    for (component, shape) in enumerate(d.alpha)
        codec = _GammaCodec(shape, one(T), rng.device, _GAMMA_CANDIDATES)
        bits_lo, bits_hi = IR._bit_span(first_span + UInt64(component - 1), span)
        position = IR._advance_position_unchecked(rng, bits_lo, bits_hi)
        destination[component] =
            _gamma_log_standard(codec, rng, position, _gamma_cursor(rng, position), T)
    end
    largest = maximum(destination)
    destination .= exp.(destination .- largest)
    destination ./= sum(destination)
    return destination
end

function _reserve_dirichlet(rng, d, draws, ::Type{T}) where {T}
    bits = _dirichlet_spans(d, draws) * UInt128(_gamma_span(T, _GAMMA_CANDIDATES))
    return IR._reserve(rng, bits % UInt64, (bits >> 64) % UInt64)
end

function IR.rand_next!(
    rng::IR._CPUGenerators,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType}
    _validate_dirichlet(d)
    size(destination, 1) == length(d) || throw(
        DimensionMismatch(
            "destination has $(size(destination, 1)) rows for a $(length(d))-component Dirichlet",
        ),
    )
    draws = size(destination, 2)
    next_rng = _reserve_dirichlet(rng, d, draws, T)
    columns = eachcol(reshape(destination, length(d), draws))
    if threaded
        Threads.@threads for ordinal = 1:draws
            _dirichlet_draw!(columns[ordinal], rng, d, ordinal)
        end
    else
        for ordinal = 1:draws
            _dirichlet_draw!(columns[ordinal], rng, d, ordinal)
        end
    end
    return destination, next_rng
end
Random.rand!(
    rng::IR._CPUGenerators,
    d::Distributions.Dirichlet{T},
    destination::AbstractVecOrMat{T};
    threaded::Bool = false,
) where {T<:_FloatType} = first(IR.rand_next!(rng, d, destination; threaded))

IR.rand_next(rng::IR._CPUGenerators, d::_FloatDirichlet) =
    IR.rand_next!(rng, d, Vector{Distributions.partype(d)}(undef, length(d)))
Random.rand(rng::IR._CPUGenerators, d::_FloatDirichlet) = first(IR.rand_next(rng, d))
IR.rand_next(
    rng::IR._CPUGenerators,
    d::_FloatDirichlet,
    n::Integer;
    threaded::Bool = false,
) = IR.rand_next!(rng, d, Matrix{Distributions.partype(d)}(undef, length(d), n); threaded)
Random.rand(
    rng::IR._CPUGenerators,
    d::_FloatDirichlet,
    n::Integer;
    threaded::Bool = false,
) = first(IR.rand_next(rng, d, n; threaded))

function IR.rand_at(rng::IR._CPUGenerators, d::_FloatDirichlet, index::Integer)
    _validate_dirichlet(d)
    index < 1 && IR._invalid_address_index()
    T = Distributions.partype(d)
    span = _gamma_span(T, _GAMMA_CANDIDATES)
    addressed = IR._addressed_rng(rng, span, (index - 1) * length(d) + 1)
    _reserve_dirichlet(addressed, d, 1, T)
    return _dirichlet_draw!(Vector{T}(undef, length(d)), addressed, d, 1)
end
