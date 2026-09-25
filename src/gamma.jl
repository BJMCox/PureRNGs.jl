# Gamma draws by Marsaglia and Tsang (2000) with a fixed stream span. A draw
# reserves one boost uniform and K candidates, each a normal and an open uniform
# on the normal's lattice: (2K + 1) n bits for n normal bits, whatever the shape.
# The first accepted candidate is the value; if all K reject, the draw continues
# the same test on a child stream keyed by its position, so the law stays exactly
# Gamma and the parent stream still advances by the fixed span. Shapes below one
# draw Gamma(shape + 1) and multiply by u^(1/shape), with the power taken in log
# space. The boost comes first, so it shares its block with the first candidate.
#
# The Distributions extension maps its types onto these codecs. The sampler
# lives here, with no Distributions dependency, so that AD extensions can attach
# the implicit shape derivative to `_gamma_value` and `_gamma_log_value`.

const _GAMMA_CANDIDATES = 8

# The floating-point type a parameter's value decodes in. AD extensions add
# methods for their number types.
@inline _primal_float(::Type{T}) where {T<:AbstractFloat} = T

@inline _gamma_span(::Type{F}, candidates::Int) where {F} =
    UInt16((2candidates + 1) * _normal_bits(F))

# The Marsaglia-Tsang constants depend only on the shape, so the codec holds
# them. `candidates` is `_GAMMA_CANDIDATES` in every public draw.
struct _GammaCodec{T,B<:_BackendToken}
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

@inline _codec_float(::_GammaCodec{T}) where {T} = _primal_float(T)

@inline function _gamma_candidate(x::F, u::F, d::F, c::F) where {F}
    v = one(F) + c * x
    v <= zero(F) && return false, zero(F)
    v = v * v * v
    squared = x * x
    # The squeeze accepts inside the exact region and skips the logarithm.
    accepted =
        u < one(F) - F(0.0331) * squared * squared ||
        log(u) < squared / 2 + d * (one(F) - v + log(v))
    return accepted, d * v
end

const _GAMMA_TAG = 0x67616d6d61636869

@inline _position_index(rng, p::_Position64) = (p.block << _block_shift(rng)) + p.bit
@inline _position_index(rng, p::_Position128) = (p.lo << _block_shift(rng)) + p.bit

@noinline function _gamma_child(codec::_GammaCodec{F}, rng, position) where {F}
    child = subrng(rng, xor(_position_index(rng, position), _GAMMA_TAG))
    n = Int(_normal_bits(F))
    while true
        x, child = randn_next(child, F)
        raw, child = rand_next(child, UInt64)
        accepted, g =
            _gamma_candidate(x, _open_midpoint(F, raw >> (64 - n)), codec.d, codec.c)
        accepted && return g
    end
end

# The first accepted candidate's Gamma(shape) value, or Gamma(shape + 1) for a
# shape below one, and the boost uniform's raw bits.
@inline function _gamma_base(codec::_GammaCodec{F}, rng, position, cursor) where {F}
    n = Int(_normal_bits(F))
    boost_raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(n))
    accepted, g = false, zero(F)
    for _ = 1:codec.candidates
        normal_raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(n))
        uniform_raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(n))
        x = _normal_from_bits(codec.device, F, normal_raw)
        accepted, g = _gamma_candidate(x, _open_midpoint(F, uniform_raw), codec.d, codec.c)
        accepted && break
    end
    accepted || (g = _gamma_child(codec, rng, position))
    return g, boost_raw
end

# The standard Gamma(shape) draw at `position`, read from `cursor`, and its
# logarithm, which stays finite where a small shape's value underflows. `shape`
# repeats the codec's shape so that an AD rule can give the draw its implicit
# shape derivative instead of differentiating the rejection test.
@inline function _gamma_value(
    shape::F,
    codec::_GammaCodec{F},
    rng,
    position,
    cursor,
) where {F<:AbstractFloat}
    g, boost_raw = _gamma_base(codec, rng, position, cursor)
    shape < one(F) || return g
    return g * exp(log(_open_midpoint(F, boost_raw)) / shape)
end

@inline function _gamma_log_value(
    shape::F,
    codec::_GammaCodec{F},
    rng,
    position,
    cursor,
) where {F<:AbstractFloat}
    g, boost_raw = _gamma_base(codec, rng, position, cursor)
    shape < one(F) || return log(g)
    return log(g) + log(_open_midpoint(F, boost_raw)) / shape
end

# The digamma function for a positive argument: the recurrence moves it to at
# least 7, where the asymptotic series with Bernoulli coefficients holds.
@inline function _digamma(x::F) where {F<:AbstractFloat}
    ψ = zero(F)
    while x < F(7)
        ψ -= inv(x)
        x += one(F)
    end
    t = inv(x)
    ψ += log(x) - t / 2
    t *= t
    coefficients = (1 / 12, -1 / 120, 1 / 252, -1 / 240, 1 / 132, -691 / 32760, 1 / 12)
    return ψ - t * evalpoly(t, map(F, coefficients))
end

# Near g = shape both expansions need about sqrt(74 shape) terms in Float64.
@inline _gamma_expansion_terms(shape::F) where {F} =
    200 + unsafe_trunc(Int, 12 * sqrt(min(shape, F(1e12))))

# d log(g) / d shape of a standard Gamma(shape) draw g is -dP/dshape / (g p(g))
# for the regularized incomplete gamma function P and the Gamma density p, by
# the implicit function theorem (Figurnov, Mohamed, and Mnih 2018). P comes from
# its series where g <= 1 or g < shape, and 1 - P from its continued fraction
# elsewhere, as in Cephes; each is differentiated term by term, as in Eigen's
# igamma derivative. The prefactor g^shape e^-g / Gamma(shape) cancels against
# g p(g), so the result needs only log(g) and digamma, and stays finite where a
# small shape's draw underflows. The arithmetic runs in the shape's type, so it
# also runs in a device kernel.
@inline function _gamma_log_shape_derivative(shape::F, log_g::F) where {F<:AbstractFloat}
    g = exp(log_g)
    g <= one(F) || g < shape || return _gamma_fraction_log_derivative(shape, g, log_g)
    return _gamma_series_log_derivative(shape, g, log_g)
end

@inline _gamma_shape_derivative(shape::F, g::F) where {F<:AbstractFloat} =
    iszero(g) ? zero(F) : g * _gamma_log_shape_derivative(shape, log(g))

# P = g^a e^-g / Gamma(a + 1) S with S = sum_k g^k / prod_{j <= k} (a + j).
@inline function _gamma_series_log_derivative(a::F, x::F, log_x::F) where {F}
    term, sum, term_slope, sum_slope = one(F), one(F), zero(F), zero(F)
    denominator = a
    for _ = 1:_gamma_expansion_terms(a)
        denominator += one(F)
        ratio = x / denominator
        term_slope = (term_slope - term / denominator) * ratio
        term *= ratio
        sum += term
        sum_slope += term_slope
        term <= eps(F) * sum && abs(term_slope) <= eps(F) * abs(sum_slope) && break
    end
    return -(sum_slope + sum * (log_x - _digamma(a + one(F)))) / a
end

# 1 - P = g^a e^-g / Gamma(a) f for the Cephes continued fraction f, whose
# convergents p / q follow a three-term recurrence; the slopes follow its
# derivative in a. Both rescale together when the convergents grow large.
@inline function _gamma_fraction_log_derivative(a::F, x::F, log_x::F) where {F}
    y = one(F) - a
    z = x + y + one(F)
    c = zero(F)
    p0, q0, p1, q1 = one(F), x, x + one(F), z * x
    dp0, dq0, dp1, dq1 = zero(F), zero(F), zero(F), -x
    fraction = p1 / q1
    slope = (dp1 - fraction * dq1) / q1
    large = inv(eps(F))
    for _ = 1:_gamma_expansion_terms(a)
        c += one(F)
        y += one(F)
        z += F(2)
        yc = y * c
        p = p1 * z - p0 * yc
        q = q1 * z - q0 * yc
        dp = dp1 * z - p1 - dp0 * yc + p0 * c
        dq = dq1 * z - q1 - dq0 * yc + q0 * c
        if !iszero(q)
            next_fraction = p / q
            next_slope = (dp - next_fraction * dq) / q
            converged =
                abs(next_fraction - fraction) <= eps(F) * abs(next_fraction) &&
                abs(next_slope - slope) <= eps(F) * abs(next_slope)
            fraction, slope = next_fraction, next_slope
            converged && break
        end
        p0, p1, q0, q1 = p1, p, q1, q
        dp0, dp1, dq0, dq1 = dp1, dp, dq1, dq
        if abs(p) > large
            p0, p1, q0, q1 = p0 / large, p1 / large, q0 / large, q1 / large
            dp0, dp1, dq0, dq1 = dp0 / large, dp1 / large, dq0 / large, dq1 / large
        end
    end
    return slope + fraction * (log_x - _digamma(a))
end

@inline _gamma_cursor(rng, position) =
    _dense_cursor(rng, _position_block(position), position.bit)
@inline _gamma_offset(rng, position, bits) =
    _advance_position_unchecked(position, UInt64(bits), UInt64(0), _block_shift(rng))

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

@inline _fill_width(codec::_GammaCodec, ::Type) =
    _gamma_span(_codec_float(codec), codec.candidates)
@inline _fill_width(codec::_InverseGammaCodec, ::Type{T}) where {T} =
    _fill_width(codec.gamma, T)
@inline _fill_width(codec::_BetaCodec, ::Type{T}) where {T} =
    UInt16(2 * _fill_width(codec.a, T))
@inline _fill_width(codec::_TDistCodec, ::Type{T}) where {T} =
    UInt16(_normal_bits(_codec_float(codec.gamma)) + _fill_width(codec.gamma, T))

@inline _gamma_draw(codec::_GammaCodec, rng, position, cursor) =
    _gamma_value(codec.shape, codec, rng, position, cursor)
@inline _gamma_log_draw(codec::_GammaCodec, rng, position, cursor) =
    _gamma_log_value(codec.shape, codec, rng, position, cursor)

@inline _family_value(codec::_GammaCodec, rng, position, cursor) =
    codec.scale * _gamma_draw(codec, rng, position, cursor)
@inline _family_value(codec::_InverseGammaCodec, rng, position, cursor) =
    codec.gamma.scale / _gamma_draw(codec.gamma, rng, position, cursor)
@inline function _family_value(codec::_BetaCodec, rng, position, cursor)
    log_x = _gamma_log_draw(codec.a, rng, position, cursor)
    second = _gamma_offset(rng, position, _fill_width(codec.a, Nothing))
    log_y = _gamma_log_draw(codec.b, rng, second, _gamma_cursor(rng, second))
    return inv(one(log_x) + exp(log_y - log_x))
end
@inline function _family_value(codec::_TDistCodec, rng, position, cursor)
    F = _codec_float(codec.gamma)
    n = Int(_normal_bits(F))
    normal_raw, cursor = _take_dense_bits_unchecked(rng, cursor, Val(n))
    z = _normal_from_bits(codec.gamma.device, F, normal_raw)
    g = _gamma_draw(codec.gamma, rng, _gamma_offset(rng, position, n), cursor)
    return z * sqrt(codec.ν / (2 * g))
end

@inline _transformed_draw_unchecked(codec::_GammaFamilyCodec, rng, position, ::Type) =
    _family_value(codec, rng, position, _gamma_cursor(rng, position))

@inline _block_start(block::UInt64) = _Position64(block, UInt16(0))
@inline _block_start(block::Tuple{UInt64,UInt64}) =
    _Position128(block[1], block[2], UInt16(0))

# The cursor already sits at the draw, so the draw reads from it and the next
# draw restarts a cursor past the span, skipping the candidates left unread.
@inline function _codec_take(codec::_GammaFamilyCodec, rng, cursor, ::Type{T}) where {T}
    offset = UInt64(cursor.lane) * UInt64(64) + UInt64(cursor.bit)
    position = _gamma_offset(rng, _block_start(cursor.block), offset)
    value = _family_value(codec, rng, position, cursor)
    next = _gamma_offset(rng, position, _fill_width(codec, T))
    return value, _gamma_cursor(rng, next)
end
