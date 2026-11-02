module PureRNGsMooncakeExt

# A draw's base variate depends only on stream bits. Mooncake cannot follow the
# bit casts that decode it, and need not: the distribution transform that
# consumes it carries every parameter derivative.

import PureRNGs
using Mooncake: @zero_derivative, @is_primitive, DefaultCtx
using Mooncake: CoDual, Dual, NoRData, primal, tangent, zero_fcodual, zero_rdata
import Mooncake: frule!!, rrule!!

const IR = PureRNGs

@zero_derivative DefaultCtx Tuple{typeof(IR._from_bits),Type,Any}
@zero_derivative DefaultCtx Tuple{typeof(IR._open_midpoint),Type,Any}
@zero_derivative DefaultCtx Tuple{typeof(IR._normal_from_bits),Any,Type,Any}
@zero_derivative DefaultCtx Tuple{typeof(IR._exponential_from_bits),Any,Type,Any}
@zero_derivative DefaultCtx Tuple{typeof(IR._exponential_transform),Any,Type,Any}
# Range reduction chooses integer exponents; only the scaled arithmetic varies.
@zero_derivative DefaultCtx Tuple{typeof(IR._gamma_scale_shift),Any}

@is_primitive DefaultCtx Tuple{typeof(IR._gamma_ldexp),F,Int} where {F<:Base.IEEEFloat}
function frule!!(
    ::Dual{typeof(IR._gamma_ldexp)},
    value::Dual{F},
    power::Dual{Int},
) where {F<:Base.IEEEFloat}
    n = primal(power)
    return Dual(ldexp(primal(value), n), ldexp(tangent(value), n))
end
function rrule!!(
    ::CoDual{typeof(IR._gamma_ldexp)},
    value::CoDual{F},
    power::CoDual{Int},
) where {F<:Base.IEEEFloat}
    n = primal(power)
    adjoint(dvalue) = (NoRData(), ldexp(dvalue, n), NoRData())
    return zero_fcodual(ldexp(primal(value), n)), adjoint
end

# The rejection test picks which candidate a Gamma draw takes, so
# differentiating through it would bias the shape gradient. The draw's shape
# derivative is the implicit one of the Gamma CDF instead.
for (draw, slope) in (
    (IR._gamma_value, IR._gamma_shape_derivative),
    (IR._gamma_log_value, IR._gamma_log_shape_derivative),
)
    @eval begin
        @is_primitive DefaultCtx Tuple{
            typeof($draw),
            F,
            IR._GammaCodec{F},
            Any,
            Any,
            Any,
        } where {F<:Base.IEEEFloat}
        function frule!!(
            ::Dual{typeof($draw)},
            shape::Dual{F},
            codec::Dual,
            rng::Dual,
            ordinal::Dual,
            cursor::Dual,
        ) where {F<:Base.IEEEFloat}
            s = primal(shape)
            value = $draw(s, primal(codec), primal(rng), primal(ordinal), primal(cursor))
            return Dual(value, IR._scale_tangent(tangent(shape), $slope(s, value)))
        end
        function rrule!!(
            ::CoDual{typeof($draw)},
            shape::CoDual{F},
            codec::CoDual,
            rng::CoDual,
            ordinal::CoDual,
            cursor::CoDual,
        ) where {F<:Base.IEEEFloat}
            s = primal(shape)
            value = $draw(s, primal(codec), primal(rng), primal(ordinal), primal(cursor))
            derivative = $slope(s, value)
            adjoint(dvalue) = (
                NoRData(),
                IR._scale_tangent(dvalue, derivative),
                zero_rdata(primal(codec)),
                zero_rdata(primal(rng)),
                zero_rdata(primal(ordinal)),
                zero_rdata(primal(cursor)),
            )
            return zero_fcodual(value), adjoint
        end
    end
end

end
