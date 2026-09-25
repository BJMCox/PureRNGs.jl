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
            position::Dual,
            cursor::Dual,
        ) where {F<:Base.IEEEFloat}
            s = primal(shape)
            value = $draw(s, primal(codec), primal(rng), primal(position), primal(cursor))
            return Dual(value, tangent(shape) * $slope(s, value))
        end
        function rrule!!(
            ::CoDual{typeof($draw)},
            shape::CoDual{F},
            codec::CoDual,
            rng::CoDual,
            position::CoDual,
            cursor::CoDual,
        ) where {F<:Base.IEEEFloat}
            s = primal(shape)
            value = $draw(s, primal(codec), primal(rng), primal(position), primal(cursor))
            derivative = $slope(s, value)
            adjoint(dvalue) = (
                NoRData(),
                dvalue * derivative,
                zero_rdata(primal(codec)),
                zero_rdata(primal(rng)),
                zero_rdata(primal(position)),
                zero_rdata(primal(cursor)),
            )
            return zero_fcodual(value), adjoint
        end
    end
end

end
