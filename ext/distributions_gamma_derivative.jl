# The implicit shape derivative of a standard Gamma draw g, dg/dshape =
# -dP(shape, g)/dshape / p(shape, g), for the regularized incomplete gamma
# function P and the Gamma density p (Figurnov, Mohamed, and Mnih 2018). It is
# the derivative of the inverse CDF at the draw's probability, the pathwise
# gradient every AD backend needs from the rejection sampler.

const _SpecialFunctions = Distributions.SpecialFunctions

# A central difference of the incomplete gamma ratio in its shape, on whichever
# of P and Q = 1 - P is smaller, so the difference keeps its relative accuracy
# in both tails; the error is near eps^(2/3).
function _incomplete_gamma_shape_derivative(shape::Float64, x::Float64)
    h = cbrt(eps(Float64)) * shape
    upper = _SpecialFunctions.gamma_inc(shape + h, x)
    lower = _SpecialFunctions.gamma_inc(shape - h, x)
    lower_tail = first(upper) + first(lower) <= 1
    return lower_tail ? (first(upper) - first(lower)) / 2h :
           (last(lower) - last(upper)) / 2h
end

function IR._gamma_shape_derivative(shape::F, g::F) where {F<:AbstractFloat}
    iszero(g) && return zero(F)
    a, x = Float64(shape), Float64(g)
    log_density = (a - 1) * log(x) - x - _SpecialFunctions.loggamma(a)
    return F(-_incomplete_gamma_shape_derivative(a, x) / exp(log_density))
end

# Below 1e-8 the draw may underflow, so the series P = e^-g g^a / Gamma(a + 1) *
# S, S = sum_k g^k / prod_{j <= k} (a + j), gives d log g / d a =
# -((log g - digamma(a + 1)) S + dS/da) / a without forming g.
function IR._gamma_log_shape_derivative(shape::F, log_g::F) where {F<:AbstractFloat}
    a, log_x = Float64(shape), Float64(log_g)
    log_x >= log(1e-8) && return F(IR._gamma_shape_derivative(a, exp(log_x)) / exp(log_x))
    x = exp(log_x)
    sum, shape_sum, term, harmonic = 1.0, 0.0, 1.0, 0.0
    for k = 1:100
        term *= x / (a + k)
        harmonic += 1 / (a + k)
        sum += term
        shape_sum -= term * harmonic
        term < eps(Float64) * sum && break
    end
    return F(-((log_x - _SpecialFunctions.digamma(a + 1)) * sum + shape_sum) / a)
end
