# In this tail P(a,g) = g^a / Γ(1+a) to working precision. Its implicit
# derivative gives dlog(g)/da = (digamma(1+a) - log(g))/a at the held draw.
function gamma_tail_derivatives()
    rng = Philox4x32(12)
    inverse(a) = first(rand_next(rng, InverseGamma(a, one(a))))
    beta(a) = first(rand_next(rng, Beta(a, one(a))))
    scale(s) = first(rand_next(rng, InverseGamma(0.1f0, s)))
    upper(b) = first(rand_next(rng, Beta(one(b), b)))
    large(a) = first(rand_next(rng, InverseGamma(a, 1.0f25)))
    a = 0.1f0
    g = Float64(first(rand_next(rng, Gamma(a))))
    dlog = (Distributions.digamma(1 + Float64(a)) - log(g)) / a
    v = Float64(beta(a))
    return (
        (inverse, a, -dlog / g),
        (scale, 1.0f0, inv(g)),
        (beta, a, v * (1 - v) * dlog),
        (beta, 0.05f0, 0.0f0),
        (upper, 0.001f0, 0.0f0),
        (large, 1.0f25, -1.0f-25),
    )
end
