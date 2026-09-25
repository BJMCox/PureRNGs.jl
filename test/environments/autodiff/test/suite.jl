using DifferentiationInterface
using Distributions
using ForwardDiff
using Mooncake
using PureRNGs
using Random
using Statistics
using Test

# Each family's parameters as functions of x, so one input vector drives both a
# location and a scale where the family has them.
const FAMILIES = (
    x -> Normal(x[1], x[2]),
    x -> Uniform(x[1], x[1] + x[2]),
    x -> Exponential(x[2]),
    x -> LogNormal(x[1], x[2]),
    x -> Weibull(x[2], x[2]),
    x -> Rayleigh(x[2]),
    x -> Laplace(x[1], x[2]),
    x -> Logistic(x[1], x[2]),
    x -> Gumbel(x[1], x[2]),
    x -> Pareto(x[2], x[2]),
    x -> Frechet(x[2], x[2]),
    x -> Cauchy(x[1], x[2]),
    x -> TriangularDist(x[1], x[1] + 2x[2], x[1] + x[2]),
)

central_difference(f, x; h = 1e-6) = [
    (f(x .+ h .* (eachindex(x) .== i)) - f(x .- h .* (eachindex(x) .== i))) / 2h for
    i in eachindex(x)
]

# A draw at a fixed position is a smooth function of the parameters, so every
# backend's gradient is the pathwise derivative a central difference estimates.
@testset "gradients of fixed draws are pathwise derivatives" begin
    rng = Philox4x32(0x6a1, 3)
    x = [0.5, 2.0]
    for family in FAMILIES, backend in (AutoMooncake(), AutoForwardDiff())
        scalar = x -> rand(rng, family(x))
        batch = x -> sum(rand(rng, family(x), 3))
        @test gradient(scalar, backend, x) ≈ central_difference(scalar, x) rtol = 1e-5
        @test gradient(batch, backend, x) ≈ central_difference(batch, x) rtol = 1e-5
    end
end

# A dual draw decodes the primal draw's base variates, so its value is the
# primal draw at every position and in every form.
@testset "a dual draw's value is the primal draw" begin
    rng = Philox4x32(0x6a2)
    dual = ForwardDiff.Dual(0.5, 1.0)
    for family in FAMILIES
        d = family([dual, 2.0])
        primal = family([0.5, 2.0])
        value, next_rng = rand_next(rng, d)
        expected, expected_next = rand_next(rng, primal)
        @test ForwardDiff.value(value) == expected
        @test next_rng == expected_next
        @test ForwardDiff.value(rand_at(rng, d, 4)) == rand_at(rng, primal, 4)
        @test ForwardDiff.value.(rand(rng, d, 2, 3; threaded = true)) ==
              rand(rng, primal, 2, 3)
    end
end

@testset "MvNormal gradients are pathwise derivatives" begin
    rng = Philox4x32(0x6a3)
    covariance(p) = [p[4]^2 0.3 0.1; 0.3 p[5]^2 0.2; 0.1 0.2 p[6]^2]
    draws = p -> sum(rand(rng, MvNormal(p[1:3], covariance(p)), 4))
    p = [1.0, -2.0, 0.5, 1.5, 0.7, 2.0]
    for backend in (AutoMooncake(), AutoForwardDiff())
        @test gradient(draws, backend, p) ≈ central_difference(draws, p) rtol = 1e-5
    end
end

# A Gamma draw moves with its shape along the inverse CDF at the draw's
# probability, dX/dshape = -dF/dshape / f, whatever candidate the rejection test
# took. Family members compose that derivative through their maps.
gamma_oracle(shape, x; h = 1e-6) = begin
    u = cdf(Gamma(shape), x)
    (quantile(Gamma(shape + h), u) - quantile(Gamma(shape - h), u)) / 2h
end
const GAMMA_SPAN = 17 * 52
advanced(rng, bits) = PureRNGs._rebuild(
    rng,
    PureRNGs._advance_position_unchecked(rng, UInt64(bits), UInt64(0)),
    rng.device,
)

@testset "Gamma family gradients follow the implicit shape derivative" begin
    rng = Philox4x32(0xb01, 3)
    x = rand(rng, Gamma(0.3))
    y = rand(advanced(rng, GAMMA_SPAN), Gamma(0.4))
    z = randn(rng, Float64)
    g = rand(advanced(rng, 52), Gamma(1.5))
    oracles = (
        (q -> Gamma(q, 2.0), 2.5, 2 * gamma_oracle(2.5, rand(rng, Gamma(2.5)))),
        (q -> Chisq(q), 3.0, gamma_oracle(1.5, rand(rng, Gamma(1.5)))),
        (
            q -> InverseGamma(q, 1.5),
            2.5,
            -1.5 * gamma_oracle(2.5, rand(rng, Gamma(2.5))) / rand(rng, Gamma(2.5))^2,
        ),
        (q -> Beta(q, 0.4), 0.3, gamma_oracle(0.3, x) * y / (x + y)^2),
        (
            q -> TDist(q),
            3.0,
            z * (1 / (2g) - 3 / (2g^2) * gamma_oracle(1.5, g) / 2) / (2sqrt(3 / (2g))),
        ),
    )
    for (family, p, expected) in oracles, backend in (AutoMooncake(), AutoForwardDiff())
        @test derivative(q -> rand(rng, family(q)), backend, p) ≈ expected rtol = 1e-6
    end
    shapes = [0.3, 1.0, 2.5]
    xs = [rand(advanced(rng, (j - 1) * GAMMA_SPAN), Gamma(shapes[j])) for j = 1:3]
    total = sum(xs)
    @test derivative(q -> rand(rng, Dirichlet([q, 1.0, 2.5]))[1], AutoMooncake(), 0.3) ≈
          gamma_oracle(0.3, xs[1]) * (total - xs[1]) / total^2 rtol = 1e-6
end

# The implicit derivative is unbiased: E[dX/dshape] = dE[X]/dshape = 1 for
# Gamma(shape, 1), down to shapes whose draws underflow.
@testset "Gamma shape gradients are unbiased" begin
    rng = Philox4x32(0xb77, 11)
    for shape in (0.02, 0.3, 2.5)
        slopes = ForwardDiff.derivative(q -> rand(rng, Gamma(q, 1.0), 200_000), shape)
        @test abs(mean(slopes) - 1) < 4 * std(slopes) / sqrt(length(slopes))
    end
end
