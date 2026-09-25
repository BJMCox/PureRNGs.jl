using DifferentiationInterface
using Distributions
using ForwardDiff
using Mooncake
using PureRNGs
using Random
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
