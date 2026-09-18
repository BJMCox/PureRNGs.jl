using Random: randexp, randn

# Distributional smoke tests. The null in every case is that the draws are iid
# from the named law. Every threshold sits far into the tail of the null, so a
# fixed seed that fails points at a defect rather than at sampling noise:
# P(sqrt(n) D > 2.0) = 6.7e-4 for the Kolmogorov limit, P(X9 > 27.9) = 1.0e-3,
# P(X5 > 20.5) = 1.0e-3, P(X63 > 103) = 1.1e-3, P(X3 > 16.3) = 1.0e-3,
# P(|Z| > 4.5) = 6.8e-6.
# The suite only catches gross distributional damage. BigCrush and PractRand
# remain the real stream-quality evidence.

const SMOKE_SEEDS = (20260918, 777)
const SMOKE_COUNT = 200_000

# Standard normal deciles from a BigFloat inversion of the erf series, seeded by
# bisection. Written as literals so the test never checks _as241 against itself.
const NORMAL_DECILES = [
    -1.2815515655446004,
    -0.8416212335729142,
    -0.5244005127080408,
    -0.2533471031357998,
    0.0,
    0.2533471031357998,
    0.5244005127080408,
    0.8416212335729142,
    1.2815515655446004,
]

function _ks_deviation(samples, cdf)
    values = sort!(map(cdf, samples))
    count = length(values)
    worst = 0.0
    for (rank, value) in enumerate(values)
        worst = max(worst, max(rank / count - value, value - (rank - 1) / count))
    end
    return sqrt(count) * worst
end

function _chisquare(counts, expected)
    total = 0.0
    for (observed, mean) in zip(counts, expected)
        total += abs2(observed - mean) / mean
    end
    return total
end

function _bin_counts(values, bins, bin_of)
    counts = zeros(Int, bins)
    for value in values
        counts[bin_of(value)] += 1
    end
    return counts
end

_uniform_expected(bins, count) = fill(count / bins, bins)

@testset "uniform draws match the uniform CDF" begin
    for F in GENERATOR_TYPES, seed in SMOKE_SEEDS
        @test _ks_deviation(rand(F(seed), Float64, SMOKE_COUNT), identity) < 2.0
    end
end

@testset "normal draws match the standard normal deciles" begin
    for seed in SMOKE_SEEDS
        draws = randn(Philox4x32(seed), Float64, SMOKE_COUNT)
        counts = _bin_counts(draws, 10, z -> searchsortedfirst(NORMAL_DECILES, z))
        @test _chisquare(counts, _uniform_expected(10, SMOKE_COUNT)) < 27.9
    end
end

@testset "exponential draws match the exponential CDF" begin
    for seed in SMOKE_SEEDS
        draws = randexp(Philox4x32(seed), Float64, SMOKE_COUNT)
        @test _ks_deviation(draws, x -> -expm1(-x)) < 2.0
    end
end

@testset "small range draws are uniform" begin
    for seed in SMOKE_SEEDS
        counts = _bin_counts(rand(Philox4x32(seed), 1:6, SMOKE_COUNT), 6, identity)
        @test _chisquare(counts, _uniform_expected(6, SMOKE_COUNT)) < 20.5
    end
end

@testset "wide range draws are uniform" begin
    # A span above 2^32 takes the 128-bit candidate path in the range reduction.
    span = 1 << 40
    width = span ÷ 64
    for seed in SMOKE_SEEDS
        draws = rand(Philox4x32(seed), 1:span, SMOKE_COUNT)
        counts = _bin_counts(draws, 64, value -> (value - 1) ÷ width + 1)
        @test _chisquare(counts, _uniform_expected(64, SMOKE_COUNT)) < 103.0
    end
end

@testset "weighted sampling matches the normalised weights" begin
    weights = [1.0, 2.0, 3.0, 4.0]
    expected = map(weight -> SMOKE_COUNT * weight / sum(weights), weights)
    for seed in SMOKE_SEEDS
        draws = randsample(Philox4x32(seed), 1:4, weights, SMOKE_COUNT)
        @test _chisquare(_bin_counts(draws, 4, identity), expected) < 16.3
    end
end

@testset "uniform draws below a threshold are binomial" begin
    probability = 0.375
    mean = SMOKE_COUNT * probability
    deviation = sqrt(SMOKE_COUNT * probability * (1 - probability))
    for seed in SMOKE_SEEDS
        below = count(<(probability), rand(Philox4x32(seed), Float64, SMOKE_COUNT))
        @test abs(below - mean) / deviation < 4.5
    end
end
