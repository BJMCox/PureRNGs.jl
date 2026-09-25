using PureRNGs
using StatsBase
using Test

# Every StatsBase form is the `randsample` draw at the held position.
@testset "StatsBase samples equal randsample" begin
    population = [40, 10, 30, 20, 50]
    raw = [1.0, 0.0, 2.0, 3.0, 0.5]
    wv = Weights(raw)
    for F in (Philox4x32, ChaCha), replace in (true, false)
        rng = F(0x5b1, 3)
        expected = randsample(rng, population, 3; replace)
        @test sample(rng, population, 3; replace) == expected
        @test sample!(rng, population, zeros(3); replace) == expected
        @test sample(rng, population, (1, 3); replace) == reshape(expected, 1, 3)
        weighted = randsample(rng, population, raw, 3; replace)
        @test sample(rng, population, wv, 3; replace) == weighted
        @test sample!(rng, population, wv, zeros(3); replace) == weighted
        @test wsample(rng, population, raw, (3,); replace) == weighted
        @test wsample!(rng, population, raw, zeros(3); replace) == weighted
    end
    rng = Philox4x32(0x5b2)
    @test sample(rng, population) == rand(rng, population)
    @test sample(rng, population, wv) == only(randsample(rng, population, raw, 1))
    @test population[sample(rng, wv)] == sample(rng, population, wv)
    @test wsample(rng, raw) == sample(rng, wv)
    # Unit weights mean no weights, as in StatsBase.
    @test sample(rng, population, UnitWeights{Float64}(5), 3) ==
          randsample(rng, population, 3)
end

@testset "samplepair makes two range draws" begin
    for n in (2, 3, 10)
        for position = 0:20
            rng = Philox4x32(0x5b4, position)
            i, next_rng = rand_next(rng, 1:n)
            j = rand(next_rng, 1:(n-1))
            @test samplepair(rng, n) == (i, j == i ? n : j)
        end
    end
    population = [40, 10, 30, 20, 50]
    rng = Philox4x32(0x5b5)
    i, j = samplepair(rng, 5)
    @test samplepair(rng, population) == (population[i], population[j])
end

# An ordered sample draws positions and lists them in population order.
@testset "ordered StatsBase samples list positions in order" begin
    population = [40, 10, 30, 20, 50]
    raw = [1.0, 0.0, 2.0, 3.0, 0.5]
    rng = Philox4x32(0x5b3)
    for replace in (true, false)
        positions = randsample(rng, 1:5, 4; replace)
        @test sample(rng, population, 4; replace, ordered = true) ==
              population[sort(positions)]
        weighted = randsample(rng, 1:5, raw, 4; replace)
        @test sample(rng, population, Weights(raw), 4; replace, ordered = true) ==
              population[sort(weighted)]
    end
end
