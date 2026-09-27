using LinearAlgebra: Diagonal

include(joinpath(@__DIR__, "..", "..", "..", "engine_fixture.jl"))

engine_distributions(::Type{T}) where {T} = (
    Normal(T(0.5), T(2)),
    Laplace(T(0.2), T(1.5)),
    Bernoulli(T(0.3)),
    Gamma(T(0.4), T(1)),
    Gamma(T(2.5), T(1.5)),
    InverseGamma(T(2.5), T(1.2)),
    Beta(T(0.7), T(1.8)),
    TDist(T(3.5)),
)

engine_result_type(::Bernoulli) = Bool
engine_result_type(::DiscreteUniform) = Int
engine_result_type(d) = partype(d)

@testset "an external engine reproduces distribution draws" begin
    univariate = (
        engine_distributions(Float32)...,
        engine_distributions(Float64)...,
        DiscreteUniform(-3, 10^12),
    )
    for rng in (Philox4x32(0xd15, 37), ChaCha8(0xd15, 500)), d in univariate
        engine = WrappedEngine(rng)
        @test unwrap(rand_next(engine, d)) === rand_next(rng, d)
        @test rand_at(engine, d, 7) === rand_at(rng, d, 7)
        @test unwrap(rand_next(engine, d, 33)) == rand_next(rng, d, 33)
        for threaded in (false, true)
            a = Vector{engine_result_type(d)}(undef, 2049)
            b = similar(a)
            @test unwrap(rand_next!(engine, d, a; threaded)) ==
                  rand_next!(rng, d, b; threaded)
        end
    end
    multivariate = (
        Dirichlet([0.5, 1.5, 2.0]),
        MvNormal([0.5, -1.0], Diagonal([2.0, 0.5])),
        MvNormal([0.5, -1.0, 2.0], [2.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.5]),
    )
    for rng in (Philox4x32(0xd15, 37), Threefry4x64(0xd15, 200)), d in multivariate
        engine = WrappedEngine(rng)
        @test unwrap(rand_next(engine, d)) == rand_next(rng, d)
        @test rand_at(engine, d, 5) == rand_at(rng, d, 5)
        @test unwrap(rand_next(engine, d, 65)) == rand_next(rng, d, 65)
        a = Matrix{Float64}(undef, length(d), 129)
        b = similar(a)
        @test unwrap(rand_next!(engine, d, a; threaded = true)) ==
              rand_next!(rng, d, b; threaded = true)
    end
end

@testset "an external engine serves Dirichlet columns and sees whole spans" begin
    d = Dirichlet([0.5, 1.5, 2.0])
    rng = Philox4x32(0xd17, 5)
    engine = WrappedEngine(rng)
    columns = COLUMN_FILLS[]
    for threaded in (false, true)
        a, b = zeros(3, 257), zeros(3, 257)
        @test unwrap(rand_next!(engine, d, a; threaded)) == rand_next!(rng, d, b; threaded)
    end
    @test rand_at(engine, d, 9) == rand_at(rng, d, 9)
    @test COLUMN_FILLS[] == columns + 3
    @test LAST_RESERVATION[] == (1, 51, 52)
    wide = UInt64(0x5555555555555557)
    @test_throws OverflowError rand_at(engine, d, wide)
    @test LAST_ADDRESS[] == (51, 52, wide)
    huge = WriteCountingMatrix(3, 2^61)
    @test_throws OverflowError rand_next!(engine, d, huge)
    @test LAST_RESERVATION[] == (2^61, 51, 52)
    @test huge.writes[] == 0
end

# One candidate sends about 5 % of shape-one draws to the child stream, so these
# draws exercise the fallback's stream ordinal through the engine's cursor.
@testset "an external engine keys the Gamma fallback by stream ordinal" begin
    rng = Philox4x32(0xfa11, 91)
    engine = WrappedEngine(rng)
    gamma(shape, candidates) = PureRNGs._GammaCodec(shape, 1.0, rng.device, candidates)
    for codec in (
        gamma(1.0, 1),
        PureRNGs._BetaCodec(gamma(0.7, 1), gamma(1.8, 1)),
        PureRNGs._TDistCodec(gamma(1.5, 1), 3.0),
    )
        a, b = Vector{Float64}(undef, 400), Vector{Float64}(undef, 400)
        @test unwrap(PureRNGs._engine_fill!(engine, a, false, codec)) ==
              PureRNGs._engine_fill!(rng, b, false, codec)
        @test all(i -> PureRNGs._engine_draw_at(engine, codec, Float64, i) == b[i], 1:400)
    end
    # At a shared start, one candidate differs from eight exactly when its only
    # candidate rejects and the draw moves to the child stream.
    starts = [PureRNGs._addressed_state(rng, 17, Val(52), j) for j = 1:400]
    one_candidate(r) = first(PureRNGs._engine_draw_next(r, gamma(1.0, 1), Float64))
    eight(r) = first(PureRNGs._engine_draw_next(r, gamma(1.0, 8), Float64))
    @test count(r -> one_candidate(r) != eight(r), starts) >= 3
    @test all(r -> one_candidate(WrappedEngine(r)) === one_candidate(r), starts)
end
