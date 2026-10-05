@testset "unified distribution fills keep the native sample shape" begin
    for F in (Philox4x32, Random.Xoshiro)
        distribution = MvNormal([1.0, 2.0], [1.0, 0.5])
        tupled, after = randgen!!(F(27), distribution, (3,))
        expanded, reference = randgen!!(F(27), distribution, 3)
        @test tupled == expanded
        @test size(tupled) == (2, 3)
        @test first(randgen!!(after, UInt64)) == first(randgen!!(reference, UInt64))
    end
    for F in (Philox4x32, Random.Xoshiro),
        (distribution, shape) in (
            (Normal(2.0, 3.0), (2, 3)),
            (MvNormal([1.0, 2.0], [1.0, 0.5]), (2,)),
            (MvNormal([1.0, 2.0], [1.0, 0.5]), (2, 3)),
        )

        rng, reference = F(21), F(21)
        destination = zeros(shape)
        expected = zeros(shape)
        expected_rng = if reference isa AbstractPureRNG
            last(rand_next!(reference, distribution, expected))
        else
            rand!(reference, distribution, expected)
            reference
        end
        value, after = randset!!(rng, distribution, destination)
        @test value === destination
        @test value == expected
        @test first(randgen!!(after, UInt64)) == first(randgen!!(expected_rng, UInt64))
    end
end

@testset "nested native fills do not overwrite borrowed samples" begin
    distribution = MvNormal([1.0, 2.0], [1.0, 0.5])
    saved = copy(distribution.μ)
    rng, reference = Random.Xoshiro(26), Random.Xoshiro(26)
    destination = [distribution.μ]
    values, after = randset!!(rng, distribution, destination)
    @test values == rand(reference, distribution, (1,))
    @test distribution.μ == saved
    @test values === destination
    @test first(randgen!!(after, UInt64)) == rand(reference, UInt64)
end

@testset "sources with hidden references use replacement storage" begin
    component = MvNormal([1.0, 2.0], [1.0, 0.5])
    distribution = MixtureModel([component])
    saved = copy(component.μ)
    rng, reference = Random.Xoshiro(29), Random.Xoshiro(29)
    expected = rand!(reference, distribution, zeros(2))
    value, after = randset!!(rng, distribution, component.μ)
    @test value == expected
    @test component.μ == saved
    @test value !== component.μ
    @test first(randgen!!(after, UInt64)) == rand(reference, UInt64)
end

@testset "unified fills preserve aliased distribution parameters" begin
    for F in (Philox4x32, Random.Xoshiro), kind in (:mean, :factor, :alpha)
        distribution =
            kind === :alpha ? Dirichlet([1.0, 2.0]) :
            MvNormal([1.0, 2.0], [2.0 0.5; 0.5 1.0])
        destination =
            kind === :mean ? distribution.μ :
            kind === :factor ? view(distribution.Σ.chol.factors, :, 1) : distribution.alpha
        saved = copy(destination)
        rng, reference = F(22), F(22)
        expected = similar(destination)
        expected_rng = if reference isa AbstractPureRNG
            last(rand_next!(reference, distribution, expected))
        else
            rand!(reference, distribution, expected)
            reference
        end
        value, after = randset!!(rng, distribution, destination)
        @test value == expected
        @test destination == saved
        @test !Base.mightalias(value, destination)
        @test first(randgen!!(after, UInt64)) == first(randgen!!(expected_rng, UInt64))
    end
end
