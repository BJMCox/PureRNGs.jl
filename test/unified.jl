using Random

@testset "unified generation follows native values and state" begin
    root = Philox4x32(17)
    for (generate, next, draw) in (
        (randugen!!, rand_next, rand),
        (randngen!!, randn_next, randn),
        (randexpgen!!, randexp_next, randexp),
    )
        @test generate(root) == next(root)
        @test generate(root, Float32, 2, 3) == next(root, Float32, 2, 3)
        @test generate(root, (2, 3)) == next(root, Float64, (2, 3))
        mutable_rng, reference = Xoshiro(17), Xoshiro(17)
        for args in ((), (Float32,), (Float64, (2, 3)))
            value, after = generate(mutable_rng, args...)
            @test value == draw(reference, args...)
            @test after === mutable_rng
        end
        @test rand(mutable_rng, UInt64) == rand(reference, UInt64)
    end
    @test randgen!!(root, (2, 3)) == rand_next(root, (2, 3))
    @test randgen!!(root, Float64, (2, 3)) == rand_next(root, Float64, (2, 3))
    mutable_rng, reference = Xoshiro(18), Xoshiro(18)
    @test first(randgen!!(mutable_rng, [:a, :b], 4)) == rand(reference, [:a, :b], 4)
    @test rand(mutable_rng, UInt64) == rand(reference, UInt64)
end

@testset "unified setters reuse storage and replace scalar values" begin
    root = Philox4x32(19)
    for (set, next, next!, fill!) in (
        (randuset!!, rand_next, rand_next!, rand!),
        (randnset!!, randn_next, randn_next!, randn!),
        (randexpset!!, randexp_next, randexp_next!, randexp!),
    )
        @test set(root, 0.0f0) == next(root, Float32)
        storage = zeros(8)
        destination = view(storage, 2:5)
        expected, expected_rng = next!(root, zeros(4))
        value, after = set(root, destination)
        @test value === destination
        @test (value, after) == (expected, expected_rng)
        @test storage[[1, 6, 7, 8]] == zeros(4)
        mutable_rng, reference = Xoshiro(19), Xoshiro(19)
        value, after = set(mutable_rng, destination)
        @test value === destination
        @test after === mutable_rng
        @test value == fill!(reference, view(zeros(8), 2:5))
        @test rand(mutable_rng, UInt64) == rand(reference, UInt64)
    end
    @test randset!!(root, Float32, 0.0) == rand_next(root, Float32)
    old = zeros(3)
    value, after = randset!!(root, UInt32, old)
    @test (value, after) == rand_next(root, UInt32, 3)
    @test old == zeros(3)
    value, after = randset!!(root, 1:6, zeros(Int, 4))
    @test (value, after) == rand_next(root, 1:6, 4)
    @test randset!!(root, Float64, 1:3) == rand_next(root, Float64, 3)
end

@testset "unified setters preserve aliased populations" begin
    for F in (Philox4x32, Xoshiro)
        rng, reference = F(20), F(20)
        population = collect(11:18)
        destination = view(population, 2:5)
        expected, expected_rng = randgen!!(reference, copy(population), 4)
        value, after = randset!!(rng, population, destination)
        @test value == expected
        @test population == collect(11:18)
        @test !Base.mightalias(value, population)
        @test first(randgen!!(after, UInt64)) == first(randgen!!(expected_rng, UInt64))
    end
end
