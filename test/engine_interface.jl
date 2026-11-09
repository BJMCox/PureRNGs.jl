include("engine_fixture.jl")

const ENGINE_GENERATORS = (
    Philox4x32(0xe19, 37),
    ChaCha8(0xe19, 500),
    Threefry4x64(0xe19, 200),
    Philox2x32(0xe19),
)

# The wrapped engine overrides the reservation and the fills; the minimal engine
# takes the core's defaults for them.
@testset "an external engine reproduces normal and exponential draws" begin
    for rng in ENGINE_GENERATORS,
        wrap in (WrappedEngine, MinimalEngine),
        (next, next!, at, types) in (
            (randn_next, randn_next!, randn_at, (Float16, Float32, Float64, ComplexF64)),
            (randexp_next, randexp_next!, randexp_at, (Float16, Float32, Float64)),
        )

        engine = wrap(rng)
        for T in types
            @test unwrap(next(engine, T)) === next(rng, T)
            @test at(engine, T, 9) === at(rng, T, 9)
            @test at(engine, T, 4:40) == at(rng, T, 4:40)
            @test unwrap(next(engine, T, 5, 3)) == next(rng, T, 5, 3)
            for threaded in (false, true)
                a, b = Vector{T}(undef, 3001), Vector{T}(undef, 3001)
                @test unwrap(next!(engine, a; threaded)) == next!(rng, b; threaded)
            end
        end
        @test unwrap(next(engine)) === next(rng)
    end
end

@testset "an external engine uses the stateful bridge" begin
    root = Philox4x32(0xe20, 37)
    bridge = StatefulRNG(WrappedEngine(root))
    cursor = root
    for (draw, next, T) in (
        (rand, rand_next, UInt64),
        (rand, rand_next, Float32),
        (randn, randn_next, ComplexF64),
        (randexp, randexp_next, Float64),
    )
        expected, cursor = next(cursor, T)
        @test draw(bridge, T) === expected
        @test parent(bridge).inner === cursor
    end
    for (draw!, next!, destination) in (
        (rand!, rand_next!, zeros(Float64, 7)),
        (randn!, randn_next!, zeros(Float32, 7)),
        (randexp!, randexp_next!, zeros(Float64, 7)),
        (rand!, rand_next!, falses(67)),
    )
        expected, cursor = next!(cursor, similar(destination); threaded = false)
        @test draw!(bridge, destination) === destination
        @test destination == expected
        @test parent(bridge).inner === cursor
    end

    replay = copy(bridge)
    @test rand(replay, UInt64) === rand(bridge, UInt64)
    @test parent(replay) === parent(bridge)
    @test Random.seed!(bridge, 0xe21) === bridge
    @test parent(bridge).inner === Philox4x32(0xe21)
    @test rand(bridge, UInt64) === first(rand_next(Philox4x32(0xe21), UInt64))
    @test Random.seed!(bridge, Random.RandomDevice()) === bridge
    @test rngposition(parent(bridge).inner) == 0

    # These operations use Random's algorithms; the foreign engine does not
    # provide PureRNGs' range, collection, or permutation interface.
    bridge = StatefulRNG(WrappedEngine(root))
    range = 1:17
    @test rand(bridge, range) in range
    population = (:first, :second, :third)
    @test rand(bridge, population) in population
    destination = zeros(Int, 7)
    @test all(in(range), rand!(bridge, destination, range))
    @test sort(shuffle!(bridge, collect(1:7))) == collect(1:7)
    @test sort(randperm!(bridge, zeros(Int, 7))) == collect(1:7)
    @test sort(randcycle!(bridge, zeros(Int, 7))) == collect(1:7)
end
