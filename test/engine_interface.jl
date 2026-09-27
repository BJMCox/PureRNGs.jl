include("engine_fixture.jl")

const ENGINE_GENERATORS = (
    Philox4x32(0xe19, 37),
    ChaCha8(0xe19, 500),
    Threefry4x64(0xe19, 200),
    Philox2x32(0xe19),
)

@testset "an external engine reproduces normal and exponential draws" begin
    for rng in ENGINE_GENERATORS,
        (next, next!, at, types) in (
            (randn_next, randn_next!, randn_at, (Float16, Float32, Float64, ComplexF64)),
            (randexp_next, randexp_next!, randexp_at, (Float16, Float32, Float64)),
        )

        engine = WrappedEngine(rng)
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
