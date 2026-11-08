@testset "Philox4x64 allocating continuation retains the stream" begin
    rng = Philox4x64(0x62a)
    _, successor = rand_next(rng, UInt64, 12)
    cursor = rng
    for _ = 1:12
        _, cursor = rand_next(cursor, UInt64)
    end
    @test first(rand_next(successor, UInt64)) == first(rand_next(cursor, UInt64))
end

@testset "CPU allocating uniform draws" begin
    for F in GENERATOR_TYPES, T in PURE_UNIFORM_TYPES
        rng = F(0x62a)

        pure = first(rand_next(rng, T, 12))
        continued, next_rng = rand_next(rng, T, 12)
        @test pure == continued
        @test first(rand_next(rng, T, 12)) == pure
        @test next_rng.position == _reference_position(rng, 12 * PureRNGs._draw_bits(T))

        cursor = rng
        chained = Vector{T}(undef, 12)
        for index in eachindex(chained)
            chained[index], cursor = rand_next(cursor, T)
        end
        @test continued == chained
        @test next_rng === cursor

        matrix = first(rand_next(rng, T, 3, 4))
        continued_matrix, matrix_next = rand_next(rng, T, 3, 4)
        @test size(matrix) == (3, 4)
        @test vec(matrix) == pure
        @test continued_matrix == matrix
        @test matrix_next === next_rng

        @test first(rand_next(rng, T, 5)) == pure[1:5]
        prefix, prefix_next = rand_next(rng, T, 5)
        cursor_after = foldl((state, _) -> last(rand_next(state, T)), 1:5; init = rng)
        @test prefix == pure[1:5]
        @test prefix_next === cursor_after

        empty = first(rand_next(rng, T, 0, 2))
        continued_empty, empty_next = rand_next(rng, T, 0, 2)
        @test size(empty) == (0, 2)
        @test continued_empty == empty
        @test empty_next === rng

    end
end

@testset "primitive defaults equal explicit Float64 draws" begin
    rng = Philox4x32(0x62b)
    for draw_next in (rand_next, randn_next, randexp_next)
        @test draw_next(rng) === draw_next(rng, Float64)
        for threaded in (false, true)
            typed = draw_next(rng, Float64, 2, 3; threaded)
            @test draw_next(rng, 2, 3; threaded) == typed
        end

        empty, empty_next = draw_next(rng, 0)
        @test empty == Float64[]
        @test empty_next === rng
    end
end

@testset "tuple dimensions match splatted dimensions" begin
    rng = Philox4x32(0x62c)
    @test first(rand_next(rng, Float32, (2, 3))) == first(rand_next(rng, Float32, 2, 3))
    @test first(randn_next(rng, Float64, (4,))) == first(randn_next(rng, Float64, 4))
    @test first(randexp_next(rng, Float32, (2, 2))) ==
          first(randexp_next(rng, Float32, 2, 2))
    @test first(rand_next(rng, 1:6, (3, 4))) == first(rand_next(rng, 1:6, 3, 4))
    @test rand_next(rng, UInt32, (3, 2)) == rand_next(rng, UInt32, 3, 2)
    @test randn_next(rng, (5,)) == randn_next(rng, 5)
    @test randexp_next(rng, (2, 3)) == randexp_next(rng, 2, 3)
    @test first(randn_next(rng, (5,); threaded = true)) ==
          first(randn_next(rng, Float64, 5))
    @test first(randexp_next(rng, (2, 3); threaded = true)) ==
          first(randexp_next(rng, Float64, 2, 3))
    @test rand_next(rng, 1:6, (3, 4)) == rand_next(rng, 1:6, 3, 4)
    @test size(first(rand_next(rng, Float64, ()))) == ()
end
