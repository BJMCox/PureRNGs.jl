using Random: randexp

@testset "R23-R26 CPU allocating uniform draws" begin
    for F in GENERATOR_TYPES, T in PURE_UNIFORM_TYPES
        rng = F(0x62a)
        original_position = rng.position

        pure = rand(rng, T, 12)
        continued, next_rng = rand_next(rng, T, 12)
        @test pure == continued
        @test rng.position == original_position
        @test next_rng.position ==
              _reference_position(rng, 12 * PureRNGs._draw_bits(T))

        cursor = rng
        chained = Vector{T}(undef, 12)
        for index in eachindex(chained)
            chained[index], cursor = rand_next(cursor, T)
        end
        @test continued == chained
        @test next_rng === cursor

        matrix = rand(rng, T, 3, 4)
        continued_matrix, matrix_next = rand_next(rng, T, 3, 4)
        @test size(matrix) == (3, 4)
        @test vec(matrix) == pure
        @test continued_matrix == matrix
        @test matrix_next === next_rng

        @test rand(rng, T, 5) == pure[1:5]
        prefix, prefix_next = rand_next(rng, T, 5)
        cursor_after = foldl((state, _) -> last(rand_next(state, T)), 1:5; init = rng)
        @test prefix == pure[1:5]
        @test prefix_next === cursor_after

        empty = rand(rng, T, 0, 2)
        continued_empty, empty_next = rand_next(rng, T, 0, 2)
        @test size(empty) == (0, 2)
        @test continued_empty == empty
        @test empty_next === rng

    end
end

@testset "R23 and R24 CPU allocating defaults and return order" begin
    for F in GENERATOR_TYPES
        rng = F(0x62b)
        typed, typed_next = rand_next(rng, Float64, 2, 3)
        default, default_next = rand_next(rng, 2, 3)
        @test default == typed
        @test default_next === typed_next
    end
end

@testset "R23 tuple dimensions match splatted dimensions" begin
    rng = Philox4x32(0x62c)
    @test rand(rng, Float32, (2, 3)) == rand(rng, Float32, 2, 3)
    @test randn(rng, Float64, (4,)) == randn(rng, Float64, 4)
    @test randexp(rng, Float32, (2, 2)) == randexp(rng, Float32, 2, 2)
    @test rand_next(rng, UInt32, (3, 2)) == rand_next(rng, UInt32, 3, 2)
    @test randn_next(rng, (5,)) == randn_next(rng, 5)
    @test randexp_next(rng, (2, 3)) == randexp_next(rng, 2, 3)
    @test size(rand(rng, Float64, ())) == ()
end
