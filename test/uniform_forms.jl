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

        pure = rand(rng, T, 12)
        continued, next_rng = rand_next(rng, T, 12)
        @test pure == continued
        @test rand(rng, T, 12) == pure
        @test next_rng.position == _reference_position(rng, 12 * PureRNGs._draw_bits(T))

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

@testset "CPU allocating defaults and return order" begin
    # The untyped array form of each family draws Float64 and reaches the same
    # generator as the typed one.
    for F in GENERATOR_TYPES
        rng = F(0x62b)
        for (typed_form, default_form) in (
            ((rng, dims...) -> rand_next(rng, Float64, dims...), rand_next),
            ((rng, dims...) -> randn_next(rng, Float64, dims...), randn_next),
            ((rng, dims...) -> randexp_next(rng, Float64, dims...), randexp_next),
        )
            typed, typed_next = typed_form(rng, 2, 3)
            default, default_next = default_form(rng, 2, 3)
            @test default == typed
            @test default_next === typed_next

            empty, empty_next = default_form(rng, 0)
            @test isempty(empty)
            @test eltype(empty) === Float64
            @test empty_next === rng
        end
    end
end

@testset "tuple dimensions match splatted dimensions" begin
    rng = Philox4x32(0x62c)
    @test rand(rng, Float32, (2, 3)) == rand(rng, Float32, 2, 3)
    @test randn(rng, Float64, (4,)) == randn(rng, Float64, 4)
    @test randexp(rng, Float32, (2, 2)) == randexp(rng, Float32, 2, 2)
    @test rand(rng, 1:6, (3, 4)) == rand(rng, 1:6, 3, 4)
    @test rand_next(rng, UInt32, (3, 2)) == rand_next(rng, UInt32, 3, 2)
    @test randn_next(rng, (5,)) == randn_next(rng, 5)
    @test randexp_next(rng, (2, 3)) == randexp_next(rng, 2, 3)
    @test rand_next(rng, 1:6, (3, 4)) == rand_next(rng, 1:6, 3, 4)
    @test size(rand(rng, Float64, ())) == ()
end
