@testset "R23-R26 CPU allocating uniform draws" begin
    for F in FAMILY_TYPES, T in PURE_UNIFORM_TYPES
        rng = F(0x62a)
        original_position = rng.position

        pure = rand(rng, T, 12)
        next_rng, continued = rand_next(rng, T, 12)
        @test pure isa Vector{T}
        @test continued isa Vector{T}
        @test pure == continued
        @test rng.position == original_position
        @test next_rng.position ==
              _reference_position(rng, 12 * PureRNGs._draw_bits(T))

        cursor = rng
        chained = Vector{T}(undef, 12)
        for index in eachindex(chained)
            cursor, chained[index] = rand_next(cursor, T)
        end
        @test continued == chained
        @test next_rng === cursor

        matrix = rand(rng, T, 3, 4)
        matrix_next, continued_matrix = rand_next(rng, T, 3, 4)
        @test matrix isa Matrix{T}
        @test size(matrix) == (3, 4)
        @test vec(matrix) == pure
        @test continued_matrix == matrix
        @test matrix_next === next_rng

        mixed_dims = rand(rng, T, UInt8(2), Int16(3))
        @test size(mixed_dims) == (2, 3)
        @test vec(mixed_dims) == pure[1:6]

        @test rand(rng, T, 5) == pure[1:5]
        prefix_next, prefix = rand_next(rng, T, 5)
        cursor_after = foldl((state, _) -> first(rand_next(state, T)), 1:5; init = rng)
        @test prefix == pure[1:5]
        @test prefix_next === cursor_after

        empty = rand(rng, T, 0, 2)
        empty_next, continued_empty = rand_next(rng, T, 0, 2)
        @test empty isa Matrix{T}
        @test size(empty) == (0, 2)
        @test continued_empty == empty
        @test empty_next === rng

        @test @inferred(rand(rng, T, 2, 3)) isa Matrix{T}
        @test @inferred(rand_next(rng, T, 2, 3)) isa Tuple{typeof(rng),Matrix{T}}
    end
end

@testset "R23 and R24 CPU allocating defaults and return order" begin
    for F in FAMILY_TYPES
        rng = F(0x62b)
        typed_next, typed = rand_next(rng, Float64, 2, 3)
        default_next, default = rand_next(rng, 2, 3)
        @test default isa Matrix{Float64}
        @test default == typed
        @test default_next === typed_next
        @test first(rand_next(rng, UInt32, 3)) isa typeof(rng)
        @test last(rand_next(rng, UInt32, 3)) isa Vector{UInt32}
        @test @inferred(rand_next(rng, 2, 3)) isa Tuple{typeof(rng),Matrix{Float64}}
    end
end

@testset "R23, R47, and R49 CPU allocating method surface" begin
    rng = Philox4x32(0x62c)
    for T in PURE_UNIFORM_TYPES
        @test which(rand, (typeof(rng), Type{T}, Int)).module === PureRNGs
        @test which(rand_next, (typeof(rng), Type{T}, Int)).module === PureRNGs
        @test Base.kwarg_decl(which(rand, (typeof(rng), Type{T}, Int))) == Symbol[]
        @test Base.kwarg_decl(which(rand_next, (typeof(rng), Type{T}, Int))) == Symbol[]
        @test_throws ArgumentError rand(rng, T, -1)
        @test_throws ArgumentError rand(rng, T, 2, -1)
        @test_throws ArgumentError rand_next(rng, T, -1)
        @test_throws ArgumentError rand_next(rng, T, 2, -1)
        @test_throws MethodError rand(rng, T, 3; threaded = false)
        @test_throws MethodError rand_next(rng, T, 3; threaded = false)
    end

    @test which(rand_next, (typeof(rng), Int)).module === PureRNGs
    @test which(rand, (typeof(rng), Int)).module !== PureRNGs
    @test_throws MethodError rand(rng, 3)
    for unsupported in (Float16, ComplexF64)
        @test !applicable(rand, rng, unsupported, 3)
        @test !applicable(rand_next, rng, unsupported, 3)
    end
end
