using LinearAlgebra

@testset "distribution parameters cannot alias their destination" begin
    rng = Philox4x32(123)
    diagonal = MvNormal([1.0, 2.0], Diagonal([1.0, 4.0]))
    dense = MvNormal([1.0, 2.0], [1.0 0.2; 0.2 2.0])
    dirichlet = Dirichlet([0.2, 0.3])
    for (d, destination) in (
        (diagonal, diagonal.μ),
        (diagonal, diagonal.Σ.diag),
        (dense, view(cholesky(dense.Σ).factors, :, 1)),
        (dirichlet, dirichlet.alpha),
    )
        saved = copy(destination)
        expected = rand(rng, d)
        @test_throws ArgumentError rand!(rng, d, destination)
        @test destination == saved
        @test rand!(rng, d, similar(destination)) == expected
    end
end

# A multivariate normal draw maps length(d) standard normal draws by `μ + L z`,
# and n draws are one fill, a column per draw.
@testset "an MvNormal draw whitens the next standard normal draws" begin
    μ = [1.0, -2.0, 0.5]
    Σ = [2.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 0.5]
    L = cholesky(Σ).L
    for d in (MvNormal(μ, Σ), MvNormal(μ, Diagonal([1.0, 4.0, 9.0]))),
        F in (Philox4x32, ChaCha)

        rng = F(0x7b1, 3)
        factor = Matrix(cholesky(Matrix(d.Σ)).L)
        normals, after = randn_next(rng, Float64, 3)
        value, next_rng = rand_next(rng, d)
        @test value ≈ μ + factor * normals
        @test next_rng == after
        @test rand(rng, d) == value
        columns, columns_next = randn_next(rng, Float64, 3, 50)
        values, values_next = rand_next(rng, d, 50; threaded = true)
        @test values ≈ μ .+ factor * columns
        @test values_next == columns_next
        # A single draw applies the factor with BLAS's vector kernel and a fill
        # with its matrix kernel, which round the last bit differently on some
        # platforms. The normals underneath are the same stream.
        chained = rng
        for j = 1:50
            column, chained = rand_next(chained, d)
            @test column ≈ values[:, j] rtol = 4 * eps(Float64)
        end
        @test chained == values_next
        @test rand_at(rng, d, 7) ≈ values[:, 7] rtol = 4 * eps(Float64)
        @test rand!(rng, d, zeros(3, 50)) == values
        @test last(rand_next!(rng, d, zeros(3))) == next_rng
    end
end
