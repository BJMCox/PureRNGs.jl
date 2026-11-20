@testset "zero-based CPU fills" begin
    for (fill!, T) in ((rand_next!, UInt32), (randn_next!, Float32))
        rng = Philox4x32(0x5240)
        serial = ZeroBasedVector(Vector{T}(undef, 9))
        threaded = ZeroBasedVector(similar(serial.data))

        fill!(rng, serial; threaded = false)
        fill!(rng, threaded; threaded = true)

        @test collect(threaded) == collect(serial)

        serial_matrix = IdentityAxesMatrix(Matrix{T}(undef, 2, 3))
        threaded_matrix = IdentityAxesMatrix(similar(serial_matrix.data))
        fill!(rng, serial_matrix; threaded = false)
        fill!(rng, threaded_matrix; threaded = true)

        @test threaded_matrix.data == serial_matrix.data
    end
end

@testset "threaded fills into packed Boolean storage" begin
    rng = Philox4x32(1729)
    population = Bool[false, true]
    n = 32_769
    order = vcat((i:4:n for i = 1:4)...)
    for weights in ((), ([1.0, 2.0],))
        expected, after = randsample_next(rng, population, weights..., n; threaded = false)
        for dest in (falses(n), view(falses(n), order))
            _, next = randsample_next!(rng, population, weights..., dest; threaded = true)
            @test dest == expected
            @test next == after
        end
    end
    dest = PermutedDimsArray(falses(8, 65536), (2, 1))
    expected, after = rand_next(rng, Bool, size(dest); threaded = false)
    _, next = rand_next!(rng, dest; threaded = true)
    @test dest == expected
    @test next == after
end
