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
