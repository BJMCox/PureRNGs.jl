function test_pcg4d_device(device)
    @testset "PCG4D device stream" begin
        for (T, count, offset) in
            ((Bool, 8193, 127), (UInt32, 1027, 0), (UInt64, 257, 63), (Float32, 257, 29))
            root = PCG4D(12345, offset)
            expected, expected_next = rand_next(root, T, count)
            values, next = rand_next(root |> device, T, count)
            @test Array(values) == expected
            @test rngposition(next) == rngposition(expected_next)
        end
    end
end
