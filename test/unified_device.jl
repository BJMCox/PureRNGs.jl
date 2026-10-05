function test_unified_device(rng)
    @testset "unified calls preserve device fills and continuation" begin
        expected, expected_rng = rand_next(rng, Float32, 257)
        values, after = randugen!!(rng, Float32, 257)
        @test typeof(values) === typeof(expected)
        @test Array(values) == Array(expected)
        @test after == expected_rng
        result, after = randset!!(rng, Float32, values)
        @test result === values
        @test Array(result) == Array(expected)
        @test after == expected_rng
    end
end
