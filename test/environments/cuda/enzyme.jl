function _enzyme_fill_result!(fill_function, rng, destination, threaded)
    return fill_function(rng, destination; threaded = threaded)
end

const CUDA_ENZYME_FILL_CASES = (
    (Random.rand!, rand_next, false),
    (rand_next!, rand_next, true),
    (Random.randn!, randn_next, false),
    (randn_next!, randn_next, true),
    (Random.randexp!, randexp_next, false),
    (randexp_next!, randexp_next, true),
)

@testset "CUDA Enzyme immutable fill rules" begin
    rng = device(Philox4x32(0x65c0))
    for T in (Float32, Float64),
        (fill_function, next_draw, continued) in CUDA_ENZYME_FILL_CASES

        expected_rng, expected = next_draw(rng, T, 17)

        values = CUDA.zeros(T, 17)
        shadow = CUDA.fill(T(7), 17)
        shadow_result, primal_result = autodiff(
            ForwardWithPrimal,
            _enzyme_fill_result!,
            Duplicated,
            Const(fill_function),
            Const(rng),
            Duplicated(values, shadow),
            Const(true),
        )
        if continued
            @test primal_result[1] === expected_rng
            @test primal_result[2] === values
            @test shadow_result[2] === shadow
        else
            @test primal_result === values
            @test shadow_result === shadow
        end
        @test isequal(Array(values), Array(expected))
        @test iszero(Array(shadow))
        @test _device_id(values) == CUDA.deviceid(primary)
        @test _device_id(shadow) == CUDA.deviceid(primary)

        reverse_values = CUDA.zeros(T, 17)
        reverse_shadow = CUDA.fill(T(5), 17)
        autodiff(
            Reverse,
            _enzyme_fill_result!,
            Const,
            Const(fill_function),
            Const(rng),
            Duplicated(reverse_values, reverse_shadow),
            Const(true),
        )
        @test isequal(Array(reverse_values), Array(expected))
        @test iszero(Array(reverse_shadow))

        batch_values = CUDA.zeros(T, 17)
        shadow_one = CUDA.fill(T(3), 17)
        shadow_two = CUDA.fill(T(4), 17)
        autodiff(
            Forward,
            _enzyme_fill_result!,
            Const,
            Const(fill_function),
            Const(rng),
            BatchDuplicated(batch_values, (shadow_one, shadow_two)),
            Const(true),
        )
        @test isequal(Array(batch_values), Array(expected))
        @test iszero(Array(shadow_one))
        @test iszero(Array(shadow_two))
    end
end

@testset "CUDA Enzyme fills do not stage through the host" begin
    rng = device(Philox4x32(0x65c1))
    values = CUDA.zeros(Float32, 4096)
    shadow = CUDA.ones(Float32, 4096)
    autodiff(
        Forward,
        _enzyme_fill_result!,
        Const,
        Const(randexp_next!),
        Const(rng),
        Duplicated(values, shadow),
        Const(true),
    )
    CUDA.synchronize()

    profile = CUDA.Profile.profile_internally(; concurrent = false, trace = true) do
        autodiff(
            Forward,
            _enzyme_fill_result!,
            Const,
            Const(randexp_next!),
            Const(rng),
            Duplicated(values, shadow),
            Const(true),
        )
        CUDA.synchronize()
    end
    h2d, d2h = _cuda_copy_sizes(profile)
    @test all(==(8), h2d)
    @test isempty(d2h)
    @test iszero(Array(shadow))
end
