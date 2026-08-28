function _enzyme_fill_result!(fill_function, rng, destination, threaded)
    return fill_function(rng, destination; threaded = threaded)
end

function _enzyme_fill_objective!(
    fill_function,
    rng,
    destination,
    scale,
    reference,
    threaded,
)
    # Exercise caller differentiation without depending on CUDACore's reduction rules.
    fill_function(rng, destination; threaded = threaded)
    return scale * reference
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
        reference = sum(Array(expected))

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

        objective_values = CUDA.zeros(T, 17)
        objective_shadow = CUDA.fill(T(6), 17)
        reverse_derivative = only(
            autodiff(
                Reverse,
                _enzyme_fill_objective!,
                Active,
                Const(fill_function),
                Const(rng),
                Duplicated(objective_values, objective_shadow),
                Active(T(1.5)),
                Const(reference),
                Const(true),
            ),
        )
        @test isequal(Array(objective_values), Array(expected))
        @test iszero(Array(objective_shadow))
        @test reverse_derivative[4] ≈ reference

        forward_values = CUDA.zeros(T, 17)
        forward_shadow = CUDA.fill(T(8), 17)
        forward_derivative = only(
            autodiff(
                Forward,
                _enzyme_fill_objective!,
                Const(fill_function),
                Const(rng),
                Duplicated(forward_values, forward_shadow),
                Duplicated(T(1.5), one(T)),
                Const(reference),
                Const(true),
            ),
        )
        @test isequal(Array(forward_values), Array(expected))
        @test iszero(Array(forward_shadow))
        @test forward_derivative ≈ reference

        batch_values = CUDA.zeros(T, 17)
        shadow_one = CUDA.fill(T(3), 17)
        shadow_two = CUDA.fill(T(4), 17)
        batch_derivative = only(
            autodiff(
                Forward,
                _enzyme_fill_objective!,
                Const(fill_function),
                Const(rng),
                BatchDuplicated(batch_values, (shadow_one, shadow_two)),
                BatchDuplicated(T(1.5), (one(T), T(2))),
                Const(reference),
                Const(true),
            ),
        )
        @test isequal(Array(batch_values), Array(expected))
        @test iszero(Array(shadow_one))
        @test iszero(Array(shadow_two))
        @test batch_derivative[1] ≈ reference
        @test batch_derivative[2] ≈ T(2) * reference
    end
end

@testset "CUDA Enzyme fill rules execute the primal once" begin
    rng = device(Philox4x32(0x65c2))
    values = CUDA.zeros(Float32, 4096)
    shadow_one = similar(values)
    shadow_two = similar(values)

    primal_events = _device_events(() -> randexp_next!(rng, values))
    zero_events = _device_events(() -> fill!(shadow_one, 0.0f0))
    @test !isempty(primal_events.kernels)
    @test isempty(primal_events.memsets)
    @test isempty(primal_events.copies)
    @test isempty(zero_events.kernels)
    @test !isempty(zero_events.memsets)
    @test isempty(zero_events.copies)

    forward_events = _device_events() do
        autodiff(
            Forward,
            _enzyme_fill_result!,
            Const,
            Const(randexp_next!),
            Const(rng),
            Duplicated(values, shadow_one),
            Const(true),
        )
    end
    @test length(forward_events.kernels) == length(primal_events.kernels)
    @test length(forward_events.memsets) == length(zero_events.memsets)
    @test isempty(forward_events.copies)

    reverse_events = _device_events() do
        autodiff(
            Reverse,
            _enzyme_fill_result!,
            Const,
            Const(randexp_next!),
            Const(rng),
            Duplicated(values, shadow_one),
            Const(true),
        )
    end
    @test length(reverse_events.kernels) == length(primal_events.kernels)
    @test length(reverse_events.memsets) == 2length(zero_events.memsets)
    @test isempty(reverse_events.copies)

    batch_events = _device_events() do
        autodiff(
            Forward,
            _enzyme_fill_result!,
            Const,
            Const(randexp_next!),
            Const(rng),
            BatchDuplicated(values, (shadow_one, shadow_two)),
            Const(true),
        )
    end
    @test length(batch_events.kernels) == length(primal_events.kernels)
    @test length(batch_events.memsets) == 2length(zero_events.memsets)
    @test isempty(batch_events.copies)
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
    events = _cuda_profile_events(profile)
    @test !isempty(events.kernels)
    @test !isempty(events.memsets)
    @test isempty(events.host_to_device)
    @test isempty(events.device_to_host)
    @test iszero(Array(shadow))
end
