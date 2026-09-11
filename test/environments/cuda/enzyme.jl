function _enzyme_fill_result!(fill_function, rng, destination, threaded)
    return fill_function(rng, destination; threaded = threaded)
end

function _enzyme_fill_objective!(fill_function, rng, destination, scale, threaded)
    result = fill_function(rng, destination; threaded = threaded)
    values = result isa Tuple ? first(result) : result
    return scale * sum(values)
end

function _enzyme_fill_batch_objective!(
    fill_function,
    rng,
    destination,
    scale,
    reference,
    threaded,
)
    # Exercise batched caller differentiation without CUDACore's reduction rules.
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

const CUDA_ENZYME_EXPONENTIAL_CASES =
    ((Random.randexp!, randexp_next, false), (randexp_next!, randexp_next, true))

_device_work(events) = length(events.kernels) + length(events.memsets)

function _check_enzyme_primal!(
    rng,
    fill_function,
    next_draw,
    continued,
    ::Type{T},
) where {T}
    expected, expected_rng = next_draw(rng, T, 17)
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
        @test primal_result[1] === values
        @test primal_result[2] === expected_rng
        @test shadow_result[1] === shadow
    else
        @test primal_result === values
        @test shadow_result === shadow
    end
    @test isequal(Array(values), Array(expected))
    @test iszero(Array(shadow))
    @test _device_id(values) == CUDA.deviceid(primary)
    @test _device_id(shadow) == CUDA.deviceid(primary)
end

function _check_enzyme_reverse_gradient!(rng, fill_function, next_draw, ::Type{T}) where {T}
    expected, _ = next_draw(rng, T, 17)
    reference = sum(Array(expected))
    values = CUDA.zeros(T, 17)
    shadow = CUDA.fill(T(6), 17)
    derivative = only(
        autodiff(
            Reverse,
            _enzyme_fill_objective!,
            Active,
            Const(fill_function),
            Const(rng),
            Duplicated(values, shadow),
            Active(T(1.5)),
            Const(true),
        ),
    )
    @test isequal(Array(values), Array(expected))
    @test iszero(Array(shadow))
    @test derivative[4] ≈ reference
end

function _check_enzyme_batch_gradient!(rng, fill_function, next_draw, ::Type{T}) where {T}
    expected, _ = next_draw(rng, T, 17)
    reference = sum(Array(expected))
    values = CUDA.zeros(T, 17)
    shadow_one = CUDA.fill(T(3), 17)
    shadow_two = CUDA.fill(T(4), 17)
    derivative = only(
        autodiff(
            Forward,
            _enzyme_fill_batch_objective!,
            Const(fill_function),
            Const(rng),
            BatchDuplicated(values, (shadow_one, shadow_two)),
            BatchDuplicated(T(1.5), (one(T), T(2))),
            Const(reference),
            Const(true),
        ),
    )
    @test isequal(Array(values), Array(expected))
    @test iszero(Array(shadow_one))
    @test iszero(Array(shadow_two))
    @test derivative[1] ≈ reference
    @test derivative[2] ≈ T(2) * reference
end

@testset "CUDA Enzyme immutable fill rules" begin
    rng = device(Philox4x32(0x65c0))
    for T in (Float32, Float64),
        (fill_function, next_draw, continued) in CUDA_ENZYME_FILL_CASES

        _check_enzyme_primal!(rng, fill_function, next_draw, continued, T)
    end

    for T in (Float32, Float64),
        (fill_function, next_draw, _) in CUDA_ENZYME_EXPONENTIAL_CASES

        _check_enzyme_reverse_gradient!(rng, fill_function, next_draw, T)
        _check_enzyme_batch_gradient!(rng, fill_function, next_draw, T)
    end
end

@testset "CUDA Enzyme fill rules execute the primal once" begin
    rng = device(Philox4x32(0x65c2))
    for T in (Float32, Float64),
        (fill_function, _, continued) in CUDA_ENZYME_EXPONENTIAL_CASES

        values = CUDA.zeros(T, 4096)
        shadow_one = similar(values)
        shadow_two = similar(values)

        primal_events = _device_events() do
            result = fill_function(rng, values)
            if continued
                @test first(result) === values
            else
                @test result === values
            end
        end
        zero_events = _device_events(() -> fill!(shadow_one, zero(T)))
        primal_work = _device_work(primal_events)
        zero_work = _device_work(zero_events)
        @test !isempty(primal_events.kernels)
        @test isempty(primal_events.memsets)
        @test isempty(primal_events.copies)
        @test zero_work > 0
        @test isempty(zero_events.copies)

        forward_events = _device_events() do
            autodiff(
                Forward,
                _enzyme_fill_result!,
                Const,
                Const(fill_function),
                Const(rng),
                Duplicated(values, shadow_one),
                Const(true),
            )
        end
        @test _device_work(forward_events) == primal_work + zero_work
        @test isempty(forward_events.copies)

        reverse_events = _device_events() do
            autodiff(
                Reverse,
                _enzyme_fill_result!,
                Const,
                Const(fill_function),
                Const(rng),
                Duplicated(values, shadow_one),
                Const(true),
            )
        end
        @test _device_work(reverse_events) == primal_work + 2zero_work
        @test isempty(reverse_events.copies)

        batch_events = _device_events() do
            autodiff(
                Forward,
                _enzyme_fill_result!,
                Const,
                Const(fill_function),
                Const(rng),
                BatchDuplicated(values, (shadow_one, shadow_two)),
                Const(true),
            )
        end
        @test _device_work(batch_events) == primal_work + 2zero_work
        @test isempty(batch_events.copies)
    end
end

@testset "CUDA Enzyme fills do not stage through the host" begin
    rng = device(Philox4x32(0x65c1))
    for T in (Float32, Float64), (fill_function, _, _) in CUDA_ENZYME_EXPONENTIAL_CASES

        values = CUDA.zeros(T, 4096)
        shadow = CUDA.ones(T, 4096)
        autodiff(
            Forward,
            _enzyme_fill_result!,
            Const,
            Const(fill_function),
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
                Const(fill_function),
                Const(rng),
                Duplicated(values, shadow),
                Const(true),
            )
            CUDA.synchronize()
        end
        events = _cuda_profile_events(profile)
        @test _device_work(events) > 0
        @test isempty(events.host_to_device)
        @test isempty(events.device_to_host)
        @test iszero(Array(shadow))
    end
end
