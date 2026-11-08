using Random

function _audit_error(call)
    try
        call()
        return nothing
    catch error
        return error
    end
end

@testset "Random calls require a mutable generator" begin
    for F in (Philox4x32, Threefry2x64, ChaCha)
        root = F(0xa71)
        for (draw, next, fill!, next!) in (
            (rand, rand_next, rand!, rand_next!),
            (randn, randn_next, randn!, randn_next!),
            (randexp, randexp_next, randexp!, randexp_next!),
        )
            for args in ((), (Float32,), (Float64, 4), (Float32, (2, 2)))
                @test_throws MethodError draw(root, args...)
                expected, expected_rng = next(root, args...)
                bridge = StatefulRNG(root)
                @test draw(bridge, args...) == expected
                @test parent(bridge) === expected_rng
            end
            destination = zeros(4)
            @test_throws MethodError fill!(root, destination)
            @test destination == zeros(4)
            expected, expected_rng = next!(root, destination)
            bridge = StatefulRNG(root)
            @test fill!(bridge, zeros(4)) == expected
            @test parent(bridge) === expected_rng
        end
        for args in ((1:6,), ([:a, :b],), ((2, 3),), (4,), ((2, 3), 4))
            @test_throws MethodError rand(root, args...)
        end
        @test randgen!!(root, Float64, 4) == rand_next(root, Float64, 4)
    end
end

@testset "assigned constants" begin
    @test IR._NARROW_SPLIT_COUNT === UInt64(0xffffffff)
end

@testset "implemented deterministic error closure" begin
    rng = Philox4x32(0xa72)
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    wrong_uniform_destination = WrongDeviceArray(Vector{UInt32}(undef, 1))
    wrong_normal_destination = WrongDeviceArray(Vector{Float32}(undef, 1))
    wrong_exponential_destination = WrongDeviceArray(Vector{Float32}(undef, 1))
    argument_errors = (
        (:negative_seed, () -> Philox2x32(-1)),
        (:oversized_seed, () -> Philox2x32(big(1) << 32)),
        (:negative_split, () -> splitrng(rng, -1)),
        (:invalid_static_split, () -> splitrng(rng, Val(UInt32(1)))),
        (:narrow_split_namespace, () -> splitrng(Philox2x32(1), UInt64(0x1_0000_0000))),
        (:rand_at_index, () -> rand_at(rng, UInt32, 0)),
        (:randn_at_index, () -> randn_at(rng, Float32, 0)),
        (:randexp_at_index, () -> randexp_at(rng, Float32, 0)),
        (:empty_range, () -> first(rand_next(rng, UInt8(2):UInt8(1)))),
        (:empty_range_continuation, () -> rand_next(rng, UInt8(2):UInt8(1))),
        (:negative_uniform_dimension, () -> first(rand_next(rng, UInt32, -1))),
        (:negative_uniform_continuation_dimension, () -> rand_next(rng, UInt32, -1)),
        (:negative_normal_dimension, () -> first(randn_next(rng, Float32, -1))),
        (:negative_normal_continuation_dimension, () -> randn_next(rng, Float32, -1)),
        (:negative_exponential_dimension, () -> first(randexp_next(rng, Float32, -1))),
        (
            :negative_exponential_continuation_dimension,
            () -> randexp_next(rng, Float32, -1),
        ),
        (:uniform_device_mismatch, () -> first(rand_next!(rng, wrong_uniform_destination))),
        (
            :uniform_continuation_device_mismatch,
            () -> rand_next!(rng, wrong_uniform_destination),
        ),
        (:normal_device_mismatch, () -> first(randn_next!(rng, wrong_normal_destination))),
        (
            :normal_continuation_device_mismatch,
            () -> randn_next!(rng, wrong_normal_destination),
        ),
        (
            :exponential_device_mismatch,
            () -> first(randexp_next!(rng, wrong_exponential_destination)),
        ),
        (
            :exponential_continuation_device_mismatch,
            () -> randexp_next!(rng, wrong_exponential_destination),
        ),
    )
    exhausted_errors = (
        (:rand_at_capacity, () -> rand_at(exhausted, UInt32, 1)),
        (:randn_at_capacity, () -> randn_at(exhausted, Float32, 1)),
        (:randexp_at_capacity, () -> randexp_at(exhausted, Float32, 1)),
        (:pure_capacity, () -> first(rand_next(exhausted, UInt32))),
        (:continuation_capacity, () -> rand_next(exhausted, UInt32)),
        (:fill_capacity, () -> first(rand_next!(exhausted, Vector{UInt32}(undef, 1)))),
        (
            :continuation_fill_capacity,
            () -> rand_next!(exhausted, Vector{UInt32}(undef, 1)),
        ),
        (:allocating_capacity, () -> first(rand_next(exhausted, UInt32, 1))),
        (:continuation_allocating_capacity, () -> rand_next(exhausted, UInt32, 1)),
        (:normal_pure_capacity, () -> first(randn_next(exhausted, Float32))),
        (:normal_continuation_capacity, () -> randn_next(exhausted, Float32)),
        (
            :normal_fill_capacity,
            () -> first(randn_next!(exhausted, Vector{Float32}(undef, 1))),
        ),
        (
            :normal_continuation_fill_capacity,
            () -> randn_next!(exhausted, Vector{Float32}(undef, 1)),
        ),
        (:normal_allocating_capacity, () -> first(randn_next(exhausted, Float32, 1))),
        (:normal_continuation_allocating_capacity, () -> randn_next(exhausted, Float32, 1)),
        (:exponential_pure_capacity, () -> first(randexp_next(exhausted, Float32))),
        (:exponential_continuation_capacity, () -> randexp_next(exhausted, Float32)),
        (
            :exponential_fill_capacity,
            () -> first(randexp_next!(exhausted, Vector{Float32}(undef, 1))),
        ),
        (
            :exponential_continuation_fill_capacity,
            () -> randexp_next!(exhausted, Vector{Float32}(undef, 1)),
        ),
        (
            :exponential_allocating_capacity,
            () -> first(randexp_next(exhausted, Float32, 1)),
        ),
        (
            :exponential_continuation_allocating_capacity,
            () -> randexp_next(exhausted, Float32, 1),
        ),
        (:range_pure_capacity, () -> first(rand_next(exhausted, UInt8(1):UInt8(2)))),
        (:range_continuation_capacity, () -> rand_next(exhausted, UInt8(1):UInt8(2))),
        (
            :range_allocating_capacity,
            () -> first(rand_next(exhausted, UInt8(1):UInt8(2), 1)),
        ),
        (
            :range_continuation_allocating_capacity,
            () -> rand_next(exhausted, UInt8(1):UInt8(2), 1),
        ),
    )
    method_errors = (
        (:uniform_result_type, () -> first(rand_next(rng, BigFloat))),
        (:uniform_continuation_result_type, () -> rand_next(rng, BigFloat)),
        (
            :uniform_destination_type,
            () -> first(rand_next!(rng, Vector{BigFloat}(undef, 1))),
        ),
        (:normal_result_type, () -> first(randn_next(rng, BigFloat))),
        (:normal_continuation_result_type, () -> randn_next(rng, BigFloat)),
        (
            :normal_destination_type,
            () -> first(randn_next!(rng, Vector{BigFloat}(undef, 1))),
        ),
        (:exponential_result_type, () -> first(randexp_next(rng, ComplexF64))),
        (:exponential_continuation_result_type, () -> randexp_next(rng, ComplexF64)),
        (
            :exponential_destination_type,
            () -> first(randexp_next!(rng, Vector{ComplexF64}(undef, 1))),
        ),
    )

    for (expected, cases) in (
        (ArgumentError, argument_errors),
        (StreamExhausted, exhausted_errors),
        (MethodError, method_errors),
    )
        for (name, call) in cases
            @testset "$name" begin
                # StreamExhausted is parametric, so compare by subtyping.
                @test typeof(_audit_error(call)) <: expected
            end
        end
    end

end

@testset "StreamExhausted payload" begin
    base = Philox4x32(0x970)
    near_end = IR._rebuild(
        base,
        _position_from_absolute(base, _stream_capacity(base) - 10),
        base.device,
    )
    population = Int32[2, 3, 5, 7]
    for (name, span, call) in (
        (:scalar, UInt128(32), () -> first(rand_next(near_end, UInt32))),
        (:fill, UInt128(64), () -> first(rand_next!(near_end, Vector{UInt32}(undef, 2)))),
        (:addressed, UInt128(64), () -> rand_at(near_end, UInt32, 2)),
        (:sampling, UInt128(128), () -> randsample(near_end, population, 2)),
    )
        @testset "$name" begin
            exhausted = _audit_error(call)
            @test exhausted isa StreamExhausted{typeof(near_end)}
            @test exhausted.bits === span
            @test fieldcount(typeof(exhausted)) == 1
            @test occursin("Philox4x32", sprint(showerror, exhausted))
        end
    end

    @test_throws ArgumentError Philox4x32(rngkey(near_end), rngposition(near_end) + 11)
end
