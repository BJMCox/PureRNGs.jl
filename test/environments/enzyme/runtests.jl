using Enzyme
using PureRNGs
using Random
using Test

const ER = Enzyme.EnzymeRules

mutable struct CountingVector{T} <: AbstractVector{T}
    data::Vector{T}
    writes::Base.RefValue{Int}
end

CountingVector(data::Vector{T}) where {T} = CountingVector(data, Ref(0))
Base.size(array::CountingVector) = size(array.data)
Base.getindex(array::CountingVector, index::Int) = array.data[index]
function Base.setindex!(array::CountingVector, value, index::Int)
    array.writes[] += 1
    return setindex!(array.data, value, index)
end
PureRNGs.MLDataDevices.get_device(::CountingVector) =
    PureRNGs.MLDataDevices.CPUDevice()
PureRNGs.KernelAbstractions.get_backend(::CountingVector) =
    PureRNGs.KernelAbstractions.CPU()

@testset "R65 Enzyme activity boundary" begin
    @test ER.inactive_type(AbstractPureRNG)
    @test !ER.inactive_type(StatefulRNG)
end

@testset "R65 closed rule surface" begin
    extension = Base.get_extension(PureRNGs, :PureRNGsEnzymeCoreExt)
    fill_functions = (
        Random.rand!,
        Random.randn!,
        Random.randexp!,
        rand_next!,
        randn_next!,
        randexp_next!,
    )
    for rule in (ER.forward, ER.augmented_primal, ER.reverse)
        owned = filter(method -> method.module === extension, methods(rule))
        @test length(owned) == 9
        for fill_function in fill_functions
            annotation = Enzyme.Const{typeof(fill_function)}
            expected = fill_function in (rand_next!, randn_next!, randexp_next!) ? 1 : 2
            @test count(
                method -> Base.unwrap_unionall(method.sig).parameters[3] === annotation,
                owned,
            ) == expected
        end
    end
    @test count(method -> method.module === extension, methods(ER.inactive_type)) == 1
end

function stateful_normal_objective!(rng, destination, scale)
    Random.randn!(rng, destination)
    return scale * sum(destination)
end

function stateful_fill_objective!(fill_function, rng, destination, scale)
    fill_function(rng, destination)
    return scale * sum(destination)
end

function fill_cases(rng, ::Type{T}, count) where {T}
    return (
        (Random.rand!, rand_next(rng, T, count)),
        (Random.randn!, randn_next(rng, T, count)),
        (Random.randexp!, randexp_next(rng, T, count)),
    )
end

function pure_fill_objective!(fill_function, rng, destination, scale, threaded)
    fill_function(rng, destination; threaded = threaded)
    return scale * sum(destination)
end

function pure_next_fill_objective!(fill_function, rng, destination, scale, threaded)
    _, result = fill_function(rng, destination; threaded = threaded)
    return scale * sum(result)
end

function pure_fill_result!(fill_function, rng, destination, threaded)
    return fill_function(rng, destination; threaded = threaded)
end

function pure_default_fill_result!(fill_function, rng, destination)
    return fill_function(rng, destination)
end

function stateful_fill_result!(fill_function, rng, destination)
    return fill_function(rng, destination)
end

function pure_fill_cases(rng, ::Type{T}, count) where {T}
    return (
        (Random.rand!, rand_next!, rand_next(rng, T, count)),
        (Random.randn!, randn_next!, randn_next(rng, T, count)),
        (Random.randexp!, randexp_next!, randexp_next(rng, T, count)),
    )
end


@testset "R65 omitted keyword and StatefulRNG annotations" begin
    rng = Philox4x32(0x6506)
    expected_rng, expected = randn_next(rng, Float64, 9)

    destination = zeros(9)
    shadow = fill(4.0, 9)
    shadow_result, primal_result = autodiff(
        ForwardWithPrimal,
        pure_default_fill_result!,
        Duplicated,
        Const(randn_next!),
        Const(rng),
        Duplicated(destination, shadow),
    )
    @test primal_result[1] === expected_rng
    @test primal_result[2] === destination
    @test shadow_result[2] === shadow
    @test destination == expected
    @test iszero(shadow)

    mutable_rng = StatefulRNG(rng)
    mutable_shadow = copy(mutable_rng)
    bridge_values = zeros(9)
    bridge_shadow = fill(3.0, 9)
    bridge_shadow_result, bridge_primal_result = autodiff(
        ForwardWithPrimal,
        stateful_fill_result!,
        Duplicated,
        Const(Random.randn!),
        Duplicated(mutable_rng, mutable_shadow),
        Duplicated(bridge_values, bridge_shadow),
    )
    @test bridge_primal_result === bridge_values
    @test bridge_shadow_result === bridge_shadow
    @test bridge_values == expected
    @test iszero(bridge_shadow)
    @test parent(mutable_rng) === expected_rng
    @test parent(mutable_shadow) === rng
end

@testset "R65 StatefulRNG reverse normal overwrite" begin
    rng = StatefulRNG(Philox4x32(0x6501))
    expected_rng, expected = randn_next(parent(rng), Float64, 8)
    destination = zeros(8)
    shadow = fill(9.0, 8)
    scale = 1.25

    derivative = only(
        autodiff(
            Reverse,
            stateful_normal_objective!,
            Active,
            Const(rng),
            Duplicated(destination, shadow),
            Active(scale),
        ),
    )

    @test destination == expected
    @test parent(rng) === expected_rng
    @test iszero(shadow)
    @test derivative[3] ≈ sum(expected)
end


@testset "R65 immutable fill rules" begin
    for T in (Float32, Float64), case in pure_fill_cases(Philox4x32(0x6503), T, 13)
        fill_function, next_fill_function, expected = case
        expected_rng, expected_values = expected
        rng = Philox4x32(0x6503)

        for (function_under_test, objective) in (
            (fill_function, pure_fill_objective!),
            (next_fill_function, pure_next_fill_objective!),
        )
            for threaded in (false, true)
                reverse_values = zeros(T, 13)
                reverse_shadow = fill(T(9), 13)
                reverse_derivative = only(
                    autodiff(
                        Reverse,
                        objective,
                        Active,
                        Const(function_under_test),
                        Const(rng),
                        Duplicated(reverse_values, reverse_shadow),
                        Active(T(1.5)),
                        Const(threaded),
                    ),
                )
                @test reverse_values == expected_values
                @test iszero(reverse_shadow)
                @test reverse_derivative[4] ≈ sum(expected_values)

                forward_values = zeros(T, 13)
                forward_shadow = fill(T(7), 13)
                forward_derivative = only(
                    autodiff(
                        Forward,
                        objective,
                        Const(function_under_test),
                        Const(rng),
                        Duplicated(forward_values, forward_shadow),
                        Duplicated(T(1.5), one(T)),
                        Const(threaded),
                    ),
                )
                @test forward_values == expected_values
                @test iszero(forward_shadow)
                @test forward_derivative ≈ sum(expected_values)
            end
        end

        alias_values = zeros(T, 13)
        alias_shadow = fill(T(5), 13)
        shadow_result, primal_result = autodiff(
            ForwardWithPrimal,
            pure_fill_result!,
            Duplicated,
            Const(fill_function),
            Const(rng),
            Duplicated(alias_values, alias_shadow),
            Const(false),
        )
        @test primal_result === alias_values
        @test shadow_result === alias_shadow
        @test alias_values == expected_values
        @test iszero(alias_shadow)

        next_values = zeros(T, 13)
        next_shadow = fill(T(5), 13)
        next_shadow_result, next_primal_result = autodiff(
            ForwardWithPrimal,
            pure_fill_result!,
            Duplicated,
            Const(next_fill_function),
            Const(rng),
            Duplicated(next_values, next_shadow),
            Const(false),
        )
        @test next_primal_result[1] === expected_rng
        @test next_primal_result[2] === next_values
        @test next_shadow_result[2] === next_shadow
        @test next_values == expected_values
        @test iszero(next_shadow)
    end
end


@testset "R65 constant and batched destinations" begin
    for T in (Float32, Float64)
        rng = Philox4x32(0x6504)
        expected_rng, expected = randexp_next(rng, T, 17)

        constant_values = zeros(T, 17)
        constant_derivative = only(
            autodiff(
                Forward,
                pure_next_fill_objective!,
                Const(randexp_next!),
                Const(rng),
                Const(constant_values),
                Duplicated(T(2), one(T)),
                Const(false),
            ),
        )
        @test constant_values == expected
        @test constant_derivative ≈ sum(expected)

        batched_values = zeros(T, 17)
        shadow_one = fill(T(3), 17)
        shadow_two = fill(T(4), 17)
        batched_derivative = only(
            autodiff(
                Forward,
                pure_next_fill_objective!,
                Const(randexp_next!),
                Const(rng),
                BatchDuplicated(batched_values, (shadow_one, shadow_two)),
                BatchDuplicated(T(2), (one(T), T(2))),
                Const(true),
            ),
        )
        @test batched_values == expected
        @test iszero(shadow_one)
        @test iszero(shadow_two)
        @test batched_derivative[1] ≈ sum(expected)
        @test batched_derivative[2] ≈ T(2) * sum(expected)

        next_rng, direct_values =
            randexp_next!(rng, similar(batched_values); threaded = true)
        @test next_rng === expected_rng
        @test direct_values == expected
    end
end


@testset "R65 failed primal preserves shadows" begin
    rng = Philox4x32(0x6505)
    exhausted = PureRNGs._rebuild(
        rng,
        PureRNGs._terminal64(PureRNGs._max_block(rng)),
        rng.device,
    )
    destination = zeros(8)
    shadow = fill(6.0, 8)
    @test_throws ArgumentError autodiff(
        Forward,
        pure_fill_result!,
        Duplicated,
        Const(Random.randn!),
        Const(exhausted),
        Duplicated(destination, shadow),
        Const(false),
    )
    @test destination == zeros(8)
    @test shadow == fill(6.0, 8)
end


@testset "R65 StatefulRNG fill rules" begin
    for T in (Float32, Float64),
        (fill_function, expected) in fill_cases(Philox4x32(0x6502), T, 11)

        expected_rng, expected_values = expected

        reverse_rng = StatefulRNG(Philox4x32(0x6502))
        reverse_values = zeros(T, 11)
        reverse_shadow = fill(T(9), 11)
        reverse_derivative = only(
            autodiff(
                Reverse,
                stateful_fill_objective!,
                Active,
                Const(fill_function),
                Const(reverse_rng),
                Duplicated(reverse_values, reverse_shadow),
                Active(T(1.25)),
            ),
        )
        @test reverse_values == expected_values
        @test parent(reverse_rng) === expected_rng
        @test iszero(reverse_shadow)
        @test reverse_derivative[4] ≈ sum(expected_values)

        forward_rng = StatefulRNG(Philox4x32(0x6502))
        forward_values = zeros(T, 11)
        forward_shadow = fill(T(7), 11)
        forward_derivative = only(
            autodiff(
                Forward,
                stateful_fill_objective!,
                Const(fill_function),
                Const(forward_rng),
                Duplicated(forward_values, forward_shadow),
                Duplicated(T(1.25), one(T)),
            ),
        )
        @test forward_values == expected_values
        @test parent(forward_rng) === expected_rng
        @test iszero(forward_shadow)
        @test forward_derivative ≈ sum(expected_values)
    end
end


@testset "R65 one primal execution" begin
    rng = Philox4x32(0x6507)
    _, expected = randn_next(rng, Float64, 10)
    for (mode, shadow_writes) in ((Forward, 10), (Reverse, 20))
        destination = CountingVector(zeros(10))
        shadow = CountingVector(fill(5.0, 10))
        autodiff(
            mode,
            pure_fill_result!,
            Const,
            Const(Random.randn!),
            Const(rng),
            Duplicated(destination, shadow),
            Const(false),
        )
        @test destination.data == expected
        @test destination.writes[] == 10
        @test iszero(shadow.data)
        @test shadow.writes[] == shadow_writes
    end
end
