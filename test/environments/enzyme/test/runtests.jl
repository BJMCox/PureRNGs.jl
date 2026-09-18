using Distributions
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
PureRNGs.MLDataDevices.get_device(::CountingVector) = PureRNGs.MLDataDevices.CPUDevice()
PureRNGs.KernelAbstractions.get_backend(::CountingVector) =
    PureRNGs.KernelAbstractions.CPU()

# An extension is not a submodule of its parent, so a recursive scan that starts
# at PureRNGs never reaches it. Scan each loaded extension itself.
@testset "R1 extension ambiguities" begin
    for name in (:PureRNGsEnzymeCoreExt, :PureRNGsDistributionsExt)
        extension = Base.get_extension(PureRNGs, name)
        @testset "$name" begin
            @test isempty(Test.detect_ambiguities(extension; recursive = true))
        end
    end
end

@testset "R65 Enzyme activity boundary" begin
    @test ER.inactive_type(AbstractPureRNG)
    @test !ER.inactive_type(StatefulRNG)
end

@testset "R65 closed rule surface" begin
    extension = Base.get_extension(PureRNGs, :PureRNGsEnzymeCoreExt)
    # rand! carries the immutable, StatefulRNG, distribution and range fills;
    # rand_next! carries all but the StatefulRNG one.
    expected_counts = (
        (Random.rand!, 4),
        (Random.randn!, 2),
        (Random.randexp!, 2),
        (rand_next!, 3),
        (randn_next!, 1),
        (randexp_next!, 1),
        (randsample!, 2),
        (randsample_next!, 2),
    )
    for rule in (ER.forward, ER.augmented_primal, ER.reverse)
        owned = filter(method -> method.module === extension, methods(rule))
        @test length(owned) == sum(last, expected_counts)
        for (fill_function, expected) in expected_counts
            annotation = Enzyme.Const{typeof(fill_function)}
            @test count(
                method -> Base.unwrap_unionall(method.sig).parameters[3] === annotation,
                owned,
            ) == expected
        end
    end
    @test count(method -> method.module === extension, methods(ER.inactive_type)) == 1
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
    result, _ = fill_function(rng, destination; threaded = threaded)
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
    expected, expected_rng = randn_next(rng, Float64, 9)

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
    @test primal_result[1] === destination
    @test primal_result[2] === expected_rng
    @test shadow_result[1] === shadow
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
    expected, expected_rng = randn_next(parent(rng), Float64, 8)
    destination = zeros(8)
    shadow = fill(9.0, 8)
    scale = 1.25

    derivative = only(
        autodiff(
            Reverse,
            stateful_fill_objective!,
            Active,
            Const(Random.randn!),
            Const(rng),
            Duplicated(destination, shadow),
            Active(scale),
        ),
    )

    @test destination == expected
    @test parent(rng) === expected_rng
    @test iszero(shadow)
    @test derivative[4] ≈ sum(expected)
end


@testset "R65 immutable fill rules" begin
    for T in (Float32, Float64), case in pure_fill_cases(Philox4x32(0x6503), T, 13)
        fill_function, next_fill_function, expected = case
        expected_values, expected_rng = expected
        rng = Philox4x32(0x6503)

        for (function_under_test, continued) in
            ((fill_function, false), (next_fill_function, true))
            values = zeros(T, 13)
            shadow = fill(T(5), 13)
            shadow_result, primal_result = autodiff(
                ForwardWithPrimal,
                pure_fill_result!,
                Duplicated,
                Const(function_under_test),
                Const(rng),
                Duplicated(values, shadow),
                Const(continued),
            )
            if continued
                @test primal_result[1] === values
                @test primal_result[2] === expected_rng
                @test shadow_result[1] === shadow
            else
                @test primal_result === values
                @test shadow_result === shadow
            end
            @test values == expected_values
            @test iszero(shadow)
        end
    end

    for T in (Float32, Float64),
        (fill_function, objective) in (
            (Random.randexp!, pure_fill_objective!),
            (randexp_next!, pure_next_fill_objective!),
        )

        rng = Philox4x32(0x6503)
        expected, _ = randexp_next(rng, T, 13)
        values = zeros(T, 13)
        shadow = fill(T(9), 13)
        derivative = only(
            autodiff(
                Reverse,
                objective,
                Active,
                Const(fill_function),
                Const(rng),
                Duplicated(values, shadow),
                Active(T(1.5)),
                Const(true),
            ),
        )
        @test values == expected
        @test iszero(shadow)
        @test derivative[4] ≈ sum(expected)
    end
end


@testset "R65 constant and batched destinations" begin
    for T in (Float32, Float64)
        rng = Philox4x32(0x6504)
        expected, expected_rng = randexp_next(rng, T, 17)

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

        direct_values, next_rng =
            randexp_next!(rng, similar(batched_values); threaded = true)
        @test next_rng === expected_rng
        @test direct_values == expected
    end
end


@testset "R65 failed primal preserves shadows" begin
    rng = Philox4x32(0x6505)
    exhausted =
        PureRNGs._rebuild(rng, PureRNGs._terminal64(PureRNGs._max_block(rng)), rng.device)
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

        expected_values, expected_rng = expected

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


function argument_fill_result!(fill_function, rng, argument, destination, threaded)
    return fill_function(rng, argument, destination; threaded = threaded)
end

function argument_fill_objective!(
    fill_function,
    rng,
    argument,
    destination,
    scale,
    threaded,
)
    fill_function(rng, argument, destination; threaded = threaded)
    return scale * sum(destination)
end

function weighted_fill_result!(
    fill_function,
    rng,
    population,
    weights,
    destination,
    threaded,
)
    return fill_function(rng, population, weights, destination; threaded = threaded)
end

function weighted_fill_objective!(
    fill_function,
    rng,
    population,
    weights,
    destination,
    scale,
    threaded,
)
    fill_function(rng, population, weights, destination; threaded = threaded)
    return scale * sum(destination)
end


@testset "R65 distribution and population fills carry no tangent" begin
    population = [2.0, 3.0, 5.0, 7.0]
    weights = [0.1, 0.2, 0.3, 0.4]
    # Each shadow is nonzero, so without a rule the fill would propagate it.
    cases = (
        (Random.rand!, rand_next!, Normal(0.0, 1.0), Normal(1.0, 0.0)),
        (randsample!, randsample_next!, population, fill(1.0, 4)),
    )

    for (fill_function, next_fill_function, argument, argument_shadow) in cases,
        (function_under_test, continued) in
        ((fill_function, false), (next_fill_function, true))

        rng = Philox4x32(0x6508)
        expected, expected_rng = next_fill_function(rng, argument, zeros(12))

        values = zeros(12)
        shadow = fill(5.0, 12)
        shadow_result, primal_result = autodiff(
            ForwardWithPrimal,
            argument_fill_result!,
            Duplicated,
            Const(function_under_test),
            Const(rng),
            Duplicated(argument, argument_shadow),
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
        @test values == expected
        @test iszero(shadow)

        reverse_values = zeros(12)
        reverse_shadow = fill(9.0, 12)
        derivative = only(
            autodiff(
                Reverse,
                argument_fill_objective!,
                Active,
                Const(function_under_test),
                Const(rng),
                Const(argument),
                Duplicated(reverse_values, reverse_shadow),
                Active(1.5),
                Const(true),
            ),
        )
        @test reverse_values == expected
        @test iszero(reverse_shadow)
        @test derivative[5] ≈ sum(expected)
    end

    for (function_under_test, continued) in ((randsample!, false), (randsample_next!, true))

        rng = Philox4x32(0x6509)
        expected, expected_rng = randsample_next!(rng, population, weights, zeros(12))

        values = zeros(12)
        shadow = fill(5.0, 12)
        shadow_result, primal_result = autodiff(
            ForwardWithPrimal,
            weighted_fill_result!,
            Duplicated,
            Const(function_under_test),
            Const(rng),
            Duplicated(population, fill(1.0, 4)),
            Duplicated(weights, fill(1.0, 4)),
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
        @test values == expected
        @test iszero(shadow)

        reverse_values = zeros(12)
        reverse_shadow = fill(9.0, 12)
        derivative = only(
            autodiff(
                Reverse,
                weighted_fill_objective!,
                Active,
                Const(function_under_test),
                Const(rng),
                Const(population),
                Const(weights),
                Duplicated(reverse_values, reverse_shadow),
                Active(1.5),
                Const(true),
            ),
        )
        @test reverse_values == expected
        @test iszero(reverse_shadow)
        @test derivative[6] ≈ sum(expected)
    end

    active_rng = Philox4x32(0x650a)
    active_expected, _ = rand_next!(active_rng, Normal(0.0, 1.0), zeros(12))
    active_values = zeros(12)
    active_shadow = fill(9.0, 12)
    active_derivative = only(
        autodiff(
            Reverse,
            argument_fill_objective!,
            Active,
            Const(Random.rand!),
            Const(active_rng),
            Active(Normal(0.0, 1.0)),
            Duplicated(active_values, active_shadow),
            Active(1.5),
            Const(true),
        ),
    )
    @test active_values == active_expected
    @test iszero(active_shadow)
    @test active_derivative[3] === Normal{Float64}(0.0, 0.0)
    @test active_derivative[5] ≈ sum(active_expected)

    bernoulli = Bernoulli(0.3)
    rng = Philox4x32(0x650b)
    expected, _ = rand_next!(rng, bernoulli, Vector{Bool}(undef, 12))
    values = Vector{Bool}(undef, 12)
    derivative = only(
        autodiff(
            Forward,
            argument_fill_objective!,
            Const(Random.rand!),
            Const(rng),
            Const(bernoulli),
            Const(values),
            Duplicated(2.0, 1.0),
            Const(true),
        ),
    )
    @test values == expected
    @test derivative ≈ sum(expected)
end


function range_fill_result!(fill_function, rng, destination, range, threaded)
    return fill_function(rng, destination, range; threaded = threaded)
end


@testset "R65 range fill keeps its destination second" begin
    rng = Philox4x32(0x650c)
    expected, _ = rand_next!(rng, Vector{Int}(undef, 8), 3:9)
    for mode in (Forward, Reverse)
        values = Vector{Int}(undef, 8)
        autodiff(
            mode,
            range_fill_result!,
            Const,
            Const(Random.rand!),
            Const(rng),
            Const(values),
            Const(3:9),
            Const(true),
        )
        @test values == expected
    end
end


@testset "R65 one primal execution" begin
    rng = Philox4x32(0x6507)
    expected, _ = randn_next(rng, Float64, 10)
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
