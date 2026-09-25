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

# An extension is not a submodule of its parent, so a recursive scan that starts
# at PureRNGs never reaches it. Scan each loaded extension itself.
@testset "extension ambiguities" begin
    for name in (:PureRNGsEnzymeCoreExt, :PureRNGsDistributionsExt)
        extension = Base.get_extension(PureRNGs, name)
        @testset "$name" begin
            @test isempty(Test.detect_ambiguities(extension; recursive = true))
        end
    end
end

@testset "Enzyme activity boundary" begin
    @test ER.inactive_type(AbstractPureRNG)
    @test !ER.inactive_type(StatefulRNG)
end

@testset "closed rule surface" begin
    extension = Base.get_extension(PureRNGs, :PureRNGsEnzymeCoreExt)
    # rand! carries the immutable, StatefulRNG, and range fills; rand_next!
    # carries all but the StatefulRNG one. Distribution and population fills have
    # no rule, and the scheduler rule stops threaded fills under differentiation.
    # The two Gamma primitives carry the implicit shape derivative, with a second
    # reverse method for a constant result.
    expected_counts = (
        (Random.rand!, 3),
        (Random.randn!, 2),
        (Random.randexp!, 2),
        (rand_next!, 2),
        (randn_next!, 1),
        (randexp_next!, 1),
        (PureRNGs._run_chunks, 1),
        (PureRNGs._gamma_value, 1),
        (PureRNGs._gamma_log_value, 1),
    )
    constant_reverse = (PureRNGs._gamma_value, PureRNGs._gamma_log_value)
    for rule in (ER.forward, ER.augmented_primal, ER.reverse)
        owned = filter(method -> method.module === extension, methods(rule))
        extra(fill_function) = rule === ER.reverse && fill_function in constant_reverse
        @test length(owned) == sum(((f, n),) -> n + extra(f), expected_counts)
        for (fill_function, expected) in expected_counts
            annotation = Enzyme.Const{typeof(fill_function)}
            @test count(
                method -> Base.unwrap_unionall(method.sig).parameters[3] === annotation,
                owned,
            ) == expected + extra(fill_function)
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


@testset "omitted keyword and StatefulRNG annotations" begin
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

@testset "StatefulRNG reverse normal overwrite" begin
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


@testset "immutable fill rules" begin
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


@testset "constant and batched destinations" begin
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


@testset "failed primal preserves shadows" begin
    rng = Philox4x32(0x6505)
    exhausted =
        PureRNGs._rebuild(rng, PureRNGs._terminal64(PureRNGs._max_block(rng)), rng.device)
    destination = zeros(8)
    shadow = fill(6.0, 8)
    @test_throws StreamExhausted autodiff(
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


@testset "StatefulRNG fill rules" begin
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


function scalar_normal_sum(rng, mu, sigma, count)
    total = 0.0
    for _ = 1:count
        value, rng = rand_next(rng, Normal(mu, sigma))
        total += value
    end
    return total
end

fill_normal_sum(rng, mu, sigma, count) =
    sum(first(rand_next!(rng, Normal(mu, sigma), zeros(count))))

fill_laplace_sum(rng, mu, count) =
    sum(first(rand_next!(rng, Laplace(mu, 1.0), zeros(count))))

fill_uniform_sum(rng, upper, count) =
    sum(first(rand_next!(rng, Uniform(0.0, upper), zeros(count))))

sample_square_sum(rng, population, count) =
    sum(abs2, first(randsample_next!(rng, population, zeros(count))))

weighted_sample_sum(rng, population, weights, count) =
    sum(first(randsample_next!(rng, population, weights, zeros(count))))

threaded_normal_sum(rng, mu, count) =
    sum(first(rand_next!(rng, Normal(mu, 1.0), zeros(count); threaded = true)))

@testset "distribution and population fills give pathwise gradients" begin
    rng = Philox4x32(0x6508)
    n = 12
    standard = first(rand_next(rng, Normal(0.0, 1.0), n))

    # A fill and the equivalent chain of scalar draws read the same bits, so
    # both differentiate to d/dmu = n and d/dsigma = sum of the standard draws.
    scalar = autodiff(
        Reverse,
        scalar_normal_sum,
        Active,
        Const(rng),
        Active(0.5),
        Active(2.0),
        Const(n),
    )
    filled = autodiff(
        Reverse,
        fill_normal_sum,
        Active,
        Const(rng),
        Active(0.5),
        Active(2.0),
        Const(n),
    )
    @test filled[1][2] ≈ scalar[1][2] ≈ n
    @test filled[1][3] ≈ scalar[1][3] ≈ sum(standard)
    @test only(
        autodiff(
            Forward,
            fill_normal_sum,
            Const(rng),
            Duplicated(0.5, 1.0),
            Const(2.0),
            Const(n),
        ),
    ) ≈ n

    @test autodiff(Reverse, fill_laplace_sum, Active, Const(rng), Active(0.5), Const(n))[1][2] ≈
          n
    unit = first(rand_next(rng, Uniform(0.0, 1.0), n))
    @test autodiff(Reverse, fill_uniform_sum, Active, Const(rng), Active(2.0), Const(n))[1][2] ≈
          sum(unit)

    # A gather is linear in the population: each element's gradient is twice its
    # value times the number of draws that picked it.
    population = [2.0, 3.0, 5.0, 7.0]
    draws = first(randsample_next(rng, population, n))
    gradient = zeros(4)
    autodiff(
        Reverse,
        sample_square_sum,
        Active,
        Const(rng),
        Duplicated(population, gradient),
        Const(n),
    )
    @test gradient ≈ [2 * value * count(==(value), draws) for value in population]

    # The drawn indices are piecewise constant in the weights, so only the
    # population carries a gradient.
    weights = [0.1, 0.2, 0.3, 0.4]
    weighted_draws = first(randsample_next(rng, population, weights, n))
    population_gradient = zeros(4)
    weight_gradient = zeros(4)
    autodiff(
        Reverse,
        weighted_sample_sum,
        Active,
        Const(rng),
        Duplicated(population, population_gradient),
        Duplicated(weights, weight_gradient),
        Const(n),
    )
    @test population_gradient == [count(==(value), weighted_draws) for value in population]
    @test iszero(weight_gradient)
end

threaded_half_fill(rng, scale, count) =
    scale * Float32(sum(randn!(rng, zeros(Float16, count); threaded = true)))

@testset "threaded primitive fills of every float type stay differentiable" begin
    rng = Philox4x32(0x650a)
    n =
        8 *
        PureRNGs._fill_chunk_elements(PureRNGs._NormalCodec(PureRNGs._CPU_BACKEND), Float16)
    values = randn(rng, Float16, n)
    derivative =
        autodiff(Reverse, threaded_half_fill, Active, Const(rng), Active(2.0f0), Const(n))
    @test derivative[1][2] == Float32(sum(values))
end

@testset "threaded fills under differentiation throw instead of crashing" begin
    rng = Philox4x32(0x6509)
    n = 8 * PureRNGs._fill_chunk_elements(Val(:uniform), Float64)
    @test_throws ArgumentError autodiff(
        Reverse,
        threaded_normal_sum,
        Active,
        Const(rng),
        Active(0.5),
        Const(n),
    )
end


function range_fill_result!(fill_function, rng, destination, range, threaded)
    return fill_function(rng, destination, range; threaded = threaded)
end


@testset "range fill keeps its destination second" begin
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


@testset "one primal execution" begin
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

# A Gamma draw moves with its shape along the inverse CDF at the draw's
# probability, so Enzyme's rule gives -dF/dshape / f instead of differentiating
# the rejection test; Beta composes it through its two Gamma draws.
@testset "Gamma shape gradients follow the implicit shape derivative" begin
    rng = Philox4x32(0xb01, 3)
    oracle(shape, x; h = 1e-6) = begin
        u = cdf(Gamma(shape), x)
        (quantile(Gamma(shape + h), u) - quantile(Gamma(shape - h), u)) / 2h
    end
    gamma_draw(q) = rand(rng, Gamma(q, 2.0))
    expected = 2 * oracle(2.5, rand(rng, Gamma(2.5)))
    @test only(only(autodiff(Reverse, gamma_draw, Active, Active(2.5)))) ≈ expected rtol =
        1e-6
    @test only(autodiff(Forward, gamma_draw, Duplicated(2.5, 1.0))) ≈ expected rtol = 1e-6
    second = PureRNGs._rebuild(
        rng,
        PureRNGs._advance_position_unchecked(rng, UInt64(17 * 52), UInt64(0)),
        rng.device,
    )
    x, y = rand(rng, Gamma(0.3)), rand(second, Gamma(0.4))
    beta_draw(q) = rand(rng, Beta(q, 0.4))
    @test only(only(autodiff(Reverse, beta_draw, Active, Active(0.3)))) ≈
          oracle(0.3, x) * y / (x + y)^2 rtol = 1e-6
end
