function expanded_distributions(::Type{T}) where {T}
    return (
        LogNormal(T(0.25), T(0.75)),
        Weibull(T(1.75), T(0.75)),
        Rayleigh(T(0.75)),
        Laplace(T(0.25), T(0.75)),
    )
end

function expanded_primitive_next(rng, d::LogNormal{T}) where {T}
    z, next_rng = randn_next(rng, T)
    return exp(fma(d.σ, z, d.μ)), next_rng
end

function expanded_primitive_next(rng, d::Weibull{T}) where {T}
    x, next_rng = randexp_next(rng, T)
    return d.θ * x^inv(d.α), next_rng
end

function expanded_primitive_next(rng, d::Rayleigh{T}) where {T}
    x, next_rng = randexp_next(rng, T)
    return d.σ * sqrt(T(2) * x), next_rng
end

function expanded_primitive_next(rng, d::Laplace{T}) where {T}
    x, after_exp = randexp_next(rng, T)
    positive, next_rng = rand_next(after_exp, Bool)
    return fma(ifelse(positive, d.θ, -d.θ), x, d.μ), next_rng
end

function expanded_primitive_chain(rng, distribution, count)
    values = Vector{typeof(first(expanded_primitive_next(rng, distribution)))}(undef, count)
    cursor = rng
    for index in eachindex(values)
        values[index], cursor = expanded_primitive_next(cursor, distribution)
    end
    return cursor, values
end

@testset "expanded fixed distribution mappings" begin
    root = Philox4x32(0x9d1)
    rng = last(rand_next(root, Bool))

    for T in (Float32, Float64), distribution in expanded_distributions(T)
        expected_scalar, expected_scalar_next = expanded_primitive_next(rng, distribution)
        expected_next, expected = expanded_primitive_chain(rng, distribution, 33)

        @test rand(rng, distribution) === expected_scalar
        actual, actual_next = rand_next(rng, distribution)
        @test actual === expected_scalar
        @test actual_next === expected_scalar_next

        @test [randat(rng, distribution, index) for index in eachindex(expected)] == expected

        allocated = rand(rng, distribution, length(expected))
        @test allocated == expected

        continued, allocated_next = rand_next(rng, distribution, length(expected))
        @test continued == expected
        @test allocated_next === expected_next

        destination = Vector{T}(undef, length(expected))
        @test rand!(rng, distribution, destination; threaded = false) === destination
        @test destination == expected

        returned, filled_next = rand_next!(rng, distribution, destination; threaded = false)
        @test returned === destination
        @test destination == expected
        @test filled_next === expected_next
    end
end

@testset "expanded LogNormal degeneracy preserves its normal span" begin
    root = Philox4x32(0x9d2)
    rng = last(rand_next(root, Bool))

    for T in (Float32, Float64)
        distribution = LogNormal(T(0.25), zero(T))
        expected, expected_next = expanded_primitive_next(rng, distribution)
        actual, actual_next = rand_next(rng, distribution)
        @test actual === expected
        @test actual_next === expected_next
    end
end

@testset "expanded Laplace zero count and preflight are atomic" begin
    distribution = Laplace(0.25, 0.75)
    rng = Philox4x32(0x9d3)
    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    empty = Vector{Float64}(undef, 0)

    @test rand!(exhausted, distribution, empty) === empty
    returned, empty_next = rand_next!(exhausted, distribution, empty)
    @test returned === empty
    @test empty_next === exhausted
    @test isempty(rand(exhausted, distribution, 0))
    allocated, allocated_next = rand_next(exhausted, distribution, 0)
    @test isempty(allocated)
    @test allocated_next === exhausted

    last = IR._rebuild(
        rng,
        IR._Position64(IR._max_block(rng), IR._block_bits(rng) - 54),
        rng.device,
    )
    destination = fill(17.0, 2)
    original = copy(destination)
    @test_throws ArgumentError rand!(last, distribution, destination)
    @test destination == original
end

@testset "expanded fixed distributions validate their parameters" begin
    rng = Philox4x32(0x9d4)
    invalid = (
        LogNormal(Inf, 1.0; check_args = false),
        Weibull(0.0, 1.0; check_args = false),
        Rayleigh(Inf; check_args = false),
        Laplace(0.0, 0.0; check_args = false),
    )

    for distribution in invalid
        @test_throws ArgumentError rand(rng, distribution)
    end

    @test_throws ArgumentError rand_next!(rng, last(invalid), Float64[])

    @test_throws MethodError rand(rng, LogNormal(Float16(0), Float16(1)))
end
