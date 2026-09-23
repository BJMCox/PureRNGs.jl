include(joinpath(@__DIR__, "..", "..", "..", "distribution_transform_cases.jl"))

function midpoint_reference_next(rng, ::Type{T}) where {T}
    value = UInt64(0)
    cursor = rng
    for _ = 1:(T===Float32 ? 23 : 52)
        bit, cursor = rand_next(cursor, Bool)
        value = (value << 1) | UInt64(bit)
    end
    scale = T === Float32 ? T(0x1p-24) : T(0x1p-53)
    return T((value << 1) | UInt64(1)) * scale, cursor
end

function six_transform_reference_next(
    rng,
    d::Union{Logistic{T},Cauchy{T},Gumbel{T},Frechet{T}},
) where {T}
    value, next_rng = midpoint_reference_next(rng, T)
    result = d isa Union{Gumbel,Frechet} ? nothing : six_transform_formula(d, value)
    return result, next_rng, value
end

function six_transform_reference_next(rng, d::Union{Pareto,TriangularDist})
    value, next_rng = six_transform_input_next(rng, d)
    return six_transform_formula(d, value), next_rng, value
end

function six_transform_chain(rng, d, count)
    values = Vector{six_transform_result_type(d)}(undef, count)
    cursor = rng
    for index in eachindex(values)
        values[index], cursor = rand_next(cursor, d)
    end
    return values, cursor
end

@testset "six-transform scalar mappings" begin
    rng = Philox4x32(0x9d5)
    for T in (Float32, Float64), d in six_transform_distributions(T)
        expected, expected_next, input = six_transform_reference_next(rng, d)
        value, next_rng = rand_next(rng, d)

        if d isa Union{Logistic,Cauchy,Pareto,TriangularDist}
            @test value === expected
        else
            @test isapprox(cdf(d, value), one(T) - input; rtol = 16eps(T))
        end
        @test rand(rng, d) === value
        @test next_rng === expected_next
    end
end

@testset "six-transform forms preserve order and state" begin
    for T in (Float32, Float64), d in six_transform_distributions(T)
        rng = Philox4x32(0x9d6)
        expected, expected_next = six_transform_chain(rng, d, 5)
        destination = similar(expected)

        @test rand(rng, d, 5) == expected
        @test rand_at(rng, d, 3) === expected[3]
        @test rand!(rng, d, destination; threaded = false) === destination
        @test destination == expected
        returned, next_rng = rand_next!(rng, d, destination; threaded = false)
        @test returned === destination
        @test destination == expected
        @test next_rng === expected_next
    end
end

@testset "six-transform boundaries and preflight" begin
    rng = Philox4x32(0x9d7)
    for T in (Float32, Float64)
        pareto = Pareto(T(2), T(3))
        @test rand(rng, pareto) >= pareto.θ

        for d in (
            TriangularDist(T(0), T(2), T(0)),
            TriangularDist(T(0), T(2), T(2)),
            TriangularDist(T(-floatmax(T) / T(4)), floatmax(T) / T(4), zero(T)),
            TriangularDist(T(2), T(2), T(2)),
        )
            value, expected_next, _ = six_transform_reference_next(rng, d)
            @test rand_next(rng, d) === (value, expected_next)
            @test d.a <= value <= d.b
        end

        triangular = TriangularDist(T(-1.5), T(2.25), T(0.25))
        for index in (1, 2, 3)
            mapped = rand_at(rng, triangular, index)
            @test isapprox(cdf(triangular, mapped), rand_at(rng, T, index); rtol = 8eps(T))
        end
    end

    exhausted = IR._rebuild(rng, IR._terminal64(IR._max_block(rng)), rng.device)
    for d in six_transform_distributions(Float64)
        destination = fill(rand(rng, d), 2)
        original = copy(destination)
        @test_throws StreamExhausted rand!(exhausted, d, destination)
        @test destination == original
    end
end
