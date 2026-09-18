include(joinpath(@__DIR__, "..", "..", "..", "distribution_transform_cases.jl"))

# Keep the formula's input expression identical: XLA can reassociate log of
# equivalent midpoint expressions differently. Eager public bits check its value.
six_transform_midpoint(rng::PureRNGs._ReactantRNG, ::Type{T}) where {T} =
    PureRNGs._midpoint_value(rng, T)

function _six_transform_snapshot(rng, d)
    input, expected_next = six_transform_input_next(rng, d)
    _, second = six_transform_input_next(expected_next, d)
    addressed_input, _ = six_transform_input_next(second, d)
    value, next_rng = rand_next(rng, d)
    return (
        input,
        rand(rng, d),
        value,
        six_transform_formula(d, input, muladd),
        next_rng,
        expected_next,
        rand_at(rng, d, 3),
        six_transform_formula(d, addressed_input, muladd),
    )
end

@testset "R64 six native compiled mappings" begin
    for T in (Float32, Float64), d in six_transform_distributions(T)
        first_rng = last(rand_next(Philox4x32(0x654), Bool))
        first_carrier = Reactant.to_rarray(first_rng)
        compiled = Reactant.@compile sync = true _six_transform_snapshot(first_carrier, d)
        for eager in (first_rng, last(rand_next(Philox4x32(0x987), UInt32)))
            carrier = Reactant.to_rarray(eager)
            got = compiled(carrier, d)
            input, expected_next = six_transform_input_next(eager, d)
            @test d isa Pareto ? _same_transform_value(got[1], input) :
                  _same_value(got[1], input)
            @test _same_value(got[2], got[4]) && _same_value(got[3], got[4])
            @test _same_value(got[7], got[8])
            @test _same_value(got[5], expected_next) && _same_value(got[6], expected_next)
            @test _same_value(compiled(carrier, d), got)
            continued = compiled(got[5], d)
            _, after_second = six_transform_input_next(expected_next, d)
            @test _same_value(continued[5], after_second)
        end
    end
end

function _triangular_compiled_boundary(u, d)
    return REACTANT_DISTRIBUTIONS_EXT._map_primitive(d, u)
end

@testset "R64 compiled triangular boundaries" begin
    for T in (Float32, Float64),
        d in (
            TriangularDist(T(0), T(2), T(0)),
            TriangularDist(T(0), T(2), T(2)),
            TriangularDist(T(2), T(2), T(2)),
        )

        input = Reactant.to_rarray(zero(T))
        compiled = Reactant.@compile sync = true _triangular_compiled_boundary(input, d)
        @test T(compiled(input, d)) === d.a
        middle = T(compiled(Reactant.to_rarray(T(0.5)), d))
        @test d.a <= middle <= d.b
    end
end
