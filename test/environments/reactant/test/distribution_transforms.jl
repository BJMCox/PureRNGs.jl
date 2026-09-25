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

@testset "six native compiled mappings" begin
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

@testset "compiled triangular boundaries" begin
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

# A compiled Gamma-family draw reads the eager draw's span and advances as far.
# XLA evaluates its own log and exp, so values agree to rounding.
_gamma_snapshot(rng, d) = (rand(rng, d), rand_next(rng, d), rand_at(rng, d, 3))

@testset "compiled Gamma family draws track the eager draws" begin
    for d in (
        Gamma(2.5, 2.0),
        Gamma(0.3f0, 1.0f0),
        Chisq(3.0),
        InverseGamma(2.5f0, 1.5f0),
        Beta(0.3, 0.4),
        TDist(3.0f0),
    )
        T = partype(d)
        eager = last(rand_next(Philox4x32(0x654), Bool))
        carrier = Reactant.to_rarray(eager)
        compiled = Reactant.@compile sync = true _gamma_snapshot(carrier, d)
        got = compiled(carrier, d)
        value, next_rng = rand_next(eager, d)
        @test T(got[1]) ≈ value rtol = 100eps(T)
        @test T(got[2][1]) ≈ value rtol = 100eps(T)
        @test _same_value(got[2][2], next_rng)
        @test T(got[3]) ≈ rand_at(eager, d, 3) rtol = 100eps(T)
    end
end

# One candidate sends about 5% of shape-one draws through the traced loop on
# the child stream, which must reach the eager fallback's value.
_forced_gamma(rng, shape) = PureRNGs._traced_gamma(rng, shape, Val(false), 1)

@testset "the compiled Gamma fallback loop reaches the eager value" begin
    base = Philox4x32(0x777)
    compiled = Reactant.@compile sync = true _forced_gamma(Reactant.to_rarray(base), 1.0)
    span = PureRNGs._gamma_span(Float64, 1)
    eager(rng, candidates) = PureRNGs._gamma_value(
        1.0,
        PureRNGs._GammaCodec(1.0, 1.0, rng.device, candidates),
        rng,
        rng.position,
        PureRNGs._gamma_cursor(rng, rng.position),
    )
    rngs = [PureRNGs._addressed_rng(base, span, j) for j = 1:200]
    @test count(rng -> eager(rng, 1) != eager(rng, 8), rngs) >= 3
    for rng in rngs
        @test Float64(compiled(Reactant.to_rarray(rng), 1.0)) ≈ eager(rng, 1) rtol = 1e-12
    end
end
