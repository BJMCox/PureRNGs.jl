const GAMMA_EXTENSION = Base.get_extension(PureRNGs, :PureRNGsDistributionsExt)

# The Marsaglia-Tsang test on one candidate, as the law states it.
function gamma_reference_candidate(x::T, u::T, d::T, c::T) where {T}
    v = one(T) + c * x
    v <= zero(T) && return false, zero(T)
    v = v^3
    accepted = u < one(T) - T(0.0331) * x^4 || log(u) < x^2 / 2 + d * (one(T) - v + log(v))
    return accepted, d * v
end

# A Gamma draw decoded field by field from the stream: the boost uniform, then K
# candidates of a normal and an open uniform.
function gamma_reference(rng, shape::T, scale::T; candidates = 8) where {T}
    n = Int(IR._normal_bits(T))
    field(offset) = begin
        p = IR._advance_position_unchecked(rng, UInt64(offset), UInt64(0))
        IR._extract_bits_unchecked(rng, IR._position_block(p), p.bit, Val(n))
    end
    boosted = shape < one(T)
    d = (boosted ? shape + one(T) : shape) - one(T) / T(3)
    c = inv(sqrt(T(9) * d))
    for k = 1:candidates
        x = IR._normal_from_bits(rng.device, T, field((2k - 1) * n))
        u = IR._open_midpoint(T, field(2k * n))
        accepted, g = gamma_reference_candidate(x, u, d, c)
        accepted || continue
        boosted && (g *= exp(log(IR._open_midpoint(T, field(0))) / shape))
        return scale * g
    end
    return nothing
end

gamma_ks(values, d) = begin
    sorted = sort(values)
    n = length(sorted)
    sqrt(n) * maximum(
        max(abs(cdf(d, sorted[i]) - i / n), abs(cdf(d, sorted[i]) - (i - 1) / n)) for
        i = 1:n
    )
end

@testset "a Gamma draw is the first accepted of K decoded candidates" begin
    for F in (Philox4x32, Threefry4x64, ChaCha),
        T in (Float64, Float32),
        shape in (0.3, 1.0, 2.5, 50.0),
        bit in (0, 5)

        rng = F(0x9c1, bit)
        d = Gamma(T(shape), T(2))
        value, next_rng = rand_next(rng, d)
        @test value === gamma_reference(rng, T(shape), T(2))
        @test rngposition(next_rng) - rngposition(rng) == 17 * IR._normal_bits(T)
        @test rand(rng, d) === value
    end
end

@testset "Gamma fills and addresses equal the chained draws" begin
    d = Gamma(2.5, 2.0)
    for threaded in (false, true)
        rng = Philox4x32(0x9c3, 7)
        values, after = rand_next(rng, d, 1000; threaded)
        chained = rng
        for i = 1:1000
            value, chained = rand_next(chained, d)
            @test value == values[i]
        end
        @test after == chained
        @test rand_at(rng, d, 17) == values[17]
        @test rand!(rng, d, zeros(1000); threaded) == values
    end
end

# Draws below Float32's smallest subnormal round to zero, so small Float32 shapes
# stay at 0.1 and above, where that mass is under 1e-4.
@testset "Gamma draws follow the Gamma distribution" begin
    rng = Philox4x32(0x9c4)
    for T in (Float64, Float32), shape in (0.1, 0.7, 1.0, 3.0, 100.0)
        values = Float64.(rand(rng, Gamma(T(shape), T(2)), 100_000))
        @test gamma_ks(values, Gamma(shape, 2.0)) < 1.95
    end
end

# With one candidate, about 5% of the draws at shape 1 reject it and continue on
# the child stream, so the distribution checks the fallback, and a draw whose
# candidate rejects equals the child continuation.
@testset "rejected candidates continue on a child stream" begin
    codec(shape) = GAMMA_EXTENSION._GammaCodec(shape, 1.0, IR._CPU_BACKEND, 1)
    rng = Philox4x32(0x9c5)
    for shape in (0.3, 1.0, 3.0)
        values = zeros(100_000)
        IR._fill_prevalidated!(rng, values, false, codec(shape))
        @test gamma_ks(values, Gamma(shape, 1.0)) < 1.95
    end
    position = findfirst(
        i ->
            gamma_reference(
                IR._addressed_rng(rng, UInt16(156), i),
                1.0,
                1.0;
                candidates = 1,
            ) === nothing,
        1:100,
    )
    addressed = IR._addressed_rng(rng, UInt16(156), position)
    child = subrng(
        addressed,
        xor(
            GAMMA_EXTENSION._position_index(addressed, addressed.position),
            GAMMA_EXTENSION._GAMMA_TAG,
        ),
    )
    d, c = 1.0 - 1 / 3, inv(sqrt(9 * (1.0 - 1 / 3)))
    expected = nothing
    while expected === nothing
        x, child = randn_next(child, Float64)
        raw, child = rand_next(child, UInt64)
        accepted, g =
            gamma_reference_candidate(x, IR._open_midpoint(Float64, raw >> 12), d, c)
        accepted && (expected = g)
    end
    @test IR._transformed_draw_unchecked(
        codec(1.0),
        addressed,
        addressed.position,
        Float64,
    ) == expected
end
