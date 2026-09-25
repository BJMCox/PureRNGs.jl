
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
function gamma_reference(
    rng,
    shape::T,
    scale::T;
    candidates = 8,
    logarithm = false,
) where {T}
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
        boost = log(IR._open_midpoint(T, field(0))) / shape
        logarithm && return boosted ? log(g) + boost : log(g)
        boosted && (g *= exp(boost))
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
    codec(shape) = IR._GammaCodec(shape, 1.0, IR._CPU_BACKEND, 1)
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
        xor(IR._position_index(addressed, addressed.position), IR._GAMMA_TAG),
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

gamma_two_sample(a, b) = begin
    a, b = sort(a), sort(b)
    i = j = 1
    distance = 0.0
    while i <= length(a) && j <= length(b)
        x = min(a[i], b[j])
        while i <= length(a) && a[i] <= x
            i += 1
        end
        while j <= length(b) && b[j] <= x
            j += 1
        end
        distance = max(distance, abs((i - 1) / length(a) - (j - 1) / length(b)))
    end
    distance * sqrt(length(a) * length(b) / (length(a) + length(b)))
end

# Each family member is a map of Gamma draws in consecutive spans.
@testset "the Gamma family maps decoded Gamma draws" begin
    for T in (Float64, Float32), bit in (0, 9)
        rng = Philox4x32(0xa02, bit)
        n = Int(IR._normal_bits(T))
        after(bits) = IR._rebuild(
            rng,
            IR._advance_position_unchecked(rng, UInt64(bits), UInt64(0)),
            rng.device,
        )
        span = 17n
        @test rand(rng, Chisq(T(3))) === gamma_reference(rng, T(1.5), T(2))
        @test rand(rng, InverseGamma(T(2.5), T(1.5))) ===
              T(1.5) / gamma_reference(rng, T(2.5), T(1))
        log_x = gamma_reference(rng, T(0.3), T(1); logarithm = true)
        log_y = gamma_reference(after(span), T(0.4), T(1); logarithm = true)
        @test rand(rng, Beta(T(0.3), T(0.4))) === inv(one(T) + exp(log_y - log_x))
        z = IR._normal_from_bits(
            rng.device,
            T,
            IR._extract_bits_unchecked(
                rng,
                IR._position_block(rng.position),
                rng.position.bit,
                Val(n),
            ),
        )
        @test rand(rng, TDist(T(3))) ===
              z * sqrt(T(3) / (T(2) * gamma_reference(after(n), T(1.5), T(1))))
    end
end

@testset "Gamma family fills and addresses equal the chained draws" begin
    for d in (Chisq(3.0), InverseGamma(2.5, 1.5), Beta(0.3, 0.4), TDist(3.0))
        rng = Philox4x32(0xa03, 5)
        values, after = rand_next(rng, d, 200)
        chained = rng
        for i = 1:200
            value, chained = rand_next(chained, d)
            @test value === values[i]
        end
        @test after == chained
        @test rand_at(rng, d, 9) === values[9]
        @test rand(rng, d, 200; threaded = true) == values
    end
end

# Shapes this small put mass within an ulp of 0 and 1, where rounding creates
# ties that a continuous CDF misreads, so they meet Distributions' own sampler.
@testset "Gamma family draws follow their distributions" begin
    rng = Philox4x32(0xa04)
    for d in (
        Chisq(3.0),
        Chisq(0.5),
        InverseGamma(2.5, 1.5),
        Beta(2.0, 5.0),
        Beta(0.3, 0.4),
        TDist(3.0),
        TDist(0.7),
        Chisq(4.0f0),
        Beta(2.0f0, 3.0f0),
    )
        @test gamma_ks(Float64.(rand(rng, d, 100_000)), d) < 1.95
    end
    extreme = Beta(0.02, 0.05)
    @test gamma_two_sample(
        rand(rng, extreme, 100_000),
        rand(Random.Xoshiro(7), extreme, 100_000),
    ) < 1.95
end

# A Dirichlet draw normalizes log-gamma draws in consecutive spans.
@testset "Dirichlet draws normalize consecutive log-gamma draws" begin
    d = Dirichlet([0.3, 1.0, 2.5])
    rng = Philox4x32(0xa05, 3)
    span = 17 * Int(IR._normal_bits(Float64))
    logs = [
        gamma_reference(
            IR._rebuild(
                rng,
                IR._advance_position_unchecked(rng, UInt64((j - 1) * span), UInt64(0)),
                rng.device,
            ),
            shape,
            1.0;
            logarithm = true,
        ) for (j, shape) in enumerate(d.alpha)
    ]
    weights = exp.(logs .- maximum(logs))
    value, next_rng = rand_next(rng, d)
    @test value == weights ./ sum(weights)
    @test rngposition(next_rng) - rngposition(rng) == 3span
    values, after = rand_next(rng, d, 20; threaded = true)
    chained = rng
    for j = 1:20
        column, chained = rand_next(chained, d)
        @test column == values[:, j]
    end
    @test after == chained
    @test rand_at(rng, d, 5) == values[:, 5]
    @test rand!(rng, d, zeros(3, 20)) == values
    draws = rand(Philox4x32(0xa06), d, 50_000)
    for j = 1:3
        marginal = Beta(d.alpha[j], sum(d.alpha) - d.alpha[j])
        @test gamma_ks(draws[j, :], marginal) < 1.95
    end
    small = rand(Philox4x32(0xa07), Dirichlet([0.01, 0.02, 0.03]), 1000)
    @test all(isfinite, small)
    @test maximum(abs.(sum(small; dims = 1) .- 1)) < 1e-12
end
