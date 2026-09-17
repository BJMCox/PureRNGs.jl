# Accuracy gates for the CPU exponential and normal transforms. Both oracles run
# in BigFloat: Distributions' own normal quantile is up to 10 ulps off in the
# tail, so it cannot bound a few-ulp claim about AS241.
# Set PURERNGS_FULL_ACCURACY_SWEEP=1 to sweep every Float32 lattice point.

const _FULL_SWEEP = get(ENV, "PURERNGS_FULL_ACCURACY_SWEEP", "0") == "1"
const _EXPONENTIAL_ULP_BOUND = Dict(Float32 => 2.0, Float64 => 2.0)
const _AS241_ULP_BOUND = 4.0

_exponential_ulp(value::T, k, bits) where {T} = setprecision(BigFloat, 160) do
    reference = -log(one(BigFloat) - BigFloat(k) * ldexp(one(BigFloat), -bits))
    denominator = iszero(value) ? BigFloat(floatmin(T)) : BigFloat(eps(value))
    return abs(BigFloat(value) - reference) / denominator
end

# Scaled series for erf. Every term carries the sign of t, so the working
# precision reaches the tail instead of cancelling there.
function _big_erf(t::BigFloat)
    term = 2 * t * exp(-t * t) / sqrt(big(pi))
    total = term
    for n = 1:10_000
        term *= 2 * t * t / (2 * n + 1)
        total += term
        abs(term) > abs(total) * eps(BigFloat) || break
    end
    return total
end

_big_normal_cdf(x::BigFloat) = (one(BigFloat) - _big_erf(-x / sqrt(big(2)))) / 2

function _big_normal_quantile(u::BigFloat)
    # The Float64 quantile only seeds the iteration; Newton removes its error.
    x = BigFloat(quantile(Normal(), Float64(u)))
    for _ = 1:6
        x -= (_big_normal_cdf(x) - u) * sqrt(2 * big(pi)) * exp(x * x / 2)
    end
    return x
end

@testset "R63 exponential transform accuracy" begin
    device = IR._CPUBackend()
    for (T, bits) in ((Float32, 24), (Float64, 53))
        points =
            _FULL_SWEEP && T === Float32 ? (UInt64(0):UInt64(2^24 - 1)) :
            (UInt64(1):UInt64(max(1, 2^bits ÷ 500_000)):UInt64(2^bits - 1))
        worst = 0.0
        for k in points
            value = IR._exponential_from_bits(device, T, k)
            worst = max(worst, Float64(_exponential_ulp(value, k, bits)))
        end
        @test worst <= _EXPONENTIAL_ULP_BOUND[T]
    end
end

@testset "R28 AS241 against a BigFloat normal quantile" begin
    setprecision(BigFloat, 256) do
        for T in (Float32, Float64)
            for u in (T(2)^-(T === Float32 ? 24 : 53), T(1e-6), T(0.001), T(0.25), T(0.5))
                reference = _big_normal_quantile(BigFloat(u))
                deviation = abs(BigFloat(IR._as241(u)) - reference)
                @test deviation <= _AS241_ULP_BOUND * eps(T(reference))
            end
        end
    end
end
