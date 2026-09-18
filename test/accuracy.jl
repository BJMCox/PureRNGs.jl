# Accuracy gates for the CPU exponential and normal transforms. Both oracles run
# in BigFloat: a Float64 normal quantile is itself several ulps off in the tail,
# so it cannot bound a few-ulp claim about AS241.
# Set PURERNGS_FULL_ACCURACY_SWEEP=1 to sweep every Float32 normal lattice point.

const _FULL_SWEEP = get(ENV, "PURERNGS_FULL_ACCURACY_SWEEP", "0") == "1"
const _EXPONENTIAL_ULP_BOUND = Dict(Float32 => 2.0, Float64 => 2.5)
# A Float64 lattice point the strided sweep misses, where the transform reaches
# its widest known error.
const _EXPONENTIAL_BREACH_POINT = 0x00097194c32c6bb3
# Worst measured on this machine: 4.813 ulp over every Float32 midpoint, reached
# at u = 0.050660312f0, and 4.840 ulp over the strided Float64 sweep.
const _AS241_ULP_BOUND = Dict(Float32 => 5.0, Float64 => 6.0)
const _AS241_SWEEP_POINTS = 20_000

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
    # AS241 only seeds the iteration; Newton removes its error, and the caller's
    # residual check confirms the result answers the CDF rather than the seed.
    x = BigFloat(Float64(IR._as241(Float64(u))))
    for _ = 1:6
        x -= (_big_normal_cdf(x) - u) * sqrt(2 * big(pi)) * exp(x * x / 2)
    end
    return x
end

# The odd multiples of the uniform draw's quantum. Each one is the midpoint of a
# 2^-p cell, is exact in T, and is never zero.
_as241_midpoint(::Type{T}, k::Integer, p::Int) where {T} = ldexp(T(2k + 1), -(p + 1))

@testset "R63 exponential transform accuracy" begin
    device = IR._CPUBackend()
    for (T, bits) in ((Float32, 24), (Float64, 53))
        sampled =
            _FULL_SWEEP && T === Float32 ? (UInt64(0):UInt64(2^24-1)) :
            (UInt64(1):UInt64(max(1, 2^bits÷500_000)):UInt64(2^bits-1))
        points =
            T === Float64 ?
            Iterators.flatten((sampled, (UInt64(_EXPONENTIAL_BREACH_POINT),))) : sampled
        worst = 0.0
        for k in points
            value = IR._exponential_from_bits(device, T, k)
            worst = max(worst, Float64(_exponential_ulp(value, k, bits)))
        end
        @test worst <= _EXPONENTIAL_ULP_BOUND[T]
    end
end

@testset "R28 AS241 against a BigFloat normal quantile" begin
    setprecision(BigFloat, 160) do
        for (T, p) in ((Float32, 23), (Float64, 52))
            stride = _FULL_SWEEP && T === Float32 ? 1 : (1 << p) ÷ _AS241_SWEEP_POINTS
            explicit = (T(2)^-(T === Float32 ? 24 : 53), T(1e-6), T(0.001), T(0.25), T(0.5))
            worst_ulp = 0.0
            worst_residual = zero(BigFloat)
            for u in Iterators.flatten((
                (_as241_midpoint(T, k, p) for k = 0:stride:((1<<p)-1)),
                explicit,
            ))
                target = BigFloat(u)
                reference = _big_normal_quantile(target)
                worst_residual =
                    max(worst_residual, abs(_big_normal_cdf(reference) - target))
                deviation = abs(BigFloat(IR._as241(u)) - reference)
                worst_ulp = max(worst_ulp, Float64(deviation / eps(T(reference))))
            end
            @test worst_ulp <= _AS241_ULP_BOUND[T]
            @test worst_residual < ldexp(one(BigFloat), -150)
        end
    end
end
