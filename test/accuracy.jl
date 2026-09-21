# Accuracy gates for the exponential and normal transforms. Both oracles run in
# BigFloat: a Float64 normal quantile is itself several ulps off in the tail, so
# it cannot bound a few-ulp claim about either normal transform. The normal
# sweeps run both transforms [R28] selects between; each is plain Julia here, so
# the token is only a dispatch tag and the host evaluates both.
# Set PURERNGS_FULL_ACCURACY_SWEEP=1 to sweep every Float32 normal lattice point.

const _FULL_SWEEP = get(ENV, "PURERNGS_FULL_ACCURACY_SWEEP", "0") == "1"
# Worst measured on this machine: 1.649 ulp over every Float32 midpoint, reached
# at k = 0x26178e, and 2.021 ulp over the Float64 midpoints, found by sampling
# 24 million points across the lattice and 20 million in the band that holds the
# peak. Both bounds carry more than 0.5 ulp of margin.
const _EXPONENTIAL_ULP_BOUND = Dict(Float32 => 2.5, Float64 => 3.0)
# A Float64 lattice point the strided sweep misses, where the transform reaches
# its widest known error.
const _EXPONENTIAL_BREACH_POINT = 0x0004b2925a3b52a5
# [R28] gates both normal transforms at 5.0 ulp for Float32 and 6.0 for Float64.
const _NORMAL_ULP_BOUND = Dict(Float32 => 5.0, Float64 => 6.0)
const _NORMAL_SWEEP_POINTS = 20_000
const _NORMAL_MONOTONE_POINTS = 1_000_000
# The two transforms the backend token selects between.
const _NORMAL_TOKENS = (IR._CPU_BACKEND, IR._CUDA_BACKEND)
# Lattice points the strided sweep steps over, where each transform reaches its
# widest measured error. Order follows `_NORMAL_TOKENS`: AS241 4.813 and 5.502,
# Giles 4.175 and 4.794.
const _NORMAL_BREACH_POINTS =
    Dict(Float32 => (0x7983f6, 0x28e400), Float64 => (0x000af7af48e12fde, 0x001263de))

_exponential_ulp(value::T, k, bits) where {T} = setprecision(BigFloat, 160) do
    u = BigFloat(2k + 1) * ldexp(one(BigFloat), -(bits + 1))
    reference = -log(one(BigFloat) - u)
    return abs(BigFloat(value) - reference) / BigFloat(eps(value))
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
_normal_lattice_point(::Type{T}, k::Integer, p::Int) where {T} = ldexp(T(2k + 1), -(p + 1))

# [R28] defines both normal transforms on the midpoint lattice only. Off the
# lattice `T(2) * u - one(T)` stops being exact and Giles loses its tail
# argument, so every probe below is snapped to the nearest lattice point.
_nearest_lattice_index(u::Real, p::Int) = max(0, round(Int, (u * (1 << (p + 1)) - 1) / 2))

@testset "R63 exponential transform accuracy" begin
    device = IR._CPUBackend()
    for (T, bits) in ((Float32, 23), (Float64, 52))
        sampled =
            _FULL_SWEEP && T === Float32 ? (UInt64(0):UInt64(2^23-1)) :
            (UInt64(0):UInt64(max(1, 2^bits÷500_000)):UInt64(2^bits-1))
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

@testset "R28 normal transforms against a BigFloat normal quantile" begin
    setprecision(BigFloat, 160) do
        for (T, p) in ((Float32, 23), (Float64, 52))
            stride = _FULL_SWEEP && T === Float32 ? 1 : (1 << p) ÷ _NORMAL_SWEEP_POINTS
            explicit = (
                _normal_lattice_point(T, _nearest_lattice_index(u, p), p) for
                u in (0.0, 1.0e-6, 0.001, 0.25, 0.5)
            )
            worst_ulp = zeros(Float64, length(_NORMAL_TOKENS))
            worst_residual = zero(BigFloat)
            for u in Iterators.flatten((
                (_normal_lattice_point(T, k, p) for k = 0:stride:((1<<p)-1)),
                explicit,
                (_normal_lattice_point(T, k, p) for k in _NORMAL_BREACH_POINTS[T]),
            ))
                target = BigFloat(u)
                reference = _big_normal_quantile(target)
                worst_residual =
                    max(worst_residual, abs(_big_normal_cdf(reference) - target))
                scale = BigFloat(eps(T(reference)))
                for index in eachindex(_NORMAL_TOKENS)
                    value = IR._normal_transform(_NORMAL_TOKENS[index], u)
                    deviation = abs(BigFloat(value) - reference)
                    worst_ulp[index] = max(worst_ulp[index], Float64(deviation / scale))
                end
            end
            @test maximum(worst_ulp) <= _NORMAL_ULP_BOUND[T]
            @test worst_residual < ldexp(one(BigFloat), -150)
        end
    end
end

@testset "R28 normal transforms are monotone over the midpoint lattice" begin
    # A quantile that steps backwards would break the coupling between the
    # uniform draw and the normal it maps to, which no ulp bound would catch.
    for (T, p) in ((Float32, 23), (Float64, 52))
        stride = (1 << p) ÷ _NORMAL_MONOTONE_POINTS
        for token in _NORMAL_TOKENS
            breaks = 0
            previous = T(-Inf)
            for k = 0:stride:((1<<p)-1)
                value = IR._normal_transform(token, _normal_lattice_point(T, k, p))
                value < previous && (breaks += 1)
                previous = value
            end
            @test breaks == 0
        end
    end
end
