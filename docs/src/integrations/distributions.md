# Distributions

Loading Distributions activates fixed-work methods for a small distribution set.

```julia
using PureRNGs, Random, Distributions

rng = Philox4x32(123456)
distribution = LogNormal(2.0, 0.5)
values, rng = rand_next(rng, distribution, 1024)

buffer = similar(values)
_, rng = rand_next!(rng, distribution, buffer)
x = rand_at(rng, distribution, 1)
```

## Direct immutable methods

Supported fixed-work distributions are `Normal`, `Uniform`, `Exponential`,
`LogNormal`, `Weibull`, `Rayleigh`, `Laplace`, `Logistic`, `Gumbel`, `Pareto`,
`Frechet`, `Cauchy`, `TriangularDist`, `Bernoulli`, and `DiscreteUniform`.

The continuous distributions and Bernoulli require `Float32` or `Float64` parameters.
Bernoulli returns `Bool`. DiscreteUniform returns `Int`.
Other destinations must match the distribution's result type.

Each distribution supports scalar, addressed, allocating, and destination-fill draws.
Every sampling form also has a continuation form where applicable.

Parameters must be finite. Normal and LogNormal permit zero scale.
Uniform requires increasing bounds with a finite difference.
Exponential and Rayleigh require positive scale. Weibull requires positive shape
and scale. Laplace requires a positive scale. Bernoulli requires a probability
in `[0, 1]`.
Logistic, Gumbel, and Cauchy require positive scales. Pareto and Frechet require
positive shapes and scales. TriangularDist requires `a ≤ c ≤ b` and a finite
width `b-a`; endpoint modes and `a == b == c` are supported.

The fixed transforms and their spans per result are:

| Distribution | Transform | Bits |
| --- | --- | --- |
| `LogNormal(μ, σ)` | `exp(fma(σ, z, μ))`, with normal `z` | 23 (`Float32`) / 52 (`Float64`) |
| `Weibull(α, θ)` | `θ * x^inv(α)`, with exponential `x` | 23 / 52 |
| `Rayleigh(σ)` | `σ * sqrt(T(2) * x)`, with exponential `x` | 23 / 52 |
| `Laplace(μ, θ)` | `fma(ifelse(b, θ, -θ), x, μ)`, with exponential `x` and Boolean `b` | 24 / 53 |
| `Logistic(μ, θ)` | `fma(θ, log(u)-log1p(-u), μ)` | 23 / 52 |
| `Cauchy(μ, σ)` | `fma(σ, tanpi(u-T(0.5)), μ)` | 23 / 52 |
| `Gumbel(μ, θ)` | `fma(-θ, log(e), μ)` | 23 / 52 |
| `Frechet(α, θ)` | `θ * e^(-inv(α))` | 23 / 52 |
| `Pareto(α, θ)` | `θ * exp(x/α)`, with exponential `x` | 23 / 52 |
| `TriangularDist(a, b, c)` | Piecewise inverse CDF | 24 / 53 |

Here `u` uses the same open midpoint grid as normal draws, before the normal
transform. It never equals zero or one. Gumbel and Frechet use `e = -log(1-u)`
with the generator's exponential backend transform, or native compiled math
under Reactant. This avoids singular input endpoints without retries.
TriangularDist uses an ordinary uniform draw and a normalized inverse CDF that
avoids products of interval widths. Even a point mass consumes its full span.

| Precision | Normal cap | Normal mass beyond it | Exponential cap | Mass beyond it |
| --- | --- | --- | --- | --- |
| `Float32` | ±5.2947 | 1.19e-7 (2^-23) | 16.6355 | 5.96e-8 (2^-24) |
| `Float64` | ±8.2095 | 2.22e-16 (2^-52) | 36.7368 | 1.11e-16 (2^-53) |

Exponential draws are strictly positive: the smallest is 5.96e-8 for `Float32`
and 1.11e-16 for `Float64`. For comparison, Base's ziggurat `randn` reaches
13.708 and `randexp` reaches 44.434.

`Float32` normals stop at 5.29 sigma and drop one draw in 8.4 million, so
tail-sensitive `Float32` work should draw `Float64` and convert.

Every derived distribution inherits these bounds through its transform.
`Bernoulli(p)` realizes `ceil(p * 2^w) / 2^w` with `w` equal to 24 or 53, so probabilities below `2^-w` round up to `2^-w`. Use `Float64` parameters for `p` below about `1e-7`.
`Categorical` shares the weighted sampling resolution described in [Sampling](@ref).

```julia
distribution = Logistic(0.0, 1.0)
values, rng = rand_next(rng, distribution, 128)
_, rng = rand_next!(rng, distribution, values)
```

`LogNormal(μ, 0)` still consumes its normal span. Valid transforms may
underflow or overflow under IEEE arithmetic; PureRNGs neither clips values nor
retries a draw. The exact random bits, order, and continuation agree within an
execution site, while transcendental transforms may differ between CPU, CUDA,
and compiled execution.

CUDA supports these direct methods.
AMDGPU serves the same direct methods as CUDA.
Reactant supports scalar, continuation, and addressed forms for these
continuous distributions, using its native compiled arithmetic. This does not
add parameter-gradient support or a general distribution AD guarantee.
Metal supports host scalar use, but rejects device-executing distribution
allocation and fills.

## Categorical

`Categorical` draws return `Int` labels in `1:length(p)`. They use the raw
probability values as weights over those labels and the same 53-bit `Float64`
weighted mapping as `randsample`; there is no extra normalization or alias
table. The probabilities must convert to finite, nonnegative `Float64` values
with a finite, positive left-fold total.

```julia
rng = Philox4x32(123456)
distribution = Categorical([0.1, 0.7, 0.2])
labels, rng = rand_next(rng, distribution, 1024)

buffer = similar(labels)
_, rng = rand_next!(rng, distribution, buffer)
label = rand_at(rng, distribution, 1)
```

Scalar and addressed Categorical calls prepare their cumulative weights on the
CPU and accept only CPU-backed or device-agnostic probabilities. Batch and
destination-fill calls prepare once per call: CPU accepts CPU-backed or
device-agnostic probabilities, while a GPU requires probabilities on the
matching device. A mismatch throws `ArgumentError` rather than copying data.
The scalar preparation can allocate its cumulative distribution. Categorical is
not a Reactant carrier operation and adds no AD support. As with the other
methods, Metal rejects device-executing allocation and fills.

## MvNormal

An `MvNormal` draw applies Distributions' own map, `μ + L z`, to the next `length(d)` standard normal draws, where `L` is the covariance's lower factor.
`n` draws are one normal fill, a column per draw, so a matrix of draws equals the chained single draws.

```julia
rng = Philox4x32(123456)
distribution = MvNormal([1.0, -2.0], [2.0 0.3; 0.3 1.0])
point, rng = rand_next(rng, distribution)
cloud, rng = rand_next(rng, distribution, 1000)
seventh = rand_at(rng, distribution, 7)
```

`PDMat`, diagonal, and scalar covariances work on every backend.
A device draw moves the covariance's factor to the device once per call and applies it there.
Gradients with respect to `μ` and the covariance are pathwise with Enzyme, Mooncake, and ForwardDiff.

## Gamma and its family

`Gamma`, `Chisq`, `InverseGamma`, `Beta`, `TDist`, and `Dirichlet` draws use the rejection sampler of Marsaglia and Tsang (2000) with a fixed stream span.
Each Gamma draw reserves one boost uniform and eight candidates, each a normal and an open uniform, and takes the first candidate the test accepts.
If all eight reject, which happens about once in 3·10¹⁰ draws at shape 1 and far less often at larger shapes, the draw continues the same test on a child stream keyed by its position.
The law stays exactly Gamma, and the stream advances by the same 17 normal widths either way.
Shapes below one draw `Gamma(shape + 1)` and multiply by `u^(1/shape)` from the boost uniform.

```julia
rng = Philox4x32(123456)
waits, rng = rand_next(rng, Gamma(2.5, 2.0), 1024)
proportion = rand_at(rng, Beta(0.3, 0.4), 7)
mixture, rng = rand_next(rng, Dirichlet([0.3, 1.0, 2.5]))
```

`Chisq(ν)` is `Gamma(ν/2, 2)`, and `InverseGamma(α, θ)` is `θ` over a `Gamma(α)` draw.
`Beta` and `Dirichlet` normalize the logarithms of Gamma draws in consecutive spans, so shapes as small as 0.01 give finite draws that sum to one.
`TDist(ν)` divides a normal by the square root of a following chi-square over `ν`.
Every member runs on the CPU, CUDA, and AMDGPU; a 2^26 `Gamma` fill takes about 6.7 ms on an A100. A device `Dirichlet` fill runs one work item per draw.

Shape gradients use the implicit derivative of the Gamma CDF, `dx/dα = -∂F(x; α)/∂α ÷ f(x; α)` (Figurnov, Mohamed, and Mnih 2018), with Enzyme, Mooncake, and ForwardDiff alike.
Differentiating through the rejection test would bias the gradient, so the sampler is a primitive with that rule.
The derivative differentiates the series and continued fraction of the incomplete gamma function term by term, in the shape's type, so it also runs in device kernels.
Its cost grows with the square root of the shape near the mode: about 40 ns at shape 2.5 and 0.6 µs at shape 10⁴ on the CPU.
In Float32 it is accurate to about 10⁻⁶ up to shape 100 and 10⁻⁴ at shape 10⁴.

## Other distributions

Use the mutable bridge for Distributions' wider API:

```julia
rng = StatefulRNG(Philox4x32(123456))
values = rand(rng, Poisson(3.0), 1024)
```

This uses Distributions' algorithms, not the package's fixed-work mappings.
Even supported distribution names can produce different streams through the bridge.
