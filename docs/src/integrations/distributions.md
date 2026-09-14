# Distributions

Loading Distributions activates fixed-work methods for a small distribution set.

```julia
using PureRNGs, Random, Distributions

rng = Philox4x32(123456)
distribution = LogNormal(2.0, 0.5)
values, rng = rand_next(rng, distribution, 1024)

buffer = similar(values)
_, rng = rand_next!(rng, distribution, buffer; threaded=false)
x = randat(rng, distribution, 1)
```

## Direct immutable methods

Supported fixed-work distributions are `Normal`, `Uniform`, `Exponential`,
`LogNormal`, `Weibull`, `Rayleigh`, `Laplace`, `Bernoulli`, and
`DiscreteUniform`.

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

The fixed transforms and their spans per result are:

| Distribution | Transform | Bits |
| --- | --- | --- |
| `LogNormal(μ, σ)` | `exp(fma(σ, z, μ))`, with normal `z` | 23 (`Float32`) / 52 (`Float64`) |
| `Weibull(α, θ)` | `θ * x^inv(α)`, with exponential `x` | 24 / 53 |
| `Rayleigh(σ)` | `σ * sqrt(T(2) * x)`, with exponential `x` | 24 / 53 |
| `Laplace(μ, θ)` | `fma(ifelse(b, θ, -θ), x, μ)`, with exponential `x` and Boolean `b` | 25 / 54 |

`LogNormal(μ, 0)` still consumes its normal span. Valid transforms may
underflow or overflow under IEEE arithmetic; PureRNGs neither clips values nor
retries a draw. The exact random bits, order, and continuation agree within an
execution site, while transcendental transforms may differ between CPU, CUDA,
and compiled execution.

CUDA supports these direct methods.
Reactant supports scalar, continuation, and addressed forms for the four new
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
_, rng = rand_next!(rng, distribution, buffer; threaded=false)
label = randat(rng, distribution, 1)
```

Scalar and addressed Categorical calls prepare their cumulative weights on the
CPU and accept only CPU-backed or device-agnostic probabilities. Batch and
destination-fill calls prepare once per call: CPU accepts CPU-backed or
device-agnostic probabilities, while a GPU requires probabilities on the
matching device. A mismatch throws `ArgumentError` rather than copying data.
The scalar preparation can allocate its cumulative distribution. Categorical is
not a Reactant carrier operation and adds no AD support. As with the other
methods, Metal rejects device-executing allocation and fills.

## Other distributions

Use the mutable bridge for Distributions' wider API:

```julia
rng = StatefulRNG(Philox4x32(123456))
values = rand(rng, Gamma(2.0), 1024)
```

This uses Distributions' algorithms, not the package's fixed-work mappings.
Even supported distribution names can produce different streams through the bridge.
