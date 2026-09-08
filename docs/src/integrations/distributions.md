# Distributions

Loading Distributions activates fixed-work methods for a small distribution set.

```julia
using PureRNGs, Random, Distributions

rng = Philox4x32(123456)
distribution = Normal(2.0, 0.5)
rng, values = rand_next(rng, distribution, 1024)

buffer = similar(values)
rng, _ = rand_next!(rng, distribution, buffer; threaded=false)
x = randat(rng, distribution, 1)
```

## Direct immutable methods

Supported distributions are `Normal`, `Uniform`, `Exponential`, `Bernoulli`, and `DiscreteUniform`.

The first four require `Float32` or `Float64` parameters.
Bernoulli returns `Bool`. DiscreteUniform returns `Int`.
Other destinations must match the distribution's result type.

Each distribution supports scalar, addressed, allocating, and destination-fill draws.
Every sampling form also has a continuation form where applicable.

Parameters must be finite. Normal permits zero scale.
Uniform requires increasing bounds with a finite difference.
Exponential requires positive scale. Bernoulli requires a probability in `[0, 1]`.

Degenerate supported cases still consume their fixed random span.

CUDA supports these direct methods.
Metal supports their host scalar use, not device allocation or fills.

## Other distributions

Use the mutable bridge for Distributions' wider API:

```julia
rng = StatefulRNG(Philox4x32(123456))
values = rand(rng, Gamma(2.0), 1024)
```

This uses Distributions' algorithms, not the package's fixed-work mappings.
Even supported distribution names can produce different streams through the bridge.
