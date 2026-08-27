# Fixed distributions

Loading Distributions.jl adds direct immutable methods for exactly five
distribution types. Here `T` is exactly `Float32` or `Float64`.

| Distribution | Result | Fixed primitive | Valid parameters |
| --- | --- | --- | --- |
| `Normal{T}` | `T` | `T` normal, 23 or 52 bits | finite `μ`, finite `σ`, `σ >= 0` |
| `Uniform{T}` | `T` | `T` uniform, 24 or 53 bits | finite `a < b`, finite `b - a` |
| `Exponential{T}` | `T` | `T` exponential, 24 or 53 bits | finite `θ > 0` |
| `Bernoulli{T}` | `Bool` | `T` uniform, 24 or 53 bits | finite `0 <= p <= 1` |
| `DiscreteUniform` | `Int` | integer range, 64 or 128 bits | `a <= b` |

No generic distribution fallback exists. Unsupported distribution or parameter
types and wrong destination element types have no method.

## Seven draw forms

Each supported distribution has exactly these forms:

```julia
rand(rng, distribution)
rand_next(rng, distribution)
randat(rng, distribution, index)
rand(rng, distribution, dim1, dims...)
rand_next(rng, distribution, dim1, dims...)
rand!(rng, distribution, destination; threaded=true)
rand_next!(rng, distribution, destination; threaded=true)
```

The following example exercises all seven with one `Normal{Float32}`:

```julia
using Distributions
using PureRNGs
using Random

root = Philox4x32(42)
distribution = Normal(1.0f0, 2.0f0)

pure = rand(root, distribution)
rng, continued = rand_next(root, distribution)
addressed = randat(root, distribution, 1)

pure_batch = rand(root, distribution, 16)
batch_rng, continued_batch = rand_next(root, distribution, 16)

destination = Vector{Float32}(undef, 16)
rand!(root, distribution, destination)
next_destination = similar(destination)
fill_rng, returned = rand_next!(root, distribution, next_destination)

@assert pure == continued == addressed
@assert pure_batch == continued_batch == destination == next_destination
@assert returned === next_destination
@assert rng != root
@assert batch_rng == fill_rng
```

A pure draw leaves `root` unchanged. A continuation returns the advanced
generator first. Addressed indices are one-based and do not advance the
generator. Batches follow native array order and match chained scalar
continuations at the primitive inputs.

## Fixed work and validation

Every result consumes one fixed primitive and applies one typed transform. The
direct methods never call an upstream distribution sampler, use rejection
sampling, retry, cache, or consume a random-dependent span.

Parameters are validated before reservation or generation. Allocating forms
check serviceability and parameters, allocate with Base-compatible dimension
validation, then preflight the full span before generation. Destination forms
validate the `threaded` value, device, serviceability, parameters, and size
before preflight or mutation. Counter exhaustion therefore returns no partial
result and leaves an owned destination unchanged.

## Devices and reproducibility

Allocating and destination forms keep results and intermediate arrays on the
generator device without host staging or CPU fallback. Host-called scalar,
continuation, and addressed forms return host scalars.

Metal excludes every device-executing fixed-distribution allocating and fill
form, including zero-size requests. Host-called scalar, continuation, and
addressed forms remain available for Metal-bound generators.

`Uniform`, `Bernoulli`, and `DiscreteUniform` results are bitwise identical
across execution sites. Final `Normal` results may differ only through their
normal primitive; the final `fma` adds no exception. Final `Exponential`
results may differ only through their exponential primitive; the final
multiplication adds no exception.

## Stateful consumers

The extension adds no `StatefulRNG` method. `rand(bridge, distribution)` uses
Distributions.jl's ordinary sampler through the bridge's primitive hooks. That
foreign composition is not promised to equal the direct immutable mapping.
