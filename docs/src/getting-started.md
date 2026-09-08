# Getting started

## Install

The package is under development. For a local checkout, run:

```julia
using Pkg
Pkg.develop(path="/path/to/PureRNGs.jl")
```

Use Julia 1.10 or later. Load `Random` for Julia's standard sampling function names.

## Continue a stream

```@example start
using PureRNGs, Random

rng = Philox4x32(123456)
rng, x = rand_next(rng, Float64)
rng, values = rand_next(rng, Float32, 4)
rng, normal = randn_next(rng, Float32)
rng, exponential = randexp_next(rng, Float32)
values
```

The `_next` functions return `(next_rng, result)`. Assign the returned generator before the next draw.

Without a result type, these continuation functions default to `Float64`. Thus, `rand_next(rng, 4)` draws four values.

## Repeat a draw

Functions without `_next` return only the result. They do not advance the generator.

```@example start
a = rand(rng, Float64, 4)
b = rand(rng, Float64, 4)
@assert a == b
```

Always specify the type for primitive `rand`, `randn`, and `randexp` calls on an immutable generator.

## Choose a result type

Uniform draws support `Bool`, `UInt32`, `Int32`, `UInt64`, `Int64`, `Float32`, and `Float64`.
Integer draws cover the entire type. Uniform floating-point draws lie in `[0, 1)`.

Normal and exponential draws support `Float32` and `Float64`.

```@example start
rng, bits = rand_next(rng, Bool, 8)
rng, integers = rand_next(rng, Int32, 4)
rng, dice = rand_next(rng, 1:6, 8)
dice
```

For existing code that expects a mutable `AbstractRNG`, use [Random interoperability](@ref).
