# PureRNGs.jl

PureRNGs provides counter-based random numbers with explicit state. Generate arrays on a CPU or GPU, or draw inside your own kernels.

```@example home
using PureRNGs, Random

rng = Philox4x32(123456)
values, next_rng = rand_next(rng, Float32, 4)
@assert values == rand(rng, Float32, 4)
values
```

The draw leaves `rng` unchanged. Use `next_rng` to continue the stream.

## Start here

Read [Getting started](@ref) for the basic workflow.
Then choose a task:

- [Arrays and performance](@ref): fill buffers and control CPU threading.
- [Sampling](@ref): draw from integer ranges or weighted populations.
- [Parallel jobs](@ref): assign stable streams to independent work.
- [GPU kernels](@ref): generate numbers where the computation runs.

## Understand the contract

[Generators and streams](@ref) explains keys, positions, and splitting.
[Reproducibility](@ref) separates stream guarantees from floating-point differences.
[Devices](@ref) lists backend support and placement rules.

[Random interoperability](@ref), [Distributions](@ref), and [Differentiation and compilation](@ref) cover optional integrations.
The [API reference](@ref) lists exported functions.

PureRNGs is not a cryptographic random-number generator.
