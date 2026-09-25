# Random interoperability

## Wrap state for existing Julia code

```@example bridge
using PureRNGs, Random

root = Philox4x32(123456)
rng = StatefulRNG(root)
a = rand(rng, Float64)
b = rand(rng, Float64)
state = parent(rng)

expected, expected_state = rand_next(root, Float64, 2)
@assert [a, b] == expected
@assert state == expected_state
```

`StatefulRNG` implements `Random.AbstractRNG`. Each draw updates its held immutable generator.

The bridge always runs on the host. Construction rebinds a GPU-bound generator to the CPU without changing its key or position.

Use `parent(rng)` to retrieve the current immutable state.
Use `copy(rng)` for an independent mutable wrapper at the same position.
Use `Random.seed!` to reset the same generator type.

## Respect mutable ownership

Do not share one wrapper between parallel tasks.
Give each task its own wrapper around a purpose-derived key.

The bridge supports consumers that use Julia's standard RNG interface.
It does not make their algorithms fixed-work or GPU-compatible.

## Array behavior

Package-owned concrete `Array` fills for uniform, normal, and exponential values preflight their entire span.
Owned Boolean `BitArray` fills preserve one-bit-per-Boolean consumption.

Other destinations can reach foreign `Random` methods.
Those methods may choose different scalar hooks or partially write before exhaustion.
The wrapper still retains a valid state after each completed scalar draw.

Use immutable destination-fill methods when you need their stronger whole-operation contract.

## StatsBase

Loading StatsBase adds `sample`, `sample!`, `wsample`, `wsample!`, and `samplepair` methods for pure generators.
Each form is the [`randsample`](@ref) draw at the held position, with the same `replace` keyword, and it does not advance the generator.

```julia
using StatsBase
rng = Philox4x32(7)
sample(rng, 1:10, 3; replace = false) == randsample(rng, 1:10, 3; replace = false)
sample(rng, [:a, :b, :c], Weights([1.0, 2.0, 3.0]), 5)
```

`ordered = true` draws positions the same way and lists them in population order.
`UnitWeights` take the unweighted law, as in StatsBase.
`sample!` draws the sample, then copies it into the destination, which may have any element type.
`samplepair(rng, n)` makes two range draws: `i` from `1:n`, then `j` from `1:n-1`, with `j == i` standing for `n`.
