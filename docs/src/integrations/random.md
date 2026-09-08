# Random interoperability

## Wrap state for existing Julia code

```@example bridge
using PureRNGs, Random

root = Philox4x32(123456)
rng = StatefulRNG(root)
a = rand(rng, Float64)
b = rand(rng, Float64)
state = parent(rng)

expected_state, expected = rand_next(root, Float64, 2)
@assert [a, b] == expected
@assert state == expected_state
```

`StatefulRNG` implements `Random.AbstractRNG`. Each draw updates its held immutable generator.

The bridge always runs on the host. Construction rebinds a GPU-bound generator to the CPU without changing its key or position.

Use `parent(rng)` to retrieve the current immutable state.
Use `copy(rng)` for an independent mutable wrapper at the same position.
Use `Random.seed!` to reset the same generator family.

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
