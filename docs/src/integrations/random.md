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

### Lux and WeightInitializers

`Lux.setup` and WeightInitializers' `glorot_uniform`, `kaiming_normal`, `orthogonal`, and the other initializers take an `AbstractRNG`, so pass the wrapper:

```julia
using WeightInitializers
weights = glorot_uniform(StatefulRNG(Philox4x32(7)), Float32, 4, 3)
```

The initializers' uniform and normal fills then follow the pure stream from the wrapped position.

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

## StaticArrays

Loading StaticArrays adds static array draws for pure generators.
A static array type with `N` elements of type `T` is the next `N` scalar draws of `T`, in linear order, so it equals a length-`N` fill.

```julia
using StaticArrays
rng = Philox4x32(7)
velocity, rng = randn_next(rng, SVector{3,Float32})
position = rand_at(rng, SVector{3,Float32}, 17)
particles, rng = rand_next(rng, SVector{3,Float32}, 1000)
faces = rand(rng, 1:6, SVector{4})
```

`rand`, `randn`, `randexp`, their `_next` and `_at` forms, and the fills accept a static array type.
The scalar forms allocate nothing, so a GPU kernel can call them, and `rand_at(rng, SA, i)` gives each thread its own array without chaining.
An array of static arrays fills as its `reinterpret` to `T`, so it runs on the same CPU and GPU fill paths.
`rand(rng, X, SA)` picks `N` times from a collection or distribution `X`.
The element type must be part of the type: `SVector{3}` alone throws, as an untyped draw does.
