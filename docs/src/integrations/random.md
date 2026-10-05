# Random interoperability

## Write code for pure or mutable generators

Use the `!!` adapter when code must accept either kind of generator. Always
rebind both returned values:

```@example unified
using PureRNGs, Random

function draw_and_fill(rng)
    die, rng = randgen!!(rng, 1:6)
    values, rng = randngen!!(rng, 4)
    values, rng = randnset!!(rng, values)
    return die, values, rng
end

pure_result = draw_and_fill(Philox4x32(7))
mutable_result = draw_and_fill(Xoshiro(7))
@assert 1 <= pure_result[1] <= 6
@assert length(mutable_result[2]) == 4
```

`randgen!!(rng, source, dims...)` accepts types, populations, and supported
distributions. `randset!!(rng, source, old_value)` puts the source before the
destination for all three. It reuses writable arrays and replaces scalars,
ranges, and immutable StaticArrays. A type source can replace an array with a
different element type. Views with writable storage remain reusable.
A static array type with a static array template denotes one whole sample:
`randset!!(rng, SVector{3,Float64}, old_static_vector)`.

`randugen!!`/`randuset!!`, `randngen!!`/`randnset!!`, and
`randexpgen!!`/`randexpset!!` serve uniform, standard normal, and standard
exponential draws without a source argument. Generators accept an optional
result type, defaulting to `Float64`. Setters use the destination's existing
element type. Tuple arguments to the primitive generators mean dimensions:
`randugen!!(rng, (2, 3))` makes a matrix, while
`randgen!!(rng, (2, 3))` picks either `2` or `3`.

Nonempty tuple dimensions are equivalent to separate dimensions. For example,
`randgen!!(rng, d, (n,))` and `randgen!!(rng, d, n)` both use the native `n` layout,
even for a multivariate distribution. Empty tuples keep the native zero-dimensional
array form where supported. Supported keywords follow the native methods. In
particular, distribution fills keep their multivariate rules: a vector is one
sample and matrix columns hold separate samples. Arrays of immutable samples
use existing bulk methods. Arbitrary tuples and structs are single values,
not recursively filled containers.

Treat the old RNG and destination as consumed in generic code. The returned RNG
may be the same mutable object or a new pure value. Pure snapshots remain valid.
The adapter does not convert through `StatefulRNG`, move data to the host, or
change any native stream. Existing device checks and bulk kernels still apply.

The source stays unchanged. If a destination aliases a supported source, the
adapter allocates replacement storage before drawing. Population elements may
still be borrowed references. Unknown sources with references use replacement
storage conservatively, without scanning their contents. Native mutable-RNG fills
of arrays of array-valued distribution samples replace each inner sample, so
they cannot overwrite a borrowed or immutable sample. A custom source must
preserve this ownership contract in its own methods. Failed native calls are not retried and can leave
mutable state advanced or a destination partly written.

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
The element type must be part of the static array type: use `SVector{3,Float64}`, not `SVector{3}`.
