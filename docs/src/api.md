# API reference

The manual explains workflows and backend limits. These docstrings describe exported entry points.

## Destination slots

A sampling specification sits in the slot its upstream package uses, and the destination takes the remaining slot.

| Fill | Signature | Follows |
| --- | --- | --- |
| Primitive and range | `rand_next!(rng, dest[, range])` | `Random.rand!` |
| Fixed distribution | `rand_next!(rng, d, dest)` | `Distributions.rand!` |
| Population | `randsample!(rng, pop[, weights], dest)` | `StatsBase.sample!` |

Continuation fills return `(dest, next_rng)`. Standard `Random` draws and fills
require a mutable generator, such as `StatefulRNG`.

## Generators

```@docs
AbstractPureRNG
Philox2x32
Philox4x32
Philox2x64
Philox4x64
Threefry2x32
Threefry4x32
Threefry2x64
Threefry4x64
Philox4x32R7
Philox2x64R6
Philox4x64R7
Threefry2x64R13
Threefry4x32R12
Threefry4x64R13
ChaCha
ChaCha8
ChaCha12
ChaCha20
```

## Continuation draws

```@docs
rand_next
rand_next!
randn_next
randn_next!
randexp_next
randexp_next!
```

## Unified pure and mutable calls

These adapters return `(value, next_rng)` for either kind of generator. Source
arguments precede destinations in `randset!!`, including for distributions.

```@docs
randgen!!
randset!!
randugen!!
randuset!!
randngen!!
randnset!!
randexpgen!!
randexpset!!
```

## Addressed draws

```@docs
rand_at
randn_at
randexp_at
```

## Key derivation

```@docs
splitrng
subrng
```

## State access

```@docs
rngkey
rngposition
```

## Population sampling

```@docs
randsample
randsample_next
randsample!
randsample_next!
WeightTable
```

## Permutations

```@docs
randperm_next
randperm_next!
randcycle_next
randcycle_next!
shuffle_next
shuffle_next!
```

## Mutable interoperability

```@docs
StatefulRNG
```

## Errors

```@docs
StreamExhausted
```
