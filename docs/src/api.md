# API reference

## Generator interface

```@docs
AbstractPureRNG
```

`AbstractPureRNG` supports dispatch and inspection only. The eight package
families below are its complete supported direct subtype set.

## Generator families

```@docs
Philox2x32
Philox4x32
Philox2x64
Philox4x64
Threefry2x32
Threefry4x32
Threefry2x64
Threefry4x64
```

## Key derivation

```@docs
splitrng
subrng
```

## Uniform draws

```@docs
rand_next
rand_next!
randat
```

The typed `Random.rand` and `Random.rand!` methods accept `Bool`, `UInt32`,
`Int32`, `UInt64`, `Int64`, `Float32`, and `Float64` results.

## Normal draws

```@docs
randn_next
randn_next!
randnat
```

## Exponential draws

```@docs
randexp_next
randexp_next!
randexpat
```

Use `Random.randexp(rng, T)` for a pure scalar draw,
`Random.randexp(rng, T, dims...)` for an allocating draw, and
`Random.randexp!(rng, destination)` for a destination fill. `T` and the
destination element type must be `Float32` or `Float64`.

All continuation methods return the next generator first. Addressed methods use
one-based indices and do not advance the generator. A pure draw never changes
its input generator.

The immutable API reports contract errors consistently:

- An untyped pure `rand`, `randn`, or `randexp` call throws `ArgumentError`.
- An unsupported result type has no method and throws `MethodError`.
- An invalid address or insufficient counter capacity throws `ArgumentError`.
- A fill on another backend throws `ArgumentError` before mutation.
- A fill with a non-`Bool` `threaded` value throws `TypeError` before mutation.

## Sampling

```@docs
randsample
randsample_next
```

## Mutable bridge

```@docs
StatefulRNG
```

Use `parent(bridge)` to inspect the exact CPU-bound immutable generator
currently held by a `StatefulRNG` bridge without changing either value.
