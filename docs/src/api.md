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

## Normal draws

```@docs
randn_next
randn_next!
randnat
```

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
